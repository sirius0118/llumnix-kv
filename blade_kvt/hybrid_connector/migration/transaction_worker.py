# ruff: noqa: PLC0415 -- lazy vLLM imports avoid connector initialization cycles
"""Worker-local KV shards and RNG state; no full KV gather or host staging."""

from __future__ import annotations


class WorkerTransactions:
    def __init__(self, worker):
        import torch
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        from blade_kvt import kv_transfer as blade

        from ..kvtbackend import kvt_protocol

        if kvt_protocol() != "rdma":
            raise ValueError("AsyncPB request migration requires the RDMA transport")
        self.worker, self.runner = worker, worker.model_runner
        self.torch, self.blade = torch, blade
        self.backend = get_kv_transfer_group()._worker._backend
        self.client = self.backend._p._bladkv_cli
        self.server = self.backend._d._bladkv_srv
        self.coordinate = {
            "dp": worker.vllm_config.parallel_config.data_parallel_rank,
            "pp": get_pp_group().rank_in_group,
            "tp": get_tp_group().rank_in_group,
        }
        groups = self.backend._p._kv_cache_config.kv_cache_groups
        self.descriptor = {
            "coordinate": self.coordinate,
            "instance": self.backend._p._inst_id,
            "worker": get_tp_group().rank,
            "info": blade.current_worker_info("server"),
            "cache_groups": [list(group.layer_names) for group in groups],
        }
        self.sends = {}
        self.runner._asyncpb_rng_states = {}

    def describe(self, **unused):
        return self.descriptor

    def fence(self, **unused):
        for work in self.worker._pp_send_work:
            work.wait()
        self.worker._pp_send_work.clear()
        self.torch.cuda.synchronize()
        return True

    def checkpoint(self, request_id):
        self.fence()
        request = self.runner.requests.get(request_id)
        # Non-final PP stages do not sample, but still acknowledge their shard.
        generator = request.generator if request is not None else None
        routing = None
        if self.runner.routed_experts_initialized:
            from vllm.model_executor.layers.fused_moe.routed_experts_capturer import get_global_experts_capturer

            capturer = get_global_experts_capturer()
            if hasattr(capturer, "finalize_pending_copy"):
                capturer.finalize_pending_copy()
            routing = capturer.get_routed_experts(request_id, free_slot=False)
        return {
            "coordinate": self.coordinate,
            "routing": routing,
            "rng": bytes(generator.get_state().tolist()) if generator is not None else None,
        }

    def _peer(self, peers):
        matches = [
            p
            for p in peers
            if (p["coordinate"]["pp"], p["coordinate"]["tp"]) == (self.coordinate["pp"], self.coordinate["tp"])
        ]
        if len(matches) != 1:
            raise ValueError("Incomplete or ambiguous PP/TP worker mapping")
        if "cache_groups" in matches[0] and matches[0]["cache_groups"] != self.descriptor["cache_groups"]:
            raise ValueError("Worker stage/cache group layer mapping differs")
        return matches[0]

    def restore(self, request_id, checkpoints):
        checkpoint = self._peer(checkpoints)
        existing = self.runner._asyncpb_rng_states.get(request_id)
        if existing is not None and existing != checkpoint["rng"]:
            raise ValueError("Conflicting RNG checkpoint")
        self.runner._asyncpb_rng_states[request_id] = checkpoint["rng"]
        routing = checkpoint.get("routing")
        if routing is not None:
            from vllm.model_executor.layers.fused_moe.routed_experts_capturer import get_global_experts_capturer

            cache = get_global_experts_capturer().get_host_cache()
            if cache is None:
                raise ValueError("Target worker lacks routed-experts host state")
            cache.get_or_grow_buffer(request_id, len(routing) - 1)[: len(routing)] = routing
            cache.update_filled_len(request_id, len(routing) - 1)
        return self.coordinate

    def send(self, transfer_id, source, target):
        peer = self._peer(target["workers"])
        identity = (source["computed"], source.get("seen", 0), source["blocks"], target["blocks"], peer)
        if transfer_id in self.sends:
            if self.sends[transfer_id] != identity:
                raise ValueError("Conflicting native transfer retry")
            return True
        seen = source.get("seen", 0)
        meta = self.blade.ReqMeta()
        meta.dst_inst, meta.dst_worker = peer["instance"], peer["worker"]
        meta.dst_worker_info = peer["info"]
        meta.reqid = transfer_id
        meta.seen_tokens = seen
        meta.new_tokens = source["computed"] - seen
        meta.src_block_ids, meta.dst_block_ids = source["blocks"][0], target["blocks"][0]
        if meta.new_tokens < 0:
            raise ValueError("KV computed prefix moved backwards")
        self.fence()  # Producer CUDA writes must be visible before the NIC reads.
        self.sends[transfer_id] = identity
        if meta.new_tokens:
            self.client.start_req_send([meta])
        return True

    def forget(self, request_id):
        self.fence()
        self.runner._asyncpb_rng_states.pop(request_id, None)
        self.runner.requests.pop(request_id, None)
        self.runner.input_batch.remove_request(request_id)
        self.runner.num_prompt_logprobs.pop(request_id, None)
        if self.runner.routed_experts_initialized:
            from vllm.model_executor.layers.fused_moe.routed_experts_capturer import free_routing_buffers

            free_routing_buffers({request_id}, set())
        return True

    def retire(self, migration_id):
        # Only called after every worker's DMA completion reached the core.
        for phase in ("prefix", "delta"):
            self.sends.pop(migration_id + "/" + phase, None)
        return True


def worker_rpc(worker, operation, payload):
    if not hasattr(worker, "_asyncpb_tail_worker"):
        worker._asyncpb_tail_worker = WorkerTransactions(worker)
    return getattr(worker._asyncpb_tail_worker, operation)(**payload)


def restore_generator(runner, request_id, generator):
    states = getattr(runner, "_asyncpb_rng_states", {})
    if request_id not in states:
        return
    state = states.pop(request_id)
    if state is None:
        return
    if generator is None:
        raise RuntimeError("Seeded sampling checkpoint has no destination generator")
    import torch

    generator.set_state(torch.tensor(list(state), dtype=torch.uint8, device="cpu"))
