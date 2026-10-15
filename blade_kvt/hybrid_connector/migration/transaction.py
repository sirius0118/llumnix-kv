# ruff: noqa: PLC0415 -- lazy vLLM imports avoid connector initialization cycles
"""Reversible EngineCore migration primitives for the AsyncPB coordinator.

All methods run on the EngineCore utility queue, never the HTTP/RPC thread.
This is separate from the legacy suspend API, which aborts the source request.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math

import cloudpickle


def encode(value):
    return base64.b64encode(cloudpickle.dumps(value)).decode("ascii")


def decode(value):
    return cloudpickle.loads(base64.b64decode(value))


class CoreTransactions:
    def __init__(self, core):
        self.core = core
        self.scheduler = core.scheduler
        self.manager = self.scheduler.kv_cache_manager
        self.transactions = {}
        self.sends = {}
        self.updating = False
        self.version = None
        self.incarnation = None
        self.tracked = {}
        self.admissions = {}
        self.accepting = False
        self.scheduler._asyncpb_new_req_ids = set()
        config = core.vllm_config
        if config.scheduler_config.async_scheduling or config.speculative_config:
            raise ValueError("Transactional migration requires synchronous scheduling without speculative decode")
        if getattr(config.model_config, "is_multimodal_model", False):
            raise ValueError("Multimodal request state is not supported by this adapter")
        groups = self.scheduler.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            # The native Blade transfer registration currently shares block IDs
            # across layers. Reject instead of copying one group's IDs to all.
            raise ValueError("BladeKV transactional transport needs per-cache-group native registrations")
        self.block_sizes = tuple(g.kv_cache_spec.block_size for g in groups)
        signature = {
            "model": config.model_config.hf_config.to_dict(),
            "dtype": str(config.model_config.dtype),
            "cache": str(groups),
            "tp": config.parallel_config.tensor_parallel_size,
            "pp": config.parallel_config.pipeline_parallel_size,
            "ep": config.parallel_config.enable_expert_parallel,
            "dp": config.parallel_config.data_parallel_size,
        }
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
        self.worker_info = self.workers("describe")

    def workers(self, operation, **payload):
        return self.core.collective_rpc("asyncpb_tail_worker", args=(operation, payload))

    def configure(self, version, incarnation):
        if not self.idle():
            raise RuntimeError("Cannot change weights with native requests or migration references")
        self.workers("fence")
        self.version, self.incarnation = version, incarnation
        self.updating = False
        return self.describe()

    def idle(self):
        return not (
            self.scheduler.requests
            or self.admissions
            or self.core.batch_queue
            or any(not tx.get("retired") for tx in self.transactions.values())
        )

    def claim_update(self):
        if not self.idle():
            return False
        self.updating = True
        return True

    def set_accepting(self, accepting):
        self.accepting = accepting
        return self.describe()

    def describe(self):
        return {
            "version": self.version,
            "incarnation": self.incarnation,
            "signature": self.signature,
            "block_sizes": self.block_sizes,
            "free_blocks": self.manager.block_pool.get_num_free_blocks(),
            "running": len(self.scheduler.running),
            "max_sequences": self.scheduler.scheduler_config.max_num_seqs,
            "workers": self.worker_info,
            "idle": self.idle(),
            "accepting": self.accepting,
            "updating": self.updating,
            "pending_parents": list(self.admissions),
            "admitted": sum(self.admissions.values()),
            "reserved": sum(bool(tx.get("slot")) for tx in self.transactions.values()),
            "requests": [
                {
                    "request_id": req.request_id,
                    "parent": self.tracked.get(req.request_id, {}).get("parent"),
                    "generation": self.tracked.get(req.request_id, {}).get("generation", 0),
                    "blocks": sum(
                        math.ceil(min(req.num_prompt_tokens + req.max_tokens, self.scheduler.max_model_len) / size)
                        for size in self.block_sizes
                    ),
                    "migratable": self._can_move(req),
                }
                for req in self.scheduler.requests.values()
            ],
        }

    def admit(self, parent, count, version):
        if parent in self.admissions:
            return True
        if self.updating or not self.accepting or version != self.version or count < 1:
            return False
        occupied = (
            len(self.scheduler.requests)
            + sum(self.admissions.values())
            + sum(bool(tx.get("slot")) for tx in self.transactions.values())
        )
        limit = self.scheduler.scheduler_config.max_num_seqs
        if occupied + min(count, limit) > limit:
            return False
        self.admissions[parent] = count
        return True

    def end_admission(self, parent):
        self.admissions.pop(parent, None)
        return True

    def on_add(self, request):
        params = (request.sampling_params.extra_args or {}) if request.sampling_params else {}
        if txid := params.get("llx_migration_id"):
            tx = self.transactions[txid]
            if tx.get("activated"):
                return True
            if not tx.get("ready") or tx.get("retired"):
                raise RuntimeError("Migration target is not ready")
            checkpoint = tx["request"]
            checkpoint.client_index = request.client_index
            checkpoint._block_hasher = request._block_hasher
            self._activate(tx, checkpoint)
            return True
        parent = params.get("llx_parent")
        if parent is None or self.admissions.get(parent, 0) <= 0:
            raise RuntimeError("Request has no native admission reservation")
        self.admissions[parent] -= 1
        if not self.admissions[parent]:
            del self.admissions[parent]
        self.track(request.request_id, parent=parent)
        self.scheduler.add_request(request)
        return True

    def cancel(self, request):
        if request["request_id"] not in self.scheduler.requests:
            return True
        if request["version"] != self.version:
            raise RuntimeError("Cannot cancel a request against another weight version")
        from vllm.v1.request import RequestStatus

        request_id = request["request_id"]
        self.scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
        if any(request_id in output.num_scheduled_tokens for _, output, _ in self.core.batch_queue or ()):
            return None
        self.workers("forget", request_id=request_id)
        self.tracked.pop(request_id, None)
        return True

    def track(self, request_id, generation=0, offset=0, parent=None):
        value = {"generation": generation, "offset": offset, "parent": parent}
        if request_id in self.tracked and self.tracked[request_id] != value:
            raise RuntimeError("Conflicting output owner")
        self.tracked[request_id] = value
        return True

    def capture_outputs(self, outputs):
        for batch in (outputs or {}).values():
            for output in batch.outputs:
                state = self.tracked.get(output.request_id)
                if state is None:
                    continue
                if not output.queue_server_address:
                    raise RuntimeError("Migrating request lost its output address")
                output.migration_offset = state["offset"]
                output.migration_generation = state["generation"]
                state["offset"] += len(output.new_token_ids)
                if output.finished:
                    self.tracked.pop(output.request_id, None)

    def _tx(self, metadata, side):
        previous = self.transactions.get(metadata["migration_id"])
        if previous is not None:
            if previous["metadata"] != metadata:
                raise RuntimeError("Migration ID was reused")
            return previous
        expected = metadata[side + "_incarnation"]
        if self.incarnation != expected or self.version != metadata["version"]:
            raise RuntimeError("Stale migration incarnation or weight version")
        key = metadata["migration_id"]
        tx = self.transactions.setdefault(key, {"metadata": copy.deepcopy(metadata), "pins": [], "retired": False})
        # A migration ID has one immutable identity.
        identity = (
            "request_id",
            "source",
            "target",
            "source_incarnation",
            "target_incarnation",
            "owner_generation",
            "version",
        )
        if any(tx["metadata"][field] != metadata[field] for field in identity):
            raise RuntimeError("Migration ID was reused")
        return tx

    def _can_move(self, request):
        return (
            request in self.scheduler.running
            and not request.is_finished()
            and request.request_id not in self.scheduler._asyncpb_new_req_ids
            and bool(request.output_token_ids)
            and (request.sampling_params.temperature == 0 or request.sampling_params.seed is not None)
            and not (request.use_structured_output or request.resumable or request.lora_request or request.mm_features)
        )

    def _live(self, metadata):
        request = self.scheduler.requests.get(metadata["request_id"])
        if request is None or request.is_finished():
            raise ValueError("Source request has already finished")
        if request.use_structured_output or request.resumable or request.lora_request or request.mm_features:
            raise ValueError("Unsupported structured/streaming/LoRA request state")
        if request not in self.scheduler.running:
            raise ValueError("Source is not a running decode request")
        state = self.tracked.get(request.request_id)
        if state is not None and state["generation"] != metadata["owner_generation"]:
            raise RuntimeError("Stale request owner")
        return request

    def _snapshot(self, tx, request, stable):
        # Optimistic scheduler counters can include not-yet-computed tokens.
        computed = min(request.num_computed_tokens, request.num_tokens - 1)
        for _, output, _ in self.core.batch_queue or ():
            computed -= output.num_scheduled_tokens.get(request.request_id, 0)
        computed = max(computed, 0)
        if stable:
            alignment = math.lcm(*self.block_sizes)
            computed = computed // alignment * alignment
        blocks = self.manager.get_blocks(request.request_id)
        pinned = [block for group in blocks.blocks for block in group if not block.is_null]
        self.manager.block_pool.touch(pinned)
        tx["pins"].extend(pinned)
        snapshot = copy.deepcopy(request)
        snapshot.num_computed_tokens = computed
        return {
            "request": encode(snapshot),
            "computed": computed,
            "blocks": blocks.get_block_ids(),
            "signature": self.signature,
            "workers": self.worker_info,
            "parent": self.tracked.get(request.request_id, {}).get("parent"),
        }

    def export_prefix(self, metadata, **unused):
        if metadata["request_id"] not in self.scheduler.requests:
            return None
        tx = self._tx(metadata, "source")
        if tx.get("retired"):
            return None
        if "prefix" not in tx:
            request = self.scheduler.requests[metadata["request_id"]]
            if not self._can_move(request):
                self._retire(tx)
                return None
            self._live(metadata)
            tx["prefix"] = self._snapshot(tx, request, stable=True)
        return tx["prefix"]

    def reserve(self, metadata, source):
        if metadata["version"] != self.version:
            return None  # The idle target may have installed newer weights.
        tx = self._tx(metadata, "target")
        if "reservation" in tx:
            return tx["reservation"]
        if tx.get("retired"):
            return None
        if source["signature"] != self.signature:
            raise ValueError("Native model/cache/parallel layouts differ")
        occupied = (
            len(self.scheduler.requests)
            + sum(self.admissions.values())
            + sum(bool(t.get("slot")) for t in self.transactions.values())
        )
        if self.updating or occupied >= self.scheduler.scheduler_config.max_num_seqs:
            tx["retired"] = True
            return None
        request = decode(source["request"])
        if request.request_id in self.scheduler.requests:
            raise ValueError("Target request ID already exists")
        request.num_computed_tokens = 0
        extent = min(request.num_prompt_tokens + request.max_tokens, self.scheduler.max_model_len)
        allocation = self.manager.allocate_slots(request, extent, delay_cache_blocks=True)
        if allocation is None:
            tx["retired"] = True
            return None
        # Native scheduling sees this reservation immediately. Ready requests
        # remain outside every runnable/waiting queue until owner commit.
        tx["request"] = request
        tx["slot"] = True
        tx["reservation"] = {
            "blocks": self.manager.get_blocks(request.request_id).get_block_ids(),
            "workers": self.worker_info,
        }
        return tx["reservation"]

    def reservation(self, metadata, **unused):
        return self._tx(metadata, "target")["reservation"]

    def _start(self, metadata, which, target):
        tx = self._tx(metadata, "source")
        if tx.get(which + "_started"):
            return True
        snapshot = tx[which]
        if which == "delta":
            snapshot["seen"] = tx["prefix"]["computed"]
        # The transport journals before submission and never restarts an
        # uncertain send under a new operation ID.
        self.sends.setdefault(metadata["migration_id"] + "/" + which, {})
        self.workers("send", transfer_id=metadata["migration_id"] + "/" + which, source=snapshot, target=target)
        tx[which + "_started"] = True
        return True

    def start_prefix(self, metadata, target):
        return self._start(metadata, "prefix", target)

    def start_delta(self, metadata, target):
        return self._start(metadata, "delta", target)

    def record_done(self, worker, result):
        receipts = self.sends.get(result.reqid)
        if receipts is None:
            tx = self.transactions.get(result.reqid.rsplit("/", 1)[0])
            return bool(tx and tx.get("retired"))
        receipts.setdefault(worker, result)
        return True

    def _done(self, metadata, which):
        tx = self._tx(metadata, "source")
        if tx.get("retired"):
            return True
        snapshot = tx[which]
        if snapshot["computed"] == snapshot.get("seen", 0):
            return True
        receipts = self.sends.get(metadata["migration_id"] + "/" + which, {})
        if len(receipts) != len(self.worker_info):
            return None
        for result in receipts.values():
            if result.ex is not None:
                raise RuntimeError("Native transfer failed; keep both memory allocations pinned") from result.ex
            if result.n < snapshot["computed"]:
                raise RuntimeError("Incomplete native transfer receipt")
        return True

    def prefix_done(self, metadata, **unused):
        return self._done(metadata, "prefix")

    def delta_done(self, metadata, **unused):
        return self._done(metadata, "delta")

    def park(self, metadata, **unused):
        tx = self._tx(metadata, "source")
        if tx.get("parked"):
            return True
        if not self.core.is_scheduler_paused() or self.core.batch_queue:
            raise RuntimeError("Park requires the whole communication group to finish keep-pause")
        request = self.scheduler.requests.get(metadata["request_id"])
        if request is None or request.is_finished() or request not in self.scheduler.running:
            return False
        self._live(metadata)
        self.workers("fence")
        self.scheduler.running.remove(request)
        tx["parked"] = True
        return True

    def checkpoint(self, metadata, **unused):
        tx = self._tx(metadata, "source")
        if "delta" in tx:
            return tx["delta"]
        if not tx.get("parked"):
            raise RuntimeError("Checkpoint requires a parked source")
        request = self.scheduler.requests.get(metadata["request_id"])
        if request is None or request.is_finished():
            raise ValueError("Source finished before park")
        for _, output, _ in self.core.batch_queue or ():
            if request.request_id in output.num_scheduled_tokens:
                return None
        self.workers("fence")
        tx["removed"] = True
        tx["delta"] = self._snapshot(tx, request, stable=False)
        tx["delta"]["seen"] = tx["prefix"]["computed"]
        tx["delta"]["rng"] = self.workers("checkpoint", request_id=request.request_id)
        return tx["delta"]

    def install_checkpoint(self, metadata, checkpoint):
        tx = self._tx(metadata, "target")
        if tx.get("retired") or not tx.get("slot"):
            raise RuntimeError("Checkpoint requires a live target reservation")
        if "checkpoint" not in tx:
            if checkpoint["signature"] != self.signature:
                raise ValueError("Checkpoint layout mismatch")
            tx["request"] = decode(checkpoint["request"])
            self.workers("restore", request_id=metadata["request_id"], checkpoints=checkpoint["rng"])
            tx["checkpoint"] = checkpoint
        return tx["reservation"]

    def ready(self, metadata, receipt):
        tx = self._tx(metadata, "target")
        if tx.get("retired") and not tx.get("activated"):
            raise RuntimeError("Target reservation was discarded")
        if tx.get("ready"):
            return [worker["coordinate"] for worker in self.worker_info]
        if "checkpoint" not in tx:
            raise RuntimeError("Missing request/RNG checkpoint")
        if receipt != {"migration_id": metadata["migration_id"], "workers": len(self.worker_info)}:
            raise RuntimeError("Missing all-worker sender completion receipt")
        self.workers("fence")
        tx["ready"] = True
        return [worker["coordinate"] for worker in tx["reservation"]["workers"]]

    def activate(self, metadata, owner):
        tx = self._tx(metadata, "target")
        if tx.get("activated"):
            return True
        if owner["owner_generation"] != metadata["owner_generation"] + 1 or owner["group_id"] != metadata["target"]:
            raise RuntimeError("Invalid execution ownership receipt")
        if not tx.get("ready"):
            raise RuntimeError("Target transfer is not ready")
        # Return an EngineCoreRequest; the owning native client must submit it
        # so that DP waves and cancellation routing stay under vLLM's control.
        from ..engine_proxy import req2corereq

        request = req2corereq(copy.deepcopy(tx["request"]))
        request.sampling_params.extra_args = dict(request.sampling_params.extra_args or {})
        request.sampling_params.extra_args["llx_migration_id"] = metadata["migration_id"]
        return encode(request)

    def _activate(self, tx, request):
        from vllm.v1.request import RequestStatus

        metadata = tx["metadata"]
        request.status = RequestStatus.RUNNING
        tx["slot"] = False
        self.scheduler.requests[request.request_id] = request
        self.scheduler.running.append(request)
        self.scheduler._asyncpb_new_req_ids.add(request.request_id)
        self.track(
            request.request_id,
            metadata["owner_generation"] + 1,
            len(request.output_token_ids),
            tx["checkpoint"].get("parent"),
        )
        tx["activated"] = True
        self._retire(tx)
        return True

    def state(self, metadata, side):
        tx = self._tx(metadata, side)
        return {key: tx.get(key, False) for key in ("parked", "ready", "activated", "retired")}

    def retire_transfers(self, metadata, **unused):
        for phase in ("prefix", "delta"):
            if metadata["migration_id"] + "/" + phase in self.sends and self._done(metadata, phase) is not True:
                return None
        return True

    def _unpin(self, tx):
        self.manager.block_pool.free_blocks(tx["pins"])
        tx["pins"].clear()

    def _retire(self, tx):
        migration_id = tx["metadata"]["migration_id"]
        # ADD can run while an execute_model future is pending. Activation is
        # scheduler-only; only source utilities have worker send journals.
        if any(migration_id + "/" + phase in self.sends for phase in ("prefix", "delta")):
            self.workers("retire", migration_id=migration_id)
        # Late transport callbacks must see the tombstone before their journal
        # disappears, otherwise they fall through to the legacy migration path.
        tx["retired"] = True
        for phase in ("prefix", "delta"):
            self.sends.pop(migration_id + "/" + phase, None)
        # Keep a small idempotency receipt, not token histories/RNG/KV metadata.
        for key in ("request", "prefix", "delta", "checkpoint", "reservation"):
            tx.pop(key, None)

    def release(self, metadata, **unused):
        tx = self._tx(metadata, "source")
        if tx.get("retired"):
            return True
        if not tx.get("parked"):
            raise RuntimeError("Cannot release an executing source")
        if self.retire_transfers(metadata) is not True:
            raise RuntimeError("Source still referenced by native transfer")
        from vllm.v1.request import RequestStatus

        self.scheduler.finish_requests(metadata["request_id"], RequestStatus.FINISHED_ABORTED)
        if self.scheduler.finished_req_ids_dict is not None:
            for ids in self.scheduler.finished_req_ids_dict.values():
                ids.discard(metadata["request_id"])
        self.workers("forget", request_id=metadata["request_id"])
        self.tracked.pop(metadata["request_id"], None)
        self._unpin(tx)
        self._retire(tx)
        return True

    def discard(self, metadata, **unused):
        tx = self._tx(metadata, "target")
        if tx.get("activated"):
            raise RuntimeError("Cannot discard a committed target")
        if tx.get("retired"):
            return True
        if tx.get("slot"):
            self.manager.free(tx["request"])
            tx["slot"] = False
        self.workers("forget", request_id=metadata["request_id"])
        self._retire(tx)
        return True

    def resume(self, metadata, **unused):
        tx = self._tx(metadata, "source")
        if tx.get("retired"):
            return True
        if self.retire_transfers(metadata) is not True:
            raise RuntimeError("Cannot resume during native snapshot access")
        request = self.scheduler.requests.get(metadata["request_id"])
        if tx.get("parked") and request is not None and not request.is_finished():
            self.scheduler.running.append(request)
        self._unpin(tx)
        self._retire(tx)
        return True


def core_rpc(core, operation, payload):
    if not hasattr(core, "_asyncpb_tail"):
        core._asyncpb_tail = CoreTransactions(core)
    args = decode(payload)
    return encode(getattr(core._asyncpb_tail, operation)(**args))
