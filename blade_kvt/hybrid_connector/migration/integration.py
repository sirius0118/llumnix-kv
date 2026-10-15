# ruff: noqa: PLC0415 -- lazy vLLM imports avoid connector initialization cycles
"""Opt-in setup for the version-pinned AsyncPB adapter."""

import os
import secrets
import socket
import uuid
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path


def configure(vllm_config):
    from vllm.config import KVTransferConfig
    from vllm.utils.network_utils import get_ip
    from vllm.v1.engine.core import EngineCore

    if version("vllm") != "0.21.0+zeta.all" or not hasattr(EngineCore, "asyncpb_tail"):
        raise RuntimeError("Install the pinned llumnix-kv v0.21 patch before starting AsyncPB migration")
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("AsyncPB migration requires exclusive ownership of the KV connector")
    # A distinct port range per replica; DP EngineCores and TP/PP workers use
    # separate offsets. Configuration reaches remote Monarch workers as well.
    parallel = vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size > 1
        and parallel.data_parallel_size > 1
        and parallel.enable_expert_parallel
        and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
    ):
        raise ValueError(
            "PP + DP + EP full CUDA graphs fail the vLLM 0.21 pause/resume regression; "
            "use enforce_eager=True for this topology"
        )
    width = parallel.data_parallel_size + parallel.world_size_across_dp
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
    if ephemeral_start <= 1024 + width:
        raise RuntimeError("No non-ephemeral port range available for migration")
    for _ in range(100):
        with ExitStack() as stack:
            # Worker startup opens many outgoing TCP sockets. Their ephemeral
            # ports must not consume a port promised to a later RDMA listener.
            port = 1024 + secrets.randbelow(ephemeral_start - 1024 - width)
            try:
                for offset in range(width):
                    stack.enter_context(socket.socket()).bind(("", port + offset))
            except OSError:
                continue
            break
    else:
        raise RuntimeError("No free port range for migration")
    extra = dict(
        backend="kvt+migration",
        naming_url="fake://",
        rpc_host=get_ip(),
        rpc_port=port,
        kvt_port=port + parallel.data_parallel_size,
        kvt_inst_id=uuid.uuid4().hex,
    )
    if affinity := os.environ.get("LLX_KVT_NIC_AFFINITY"):
        extra["nic_affinity"] = affinity
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="HybridConnector", kv_role="kv_both", engine_available_port=port, kv_connector_extra_config=extra
    )
    vllm_config.cache_config.enable_prefix_caching = False
