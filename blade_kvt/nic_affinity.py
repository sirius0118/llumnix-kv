import json
import logging
import os
import subprocess
import uuid

from blade_kvt import is_nv_gpu

logger = logging.getLogger("blade_kvt")


def _persist_aff_data(aff_data, filepath):
    max_gpu_rank = max([int(k) for k in aff_data])
    out = ','.join(aff_data[str(gpu_rank)] for gpu_rank in range(max_gpu_rank + 1))
    logger.info("persist_aff_data out=%s", out)

    tmpfilepath = f"/tmp/persist_aff_data-{str(uuid.uuid4())}.txt"
    with open(tmpfilepath, 'w') as f:
        f.write(out)

    os.replace(tmpfilepath, filepath)  # atomic operation


def _do_generate_nic_affinity_file(filepath):
    cmdline = ['run_affinity', '-n']
    if not is_nv_gpu():
        cmdline += ['PPU']
    else:
        cmdline += ['GPU']
    filename = f"/tmp/generate_nic_affinity_file-{str(uuid.uuid4())}"
    cmdline += ['-o', filename]
    logger.info("do_generate_nic_affinity_file cmdline=%s", cmdline)

    subprocess.check_call(cmdline)
    with open(filename) as input:
        aff_data = json.loads(input.read())
        _persist_aff_data(aff_data, filepath)
    os.remove(filename)


def generate():
    FILEPATH = "/tmp/pai_blade_llm_kvtransfer_rdma_nic_affinity.txt"
    explicit = os.environ.get("LLX_KVT_NIC_AFFINITY")
    if explicit:
        names = [name.strip() for name in explicit.split(",")]
        if any(not name or "/" in name or not os.path.isdir("/sys/class/infiniband/" + name) for name in names):
            raise ValueError("Invalid explicit RDMA NIC affinity")
        _persist_aff_data({str(i): name for i, name in enumerate(names)}, FILEPATH)
        return
    try:
        _do_generate_nic_affinity_file(FILEPATH)
        return
    except Exception:
        logger.warning("do_generate_nic_affinity_file failed")

    # use fallback logic
    aff_data = {str(rank): "UNKNOWN-NIC-NAME" for rank in range(129)}
    _persist_aff_data(aff_data, FILEPATH)
    return
