"""Build the pinned adapter and patch an exact vLLM wheel before importing it."""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path


def transport_dependencies(repo):
    # Keep the CUDA-12 Barex runtime private; do not replace the image's CUDA
    # packages (the supported vLLM wheel itself uses CUDA 13).
    vendor = repo / "build/transport-deps"
    vendor.mkdir(parents=True, exist_ok=True)
    archive = vendor / "barex.deb"
    digest = "008f7c2556d616bbfe4333d73890e9d00fbfabfb9297c65473de0a2559b9eebd"
    if not archive.exists():
        urllib.request.urlretrieve(
            "https://eflop.oss-cn-beijing.aliyuncs.com/accl-barex/tcp/"
            "accl-barex-pkg-test-tcp-v1.1.7/mlx/accl-barex-cuda12.2-devel-1.5.3-1.deb",
            archive,
        )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Barex archive checksum differs from the validated build")
    subprocess.run(["dpkg-deb", "-x", str(archive), str(vendor / "barex")], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "nvidia-cuda-runtime-cu12==12.2.140",
            "-d",
            str(vendor),
        ],
        check=True,
    )
    libs = repo / "blade_kvt/.libs"
    libs.mkdir(exist_ok=True)
    shutil.copy2(vendor / "barex/usr/lib/libaccl_barex.so", libs / "libaccl_barex.so")
    shutil.copy2(libs / "libaccl_barex.so", libs / "libaccl_barex.so.1")
    with zipfile.ZipFile(next(vendor.glob("nvidia_cuda_runtime_cu12-12.2.140-*.whl"))) as wheel:
        (libs / "libcudart.so.12").write_bytes(wheel.read("nvidia/cuda_runtime/lib/libcudart.so.12"))
    return [
        "-DACCL_BAREX_LIBRARY=" + str(libs / "libaccl_barex.so"),
        "-DACCL_BAREX_INCLUDE=" + str(vendor / "barex/usr/include"),
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DCMAKE_INSTALL_RPATH=$ORIGIN/.libs",
        "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,--disable-new-dtags",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify the installed patch without modifying files")
    args = parser.parse_args()
    if version("vllm") != "0.21.0+zeta.all":
        raise RuntimeError("Expected vLLM 0.21.0+zeta.all; refusing to patch another build")
    repo = Path(__file__).resolve().parents[1]
    site = Path(importlib.util.find_spec("vllm").origin).parent.parent
    manifest = json.loads((repo / "patches/vllm-v0.21.0.manifest.json").read_text())
    hashes = {name: hashlib.sha256((site / name).read_bytes()).hexdigest() for name in manifest}
    installed = all(hashes[name] == item["patched"] for name, item in manifest.items())
    if args.check:
        if not installed:
            raise RuntimeError("Installed vLLM sources do not match the pinned migration patch")
        return
    if not installed and any(hashes[name] != item["base"] for name, item in manifest.items()):
        raise RuntimeError("Unexpected vLLM source changes; install a clean matching wheel before applying this patch")
    patch = repo / "patches/vllm-v0.21.0.patch"
    if not installed:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=site, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run([sys.executable, "-m", "pip", "install", "pybind11==2.13.6", "setuptools-scm", "wheel"], check=True)
    pybind_dir = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()
    env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+g" + revision[:12])
    env["CMAKE_ARGS"] = (
        shlex.join(
            [
                "-DUSE_PYBIND11=ON",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON",
                "-DENABLE_BATCH_COPY=OFF",
                "-Dpybind11_DIR=" + pybind_dir,
            ]
            + transport_dependencies(repo)
        )
        + " "
        + env.get("CMAKE_ARGS", "")
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(repo)], env=env, check=True
    )
    if not installed:
        subprocess.run(["git", "apply", str(patch)], cwd=site, check=True)
    subprocess.run([sys.executable, __file__, "--check"], check=True)
    subprocess.run([sys.executable, "-I", "-c", "import blade_kvt.kvtransfer_ops"], check=True)


if __name__ == "__main__":
    main()
