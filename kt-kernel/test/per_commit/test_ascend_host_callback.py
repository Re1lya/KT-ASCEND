from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "test" / "fixtures" / "ascend" / "host_callback_probe.cpp"


def _ascend_paths() -> tuple[Path, Path]:
    root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    candidates = [root, root / "aarch64-linux"]
    include = next((candidate / "include" for candidate in candidates if (candidate / "include/acl/acl_rt.h").is_file()), None)
    library = next((candidate / "lib64" for candidate in candidates if (candidate / "lib64/libacl_rt.so").is_file()), None)
    if include is None or library is None:
        pytest.skip("CANN acl_rt headers and library are unavailable")
    return include, library


@pytest.fixture(scope="module")
def callback_probe(tmp_path_factory) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    include, library = _ascend_paths()
    output = tmp_path_factory.mktemp("ascend-callback") / "host_callback_probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DKTRANSFORMERS_USE_ASCEND=1",
            "-I",
            str(include),
            "-I",
            str(REPO_ROOT / "cpu_backend/vendors"),
            str(SOURCE),
            "-L",
            str(library),
            f"-Wl,-rpath,{library}",
            "-lacl_rt",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.parametrize("mode", ["single", "stress", "multi"])
def test_cann_host_callback_ordering(callback_probe: Path, mode: str):
    result = subprocess.run([str(callback_probe), mode], check=False, capture_output=True, text=True, timeout=120)
    print(result.stdout)
    assert result.returncode == 0, result.stderr


def test_production_callback_contains_no_device_work():
    source = SOURCE.read_text(encoding="utf-8")
    callback_body = source.split("static void ordered_callback", 1)[1].split("}\n", 1)[0]
    for forbidden in ("aclrtMemcpy", "aclrtSynchronize", "aclrtMalloc", "aclrtFree", "torch"):
        assert forbidden not in callback_body
