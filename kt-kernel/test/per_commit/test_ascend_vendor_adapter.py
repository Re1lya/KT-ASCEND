from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORS = REPO_ROOT / "cpu_backend" / "vendors"


def _compile_and_run(tmp_path: Path, source: str, *, include_dirs: list[Path]) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the vendor header compile contract")
    source_path = tmp_path / "probe.cpp"
    binary_path = tmp_path / "probe"
    source_path.write_text(source, encoding="utf-8")
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror"]
    for include_dir in include_dirs:
        command.extend(("-I", str(include_dir)))
    command.extend((str(source_path), "-o", str(binary_path)))
    subprocess.run(command, check=True, capture_output=True, text=True)
    return subprocess.run([str(binary_path)], check=False, capture_output=True, text=True)


def test_cpu_only_vendor_header_has_no_acl_dependency(tmp_path):
    result = _compile_and_run(
        tmp_path,
        '#include "vendor.h"\nint main() { return 0; }\n',
        include_dirs=[VENDORS],
    )
    assert result.returncode == 0, result.stderr


def test_ascend_adapter_checks_null_and_runtime_errors(tmp_path):
    acl_include = tmp_path / "stub" / "acl"
    acl_include.mkdir(parents=True)
    (acl_include / "acl_rt.h").write_text(
        """
#pragma once
#include <cstdint>
using aclError = int32_t;
using aclrtStream = void*;
using aclrtHostFunc = void (*)(void*);
constexpr aclError ACL_SUCCESS = 0;
inline aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc, void*) {
  return stream == reinterpret_cast<void*>(1) ? ACL_SUCCESS : 507899;
}
""",
        encoding="utf-8",
    )
    result = _compile_and_run(
        tmp_path,
        """
#include "ascend.h"
#include <stdexcept>
#include <string>
static void callback(void*) {}
int main() {
  try {
    ktransformers::vendor::launch_host_function(0, &callback, nullptr, "probe.cpp", 11);
    return 1;
  } catch (const std::invalid_argument&) {
  }
  try {
    ktransformers::vendor::launch_host_function(2, &callback, nullptr, "probe.cpp", 22);
    return 2;
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    if (message.find("aclrtLaunchHostFunc") == std::string::npos ||
        message.find("507899") == std::string::npos ||
        message.find("probe.cpp:22") == std::string::npos) {
      return 3;
    }
  }
  ktransformers::vendor::launch_host_function(1, &callback, nullptr, "probe.cpp", 33);
  return 0;
}
""",
        include_dirs=[tmp_path / "stub", VENDORS],
    )
    assert result.returncode == 0, result.stderr
