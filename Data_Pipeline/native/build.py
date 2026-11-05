"""Build the optional Leisen-Reimer C++ extension in place, without Python build packages.

Run from any directory with: python Data_Pipeline/native/build.py
Requires a C++17 compiler and Python development headers on macOS or Linux.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    source = Path(__file__).with_name("lr_native.cpp")
    native_dir = source.parent
    target_dir = native_dir.parent
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    include_dir = sysconfig.get_paths()["include"]
    if not extension_suffix or not include_dir:
        raise RuntimeError("Python extension build settings are unavailable")
    if sys.platform not in {"darwin", "linux"}:
        raise RuntimeError("This build helper currently supports macOS and Linux")

    libdeflate_dir = native_dir / "libdeflate"
    libdeflate_build_dir = libdeflate_dir / "build"
    libdeflate_a = libdeflate_build_dir / "libdeflate.a"
    if not libdeflate_a.exists() and libdeflate_dir.exists():
        cmake_bin = shutil.which("cmake") or str(Path(sys.executable).parent / "cmake")
        subprocess.run(
            [
                cmake_bin, "-B", str(libdeflate_build_dir), "-S", str(libdeflate_dir),
                "-DLIBDEFLATE_BUILD_STATIC_LIB=ON", "-DLIBDEFLATE_BUILD_SHARED_LIB=OFF",
                "-DLIBDEFLATE_BUILD_GZIP=OFF", "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
            ],
            check=True,
        )
        subprocess.run([cmake_bin, "--build", str(libdeflate_build_dir), "-j4"], check=True)

    target = target_dir / f"_lr_native{extension_suffix}"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    command = [
        *compiler, "-O3", "-std=c++17", "-fPIC", "-ffp-contract=off",
        f"-I{include_dir}", f"-I{libdeflate_dir}",
        str(source),
    ]
    if libdeflate_a.exists():
        command.append(str(libdeflate_a))
    command.append("-lz")

    if sys.platform == "darwin":
        command += ["-bundle", "-undefined", "dynamic_lookup"]
    else:
        command += ["-shared"]
    command += ["-o", str(target)]
    subprocess.run(command, check=True)
    print(target)


if __name__ == "__main__":
    main()
