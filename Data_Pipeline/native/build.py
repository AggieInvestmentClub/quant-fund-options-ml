"""Build the optional CRR C++ extension in place, without Python build packages.

Run from any directory with: python Data_Pipeline/native/build.py
Requires a C++17 compiler and Python development headers on macOS or Linux.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    source = Path(__file__).with_name("crr_native.cpp")
    target_dir = source.parent.parent
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    include_dir = sysconfig.get_paths()["include"]
    if not extension_suffix or not include_dir:
        raise RuntimeError("Python extension build settings are unavailable")
    if sys.platform not in {"darwin", "linux"}:
        raise RuntimeError("This build helper currently supports macOS and Linux")

    target = target_dir / f"_crr_native{extension_suffix}"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    command = [
        *compiler, "-O3", "-std=c++17", "-fPIC", "-ffp-contract=off",
        f"-I{include_dir}", str(source),
    ]
    if sys.platform == "darwin":
        command += ["-bundle", "-undefined", "dynamic_lookup"]
    else:
        command += ["-shared"]
    command += ["-o", str(target)]
    subprocess.run(command, check=True)
    print(target)


if __name__ == "__main__":
    main()
