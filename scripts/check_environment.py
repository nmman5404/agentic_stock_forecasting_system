from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv


PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("lightgbm", "lightgbm"),
    ("langchain_google_genai", "langchain_google_genai"),
    ("vnstock", "vnstock"),
    ("feedparser", "feedparser"),
]

KEYS = [
    "GOOGLE_API_KEY",
]


def main() -> int:
    load_dotenv()
    print(f"python_version: {sys.version.split()[0]}")
    print(f"python_executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"working_directory: {Path.cwd()}")

    print("\npackages:")
    for label, module_name in PACKAGES:
        print(f"- {label}: {_package_version(module_name)}")

    print("\nenv_keys:")
    for key in KEYS:
        print(f"- {key}: exists={bool(os.getenv(key))}")
    return 0


def _package_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"not_installed ({type(exc).__name__})"
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as dist_version

        return dist_version(module_name.split(".")[0])
    except Exception:
        return "installed_version_unknown"


if __name__ == "__main__":
    raise SystemExit(main())
