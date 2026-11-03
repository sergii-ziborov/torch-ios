"""Packs a staged torch tree into a wheel: make_wheel.py STAGE VERSION TAG DEST."""

import base64
import hashlib
import os
import sys
import zipfile

# setuptools is left out: on iOS it would only serve torch.utils.cpp_extension,
# and nothing can be compiled on the device.
REQUIRES = ["filelock", "typing-extensions>=4.13.0", "sympy>=1.13.3", "networkx>=2.5.1", "jinja2", "fsspec>=0.8.5"]


def main(stage, version, tag, dest):
    info = f"torch-{version}.dist-info"
    os.makedirs(os.path.join(stage, info), exist_ok=True)
    files = {
        "METADATA": (
            f"Metadata-Version: 2.1\nName: torch\nVersion: {version}\n"
            "Summary: PyTorch built for iOS (unofficial, https://github.com/sergii-ziborov/torch-ios)\n"
            "License: BSD-3-Clause\n" + "".join(f"Requires-Dist: {r}\n" for r in REQUIRES)
        ),
        "WHEEL": f"Wheel-Version: 1.0\nGenerator: torch-ios\nRoot-Is-Purelib: false\nTag: {tag}\n",
        "top_level.txt": "torch\ntorchgen\nfunctorch\n",
    }
    for name, text in files.items():
        with open(os.path.join(stage, info, name), "w") as handle:
            handle.write(text)

    path = os.path.join(dest, f"torch-{version}-{tag}.whl")
    records = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder, _, names in sorted(os.walk(stage)):
            for name in sorted(names):
                full = os.path.join(folder, name)
                relative = os.path.relpath(full, stage)
                with open(full, "rb") as handle:
                    data = handle.read()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                records.append(f"{relative},sha256={digest},{len(data)}")
                archive.writestr(relative, data)
        records.append(f"{info}/RECORD,,")
        archive.writestr(f"{info}/RECORD", "\n".join(records) + "\n")
    print(path, f"{os.path.getsize(path) / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
