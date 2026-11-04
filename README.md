# torch-ios

PyTorch for iPhone and iPad, as Python wheels: `import torch` inside an iOS app
that embeds CPython 3.14, with training, not just inference.

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = nn.functional.cross_entropy(model(x), y)
loss.backward()
optimizer.step()
```

This is an unofficial build of upstream PyTorch, not affiliated with the PyTorch
project or the Linux Foundation. PyTorch's own iOS story was LibTorch-Lite (C++,
inference, no longer developed) and is now ExecuTorch (exported models, C++ and
Swift). Neither gives you the `torch` Python package on the device. This does.

## What you get

The full library, built from the PyTorch source with a small patch:

- tensors, autograd, `torch.nn`, `torch.optim`, `torch.func`
- `torch.linalg` and LAPACK, matrix maths through Apple's Accelerate
- `torch.fft`, `torch.distributions`, `torch.save` and `torch.load`
- NumPy interop: `torch.from_numpy`, `tensor.numpy()`
- `torch.jit.trace` and `torch.jit.script` (interpreted), `torch.export`
- `torch.compile` with the `eager` and `aot_eager` backends (no code generation)
- quantisation on the QNNPACK engine: dynamic, static and quantised ops

Not in this build:

- GPU: no CUDA and no MPS, so training runs on the CPU
- `torch.compile` with Inductor, or anything else that generates and compiles code at
  run time: iOS has no compiler on the device and forbids JIT
- `torch.distributed`, and DataLoader workers (one process only; keep `num_workers=0`)

## Status

| | |
| --- | --- |
| PyTorch | 2.15.0 |
| Python | CPython 3.14 (BeeWare's [Python-Apple-support](https://github.com/beeware/Python-Apple-support) 3.14-b14) |
| Simulator (arm64) | built and tested inside an iOS app: training, NumPy, linalg, fft, conv, `torch.func`, `torch.jit`, DataLoader, save and load |
| Device (arm64) | built, minimum iOS 13.0; first runs on an iPhone pending |
| Wheel | about 50 MB; `torch/_C` is about 115 MB once installed |

## Download

Prebuilt wheels for both platforms are on the
[releases page](https://github.com/sergii-ziborov/torch-ios/releases). Or build them:

## Build

On a Mac with Xcode, CMake 3.27 or newer, git and `python3.14`:

```sh
./build.sh iphonesimulator      # dist/torch-2.15.0-cp314-cp314-ios_14_0_arm64_iphonesimulator.whl
./build.sh iphoneos             # dist/torch-2.15.0-cp314-cp314-ios_13_0_arm64_iphoneos.whl
```

It clones PyTorch at the tag (only the submodules it needs), applies
`patches/pytorch-v2.15.0-ios.patch`, builds ONNX's protobuf compiler and SLEEF's
code generators for the Mac, downloads BeeWare's CPython for iOS and the iOS NumPy
wheel for their headers (both checked), then cross-compiles. Expect about an hour per
platform on an M-series Mac, 2 GB for the sources and 6 GB per platform while it builds.

Steps can be run one at a time: `./build.sh iphoneos fetch|configure|build|wheel`.
`PYTHON_XCFRAMEWORK`, `NUMPY_INCLUDE`, `WORK`, `DIST` and `JOBS` override the defaults
(see the top of `build.sh`).

## Checking a build

`tests/conformance.py` runs about 200 groups of operations (elementwise maths, reductions,
linear algebra, convolutions, pooling, normalisation, losses, attention, recurrent layers,
FFT, special functions, autograd and gradcheck, optimisers, `torch.func`, sparse,
quantisation, `torch.compile`, `torch.export`, serialisation, data loading) on inputs
made from formulas. Run it in the app and on a Mac with the same torch version, then:

```sh
python tests/compare.py mac.json ios.json
```

The current build matches the Mac on every group.

## Use it in an app

The wheels follow [PEP 730](https://peps.python.org/pep-0730/) and install with pip like any iOS wheel:

```sh
python3.14 -m pip install --target app_packages --only-binary :all: \
  --platform ios_13_0_arm64_iphoneos --python-version 3.14 --implementation cp --abi cp314 \
  --find-links dist torch
```

pip also installs torch's pure-Python dependencies (sympy, networkx, jinja2, fsspec,
filelock, typing-extensions). Your app needs NumPy for iOS as well, from BeeWare's
channel (`--extra-index-url https://pypi.anaconda.org/beeware/simple`).

The app must turn `torch/_C.cpython-314-iphoneos.so` into a framework when it is
built, as it does every other extension module; BeeWare's `install_python` build step
(in `Python.xcframework/build/utils.sh`) does that. All of PyTorch's native code is in
that one module, so there are no other libraries to embed.

## How it works

PyTorch treats every iOS build as "mobile": no autograd, no Python bindings. The patch
adds one switch, `PYTHILO_IOS_FULL`, that keeps the desktop build while targeting iOS:

- `CMakeLists.txt` and `Macros.h`: not the mobile build, so autograd and the Python
  bindings stay, and `C10_MOBILE` is not defined
- `torch/CMakeLists.txt`: `torch_python` and `libshm` are static, and everything is
  force-loaded into `torch._C`, a dynamic library (App Store Connect refuses Mach-O
  bundles inside frameworks) that links `Python.framework` and takes the iOS name
- `FindvecLib.cmake`: iOS has Accelerate but no separate `libcblas`
- `Dependencies.cmake`: the full ONNX library, not only its protobuf messages
- `Dependencies.cmake`: with Accelerate, `torch.dot` goes through `cblas_sdot`; Accelerate's
  Fortran `sdot_` returns a double, which read as a float gave 0
- two places that call `system()`, which iOS lacks, now fail cleanly instead
- QNNPACK's CMake accepts `CMAKE_SYSTEM_NAME` iOS, so quantised kernels are built
- `torch/__init__.py`: no `libtorch_global_deps` and no `torch_shm_manager` to look for on iOS
- `torch/multiprocessing`: iOS's CPython has no `_multiprocessing`, so the resource tracker is optional
- `PreBuildSteps.cmake`: only the submodules this build uses have to be present

## Licence

The scripts, tools and patch here are BSD-3-Clause ([LICENSE](LICENSE)). PyTorch and
the wheels built from it are under PyTorch's licence, which each wheel carries as
`torch/LICENSE.pytorch`.

Built for [Pythilo](https://pythilo.com), a Python IDE and course for iPhone and iPad,
where it powers the machine learning lessons.
