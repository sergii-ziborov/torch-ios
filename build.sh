#!/bin/bash
# Builds PyTorch for iOS as CPython 3.14 wheels (PEP 730), one per platform:
#
#   ./build.sh iphonesimulator      # Apple silicon simulator
#   ./build.sh iphoneos             # iPhone and iPad
#   ./build.sh <platform> fetch|configure|build|wheel
#
# The wheel holds the full library (tensors, autograd, nn, optim, linalg, fft,
# save/load, the Python bindings), not PyTorch's trimmed mobile runtime. Everything
# native is linked into the single extension module torch/_C, because an iOS app
# loads native code only from frameworks, and BeeWare-style apps turn each .so
# into one. Matrix maths and LAPACK go to Accelerate. No CUDA, MPS, distributed,
# OpenMP or run-time compilation.
#
# Needs a Mac with Xcode, CMake 3.27 or newer, git, and python3.14 on PATH.
# About 2 GB for the sources and 6 GB per platform while it builds.
#
# Environment:
#   PYTHON_XCFRAMEWORK  a CPython 3.14 Python.xcframework (default: BeeWare's
#                       Python-Apple-support 3.14-b14, downloaded and checked)
#   NUMPY_INCLUDE       NumPy's C headers for the platform (default: taken from
#                       the iOS NumPy wheel, downloaded)
#   WORK                build folder (default: ./work)
#   DIST                where wheels go (default: ./dist)
#   JOBS                parallel compile jobs (default: 8)
set -euo pipefail

platform=${1:?iphonesimulator or iphoneos}
step=${2:-all}
version=2.15.0
numpy_version=2.5.2.post1
support_tag=3.14-b14
support_asset=Python-3.14-iOS-support.b14.tar.gz
support_sha256=703ff822529f46edcbcd8b88096b2544f6990f3109ad38b4b4b61bb489a7c1ee

here=$(cd "$(dirname "$0")" && pwd)
work=${WORK:-$here/work}
dist=${DIST:-$here/dist}
jobs=${JOBS:-8}
src="$work/src"
out="$work/build-$platform"
venv="$work/venv"

# Native code must not need a newer iOS than the frameworks around it claim, or
# App Store Connect refuses the upload; BeeWare's CPython says 13.0.
case "$platform" in
  iphonesimulator) slice=ios-arm64_x86_64-simulator; minimum=14.0 ;;
  iphoneos) slice=ios-arm64; minimum=13.0 ;;
  *) echo "Unknown platform $platform" >&2; exit 1 ;;
esac
tag="cp314-cp314-ios_${minimum//./_}_arm64_${platform}"
sdk=$(xcrun --sdk "$platform" --show-sdk-path)
mkdir -p "$work"

python_framework() {
  if [[ -n "${PYTHON_XCFRAMEWORK:-}" ]]; then echo "$PYTHON_XCFRAMEWORK"; return; fi
  local support="$work/python-support"
  if [[ ! -d "$support/Python.xcframework" ]]; then
    mkdir -p "$support"
    curl -fsSL -o "$support/support.tar.gz" \
      "https://github.com/beeware/Python-Apple-support/releases/download/$support_tag/$support_asset" >&2
    echo "$support_sha256  $support/support.tar.gz" | shasum -a 256 -c - >&2
    tar xzf "$support/support.tar.gz" -C "$support"
  fi
  echo "$support/Python.xcframework"
}

numpy_include() {
  if [[ -n "${NUMPY_INCLUDE:-}" ]]; then echo "$NUMPY_INCLUDE"; return; fi
  local folder="$work/numpy-$platform" platforms=()
  if [[ ! -d "$folder/numpy/_core/include" ]]; then
    for v in 13_0 14_0 15_0 16_0 17_0 18_0; do platforms+=(--platform "ios_${v}_arm64_${platform}"); done
    "$venv/bin/python" -m pip download --quiet --disable-pip-version-check --no-deps "numpy==$numpy_version" \
      --only-binary :all: --implementation cp --python-version 3.14 --abi cp314 "${platforms[@]}" \
      --index-url https://pypi.org/simple --extra-index-url https://pypi.anaconda.org/beeware/simple \
      -d "$folder" >&2
    (cd "$folder" && unzip -q -o numpy-*.whl 'numpy/_core/include/*')
  fi
  echo "$folder/numpy/_core/include"
}

fetch() {
  if [[ ! -d "$src" ]]; then
    git clone --depth 1 --branch "v$version" https://github.com/pytorch/pytorch "$src"
    # Only what this build uses; the full set is several gigabytes of GPU code.
    git -C "$src" submodule update --init --depth 1 --jobs 8 \
      third_party/pybind11 third_party/fmt third_party/sleef third_party/cpuinfo \
      third_party/pthreadpool third_party/FXdiv third_party/FP16 third_party/psimd \
      third_party/flatbuffers third_party/onnx third_party/nlohmann \
      third_party/pocketfft third_party/cpp-httplib third_party/protobuf
    git -C "$src" apply "$here/patches/pytorch-v$version-ios.patch"
  fi
  if [[ ! -x "$venv/bin/python" ]]; then
    python3.14 -m venv "$venv"
    "$venv/bin/pip" install --quiet --disable-pip-version-check pyyaml typing_extensions ninja setuptools packaging
  fi
  # Code generators run on the Mac: ONNX's protobuf compiler, of the same version
  # as the protobuf linked into torch, and SLEEF's vector maths generators.
  if [[ ! -x "$src/build_host_protoc/bin/protoc" ]]; then
    PATH="$venv/bin:$PATH" "$src/scripts/build_host_protoc.sh" --other-flags -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  fi
  if [[ ! -x "$work/sleef-host/bin/mkdisp" ]]; then
    cmake -S "$src/third_party/sleef" -B "$work/sleef-host" -G Ninja -DCMAKE_MAKE_PROGRAM="$venv/bin/ninja" \
      -DCMAKE_BUILD_TYPE=Release -DSLEEF_BUILD_TESTS=OFF -DSLEEF_BUILD_DFT=OFF -DSLEEF_BUILD_GNUABI_LIBS=OFF \
      -DSLEEF_BUILD_SHARED_LIBS=OFF -DSLEEF_DISABLE_OPENMP=ON
    cmake --build "$work/sleef-host"
  fi
}

configure() {
  local framework numpy
  framework="$(python_framework)/$slice"
  numpy=$(numpy_include)
  cmake -S "$src" -B "$out" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$venv/bin/ninja" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$sdk" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_SYSTEM_PROCESSOR=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$minimum" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-DPYTHILO_IOS_FULL" \
    -DCMAKE_CXX_FLAGS="-DPYTHILO_IOS_FULL" \
    -DCMAKE_INSTALL_PREFIX="$out/install" \
    -DPYTHILO_IOS_FULL=ON \
    -DPYTHILO_PLATFORM="$platform" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_PYTHON=ON -DBUILD_TEST=OFF -DBUILD_BINARY=OFF -DBUILD_LAZY_TS_BACKEND=OFF -DBUILD_FUNCTORCH=OFF \
    -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF -DUSE_MPS=OFF -DUSE_VULKAN=OFF -DUSE_PYTORCH_METAL=OFF \
    -DUSE_DISTRIBUTED=OFF -DUSE_GLOO=OFF -DUSE_TENSORPIPE=OFF -DUSE_NCCL=OFF -DUSE_MPI=OFF \
    -DUSE_MKLDNN=OFF -DUSE_MKL=OFF -DUSE_FBGEMM=OFF -DUSE_KINETO=OFF -DUSE_ITT=OFF -DUSE_KLEIDIAI=OFF \
    -DUSE_NNPACK=OFF -DUSE_XNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_QNNPACK=OFF \
    -DUSE_OPENMP=OFF -DUSE_MAGMA=OFF -DUSE_MIMALLOC=OFF -DUSE_FLASH_ATTENTION=OFF -DUSE_MEM_EFF_ATTENTION=OFF \
    -DUSE_NUMPY=ON -DUSE_BLAS=ON -DUSE_LAPACK=ON -DBLAS=vecLib \
    -DvecLib_INCLUDE_DIR="$sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers" \
    -DPython_EXECUTABLE="$venv/bin/python" \
    -DPython_INCLUDE_DIR="$framework/include/python3.14" \
    -DPython_LIBRARY="$framework/Python.framework/Python" \
    -DPython_NumPy_INCLUDE_DIR="$numpy" \
    -DPython_SOABI="cpython-314-$platform" \
    -DBUILD_CUSTOM_PROTOBUF=ON \
    -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE="$src/build_host_protoc/bin/protoc" \
    -DNATIVE_BUILD_DIR="$work/sleef-host" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
}

build() {
  # A release version string, not 2.15.0a0+git<hash>.
  PYTORCH_BUILD_VERSION="$version" PYTORCH_BUILD_NUMBER=0 \
    cmake --build "$out" --target _C -j "$jobs"
}

# The Python side of torch, torchgen and functorch, plus the one native module.
# Headers, static libraries, CMake files and C++ sources stay behind.
wheel() {
  local stage="$work/wheel-$platform" module="$out/torch/_C.cpython-314-$platform.so"
  [[ -f "$module" ]] || { echo "Build first: $module is missing." >&2; exit 1; }
  (cd "$src" && PYTORCH_BUILD_VERSION="$version" PYTORCH_BUILD_NUMBER=0 "$venv/bin/python" -m tools.generate_torch_version \
    --is-debug=0 --cuda-version= --hip-version= --rocm-version= --xpu-version=)
  rm -rf "$stage" && mkdir -p "$stage" "$dist"
  for package in torch torchgen functorch; do
    rsync -a --prune-empty-dirs \
      --exclude 'csrc/' --exclude 'include/' --exclude 'lib/' --exclude 'share/' --exclude 'bin/' --exclude '__pycache__/' \
      --include '*/' --include '*.py' --include '*.yaml' --include '*.json' --include 'py.typed' --exclude '*' \
      "$src/$package/" "$stage/$package/"
  done
  cp "$module" "$stage/torch/"
  strip -x "$stage/torch/$(basename "$module")"
  cp "$here/LICENSE" "$stage/torch/LICENSE.torch-ios"
  cp "$src/LICENSE" "$stage/torch/LICENSE.pytorch"
  "$venv/bin/python" "$here/tools/make_wheel.py" "$stage" "$version" "$tag" "$dist"
}

case "$step" in
  fetch) fetch ;;
  configure) fetch; configure ;;
  build) build ;;
  wheel) wheel ;;
  all) fetch; configure; build; wheel ;;
  *) echo "Unknown step $step" >&2; exit 1 ;;
esac
