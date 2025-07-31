#!/usr/bin/env bash

set -ex

# Build FlashInfer with AOT kernels
# This script is used by both the Dockerfile and standalone wheel building

FLASHINFER_GIT_REPO="${FLASHINFER_GIT_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
FLASHINFER_GIT_REF="${FLASHINFER_GIT_REF:-v0.2.9rc2}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
BUILD_WHEEL="${BUILD_WHEEL:-false}"

echo "🏗️  Building FlashInfer ${FLASHINFER_GIT_REF} for CUDA ${CUDA_VERSION}"

# Clone FlashInfer
git clone --depth 1 --recursive --shallow-submodules \
    --branch ${FLASHINFER_GIT_REF} \
    ${FLASHINFER_GIT_REPO} flashinfer

# Set CUDA arch list based on CUDA version
# Exclude CUDA arches for older versions (11.x and 12.0-12.7)
if [[ "${CUDA_VERSION}" == 11.* ]]; then
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
elif [[ "${CUDA_VERSION}" == 12.[0-7]* ]]; then
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
else
    # CUDA 12.8+ supports 10.0a and 12.0
    FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 12.0"
fi

echo "🏗️  Building FlashInfer for arches: ${FI_TORCH_CUDA_ARCH_LIST}"

# Build AOT kernels and install/build wheel
pushd flashinfer
    TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
        python3 -m flashinfer.aot
    
    if [[ "${BUILD_WHEEL}" == "true" ]]; then
        # Build wheel for distribution
        TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
            uv build --wheel .
        mkdir -p ../wheels
        for wheel in dist/*.whl; do
            if [[ -f "$wheel" ]]; then
                # Extract CUDA major.minor version (e.g., 12.8.1 -> cu128)
                cuda_tag="cu$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d .)"
                # Get original wheel name parts
                wheel_name=$(basename "$wheel")
                # Replace version with version+cuda_tag and fix platform tag
                new_wheel_name=$(echo "$wheel_name" | sed -E "s/(-[0-9]+\.[0-9]+\.[0-9]+[^-]*)/\1+${cuda_tag}/" | sed 's/linux_x86_64/manylinux1_x86_64/')
                # Copy with new name
                cp "$wheel" "../wheels/$new_wheel_name"
                echo "📦 Created wheel: $new_wheel_name"
            fi
        done
        echo "✅ FlashInfer wheel built successfully"
    else
        # Install directly (for Dockerfile)
        TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
            uv pip install --system --no-build-isolation --force-reinstall --no-deps .
        echo "✅ FlashInfer installed successfully"
    fi
popd

# Cleanup
rm -rf flashinfer