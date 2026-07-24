# CUDA 13.4 is currently a developer preview. The default Rubin build injects
# the cuda134-base stage from docker/Dockerfile wherever the otherwise-local
# vllm/cuda:13.4-devel image is referenced. Override these variables and the
# target contexts when public or preview-registry images become available.
# pytorch/pytorch#190639 is expected to publish this build image after merge:
# ghcr.io/pytorch/ci-image:pytorch-linux-jammy-cuda13.4-cudnn9-py3-gcc11
# Build from vllm-project/vllm#49387 until its SM107 changes reach main.
#
#   docker buildx bake --allow=network.host \
#     -f docker/docker-bake.hcl -f docker/docker-bake-rubin.hcl rubin
#
# Podman users can pass the equivalent defaults with:
#
#   podman build --platform linux/amd64 \
#     --build-arg-file docker/rubin-build-args.conf \
#     --target vllm-openai-rubin -t localhost/vllm:rubin-cu134 \
#     -f docker/Dockerfile .
#
# Use `--platform linux/arm64` and tag `localhost/vllm:rubin-cu134-arm64`
# when running the same command natively on a Vera Rubin ARM64 host.

variable "RUBIN_BUILD_BASE_IMAGE" {
  default = "vllm/cuda:13.4-devel"
}

variable "RUBIN_FINAL_BASE_IMAGE" {
  default = "vllm/cuda:13.4-devel"
}

variable "RUBIN_PYTORCH_CUDA_INDEX" {
  # CUDA 13.4 wheels are not published yet. This cu132 nightly is Rubin-aware
  # and carries regular sm_100 cubins, which are compatible with sm_107.
  default = "cu132"
}

variable "RUBIN_PYTORCH_NIGHTLY_PACKAGES" {
  default = "torch==2.14.0.dev20260723+cu132"
}

# The companion projects' July 23 wheels declare the July 22 torch nightly.
# Install them without dependency resolution, then validate both imports.
variable "RUBIN_PYTORCH_NIGHTLY_NO_DEPS_PACKAGES" {
  default = "torchaudio==2.11.0.dev20260723+cu132 torchvision==0.29.0.dev20260723+cu132"
}

variable "RUBIN_FLASHINFER_CUDA_INDEX" {
  # FlashInfer currently publishes CUDA 13 wheels through the cu130 channel.
  default = "cu130"
}

target "rubin-cuda-base" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  target     = "cuda134-base"
  network    = "host"
}

target "rubin" {
  inherits = ["_common", "_labels"]
  target   = "vllm-openai-rubin"
  tags     = ["vllm:rubin-cu134"]
  network  = "host"
  contexts = {
    "vllm/cuda:13.4-devel" = "target:rubin-cuda-base"
  }
  args = {
    CUDA_VERSION              = "13.4.0"
    BUILD_BASE_IMAGE          = RUBIN_BUILD_BASE_IMAGE
    FINAL_BASE_IMAGE          = RUBIN_FINAL_BASE_IMAGE
    PYTORCH_NIGHTLY           = "1"
    USE_SCCACHE               = "0"
    VLLM_VERSION_OVERRIDE     = "0.25.0.dev440+g1479bd9e9d.rubin"
    VLLM_MAX_SIZE_MB          = "550"
    PYTORCH_CUDA_INDEX        = RUBIN_PYTORCH_CUDA_INDEX
    PYTORCH_NIGHTLY_PACKAGES  = RUBIN_PYTORCH_NIGHTLY_PACKAGES
    PYTORCH_NIGHTLY_NO_DEPS_PACKAGES = RUBIN_PYTORCH_NIGHTLY_NO_DEPS_PACKAGES
    FLASHINFER_CUDA_INDEX     = RUBIN_FLASHINFER_CUDA_INDEX
    max_jobs                  = "96"
    nvcc_threads              = "4"
    torch_cuda_arch_list      = "10.7"
  }
  output = ["type=docker"]
}

target "rubin-arm64" {
  inherits  = ["rubin"]
  platforms = ["linux/arm64"]
  tags      = ["vllm:rubin-cu134-arm64"]
  args = {
    # setup.py divides MAX_JOBS by NVCC_THREADS. This yields six concurrent
    # NVCC processes, a safer default for an Apple Silicon Docker VM.
    max_jobs     = "24"
    nvcc_threads = "4"
  }
}
