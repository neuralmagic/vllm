# Expert parallel kernels

Large-scale cluster-level expert parallel, as described in the [DeepSeek-V3 Technical Report](http://arxiv.org/abs/2412.19437), is an efficient way to deploy sparse MoE models with many experts. However, such deployment requires many components beyond a normal Python package, including system package support and system driver support. It is impossible to bundle all these components into a Python package.

Here we break down the requirements in 2 steps:

1. Build and install the Python libraries ([DeepEP](https://github.com/deepseek-ai/DeepEP)). The installer also initializes the bundled [DeepJIT](https://github.com/deepseek-ai/DeepJIT) submodule. This step does not require any privileged access. Any user can do it.
2. Configure NVIDIA driver to enable IBGDA. This step requires root access, and must be done on the host machine.

Step 2 is necessary for multi-node deployment.

`install_python_libraries.sh` stages its build under `$(pwd)/ep_kernels_workspace` by default. It accepts these options:

* `--workspace <dir>`: workspace directory for the DeepEP checkout and wheels.
* `--mode <mode>`: `install` (default) or `wheel`.
* `--deepep-ref <commit>`: DeepEP commit to build. The default is a full commit hash.

## DeepEP requirements

DeepEP 2.5 requires:

* Linux with Python 3.10 or newer
* NVIDIA Hopper or newer GPUs
* CUDA Toolkit 13.1 or newer, with support for the target GPU
* A C++20 compiler and standard library with `std::format` support
* PyTorch 2.10 or newer with CUDA support
* NCCL 2.32.3 or newer
* NVLink for intranode communication
* RDMA networking for internode communication

DeepEP upstream documents CUDA Toolkit 13.1 as its minimum supported version.
For validation only, this branch was also built and tested successfully with
CUDA Toolkit 13.0.88 on Hopper H200 GPUs.

The installer builds DeepEP's host extension against CUDA and NCCL. DeepJIT
compiles the GPU communication kernels for the current device at runtime, so
installation does not require a visible GPU or `TORCH_CUDA_ARCH_LIST`. Keep the
CUDA toolkit and host compiler available at runtime for DeepJIT compilation.

## NCCL version requirement for DeepEPv2

DeepEPv2 (`--all2all-backend deepep_v2`) uses the NCCL GIN (GPU-Initiated
Networking) backend, which requires NCCL >= 2.32.3 both when DeepEP is built
and at runtime. PyTorch pins an older release as a dependency (PyTorch 2.13
pins `nvidia-nccl-cu13==2.29.7`), so a plain install of vLLM is not enough.
The vLLM Docker images already override it.

Upgrade NCCL before running `install_python_libraries.sh`. Use
`nvidia-nccl-cu12` on CUDA 12. For a custom NCCL installation, set
`EP_NCCL_ROOT_DIR` to a directory containing `include/` and `lib/`.

**With uv** (recommended):

```bash
# Create an override file
echo "nvidia-nccl-cu13>=2.32.3" > /tmp/nccl-override.txt
export UV_OVERRIDE=/tmp/nccl-override.txt

# All subsequent uv pip install commands will respect the override
uv pip install vllm
```

Keep `UV_OVERRIDE` set for later installs into the same environment. Without
it, any `uv pip install` that resolves PyTorch again restores the pinned NCCL.

To fix an existing environment:

```bash
uv pip install "nvidia-nccl-cu13>=2.32.3" --no-deps
```

`install_python_libraries.sh` warns when the installed NCCL is too old. If
DeepEP was built against an older NCCL, upgrade NCCL and run the script again.
When NCCL is too old at runtime, vLLM logs the reason at startup and
`deepep_v2` fails with the same message. You can check with:

```bash
.venv/bin/python -c "from vllm.utils.import_utils import deep_ep_v2_unavailable_reason; print(deep_ep_v2_unavailable_reason() or 'deepep_v2 is available')"
```

## Usage

```bash
bash tools/ep_kernels/install_python_libraries.sh
```

Additional step for multi-node deployment:

```bash
sudo bash tools/ep_kernels/configure_system_drivers.sh # update-initramfs can take several minutes
sudo reboot # Reboot is required to load the new driver
```
