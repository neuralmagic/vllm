#!/usr/bin/env bash
set -ex

# usage: ./install_python_libraries.sh [options]
#   --workspace <dir>    workspace directory (default: ./ep_kernels_workspace)
#   --mode <mode>        "install" (default) or "wheel"
#   --deepep-ref <commit> DeepEP commit hash

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# Pinned in full: an abbreviated hash is not a ref, so a consumer that
# fetches the pin directly ("git fetch origin <sha>") cannot resolve it.
DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"f6aa38fe06cd4a0489e7730365d4e0114970a2f3"}
WORKSPACE=${WORKSPACE:-$(pwd)/ep_kernels_workspace}
MODE=${MODE:-install}
CUDA_VERSION_MAJOR=$("${CUDA_HOME}"/bin/nvcc --version | grep -E -o "release [0-9]+" | cut -d ' ' -f 2)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --workspace requires an argument." >&2
                exit 1
            fi
            WORKSPACE="$2"
            shift 2
            ;;
        --mode)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --mode requires an argument." >&2
                exit 1
            fi
            MODE="$2"
            shift 2
            ;;
        --deepep-ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --deepep-ref requires an argument." >&2
                exit 1
            fi
            DEEPEP_COMMIT_HASH="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$WORKSPACE"

WHEEL_DIR="$WORKSPACE/dist"
mkdir -p "$WHEEL_DIR"

pushd "$WORKSPACE"

# install dependencies if not installed
UV_PIP_TARGET=()
if [ -z "$VIRTUAL_ENV" ]; then
  UV_PIP_TARGET=(--system)
fi
uv pip install "${UV_PIP_TARGET[@]}" cmake torch ninja

# DeepEPv2 needs NCCL >= 2.32.3 at both build time and runtime, but PyTorch
# pins an older release.
DEEPEP_V2_MIN_NCCL="2.32.3"
NCCL_PACKAGE="nvidia-nccl-cu${CUDA_VERSION_MAJOR}"
NCCL_WARNING=""
NCCL_INSTALLED=$(uv pip show "${UV_PIP_TARGET[@]}" "$NCCL_PACKAGE" 2>/dev/null | sed -n 's/^Version: //p')
if [ -n "$NCCL_INSTALLED" ] && \
    [ "$(printf '%s\n' "$DEEPEP_V2_MIN_NCCL" "$NCCL_INSTALLED" | sort -V | head -n1)" != "$DEEPEP_V2_MIN_NCCL" ]; then
    NCCL_WARNING="WARNING: ${NCCL_PACKAGE} ${NCCL_INSTALLED} is installed, but the deepep_v2 backend
requires NCCL >= ${DEEPEP_V2_MIN_NCCL} when DeepEP is built and when it runs. To use deepep_v2, run
    uv pip install \"${NCCL_PACKAGE}>=${DEEPEP_V2_MIN_NCCL}\" --no-deps
and then run this script again. See tools/ep_kernels/README.md."
    echo "$NCCL_WARNING" >&2
fi

is_git_dirty() {
    local dir=$1
    pushd "$dir" > /dev/null
    if [ -d ".git" ] && [ -n "$(git status --porcelain 3>/dev/null)" ]; then
        popd > /dev/null
        return 0
    else
        popd > /dev/null
        return 1
    fi
}

clone_repo() {
    local repo_url=$1
    local dir_name=$2
    local key_file=$3
    local commit_hash=$4
    if [ -d "$dir_name" ]; then
        if is_git_dirty "$dir_name"; then
            echo "$dir_name directory is dirty, skipping clone"
        elif [ ! -d "$dir_name/.git" ] || [ ! -f "$dir_name/$key_file" ]; then
            echo "$dir_name directory exists but clone appears incomplete, cleaning up and re-cloning"
            rm -rf "$dir_name"
            git clone "$repo_url"
            if [ -n "$commit_hash" ]; then
                git -C "$dir_name" checkout "$commit_hash"
            fi
        else
            echo "$dir_name directory exists and appears complete"
            git -C "$dir_name" fetch origin
            if [ -n "$commit_hash" ]; then
                git -C "$dir_name" checkout "$commit_hash"
            fi
        fi
    else
        git clone "$repo_url"
        if [ -n "$commit_hash" ]; then
            git -C "$dir_name" checkout "$commit_hash"
        fi
    fi
}

do_build() {
    local repo=$1
    local name=$2
    local key=$3
    local commit=$4

    pushd "$WORKSPACE"
    clone_repo "$repo" "$name" "$key" "$commit"
    cd "$name"

    if [[ "$name" == "DeepEP" ]]; then
        git submodule update --init --recursive
        # DeepEP links against the CUDA driver API in driverless build images.
        local cuda_driver_stub
        local cuda_driver_stub_dir
        cuda_driver_stub=$(
            find -H "$CUDA_HOME" -path "*/stubs/libcuda.so" -print -quit
        )
        if [[ -z "$cuda_driver_stub" ]]; then
            echo "CUDA driver stub not found under $CUDA_HOME" >&2
            exit 1
        fi
        cuda_driver_stub_dir=$(dirname "$cuda_driver_stub")
        export LIBRARY_PATH="${cuda_driver_stub_dir}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi

    if [ "$MODE" = "install" ]; then
        echo "Installing $name into environment"
        uv pip install --no-build-isolation -vvv .
    else
        echo "Building $name wheel into $WHEEL_DIR"
        uv build --wheel --no-build-isolation -vvv --out-dir "$WHEEL_DIR" .
    fi
    popd
}

# build DeepEP
do_build \
    "https://github.com/deepseek-ai/DeepEP" \
    "DeepEP" \
    "setup.py" \
    "$DEEPEP_COMMIT_HASH"

if [ "$MODE" = "wheel" ]; then
    echo "All wheels written to $WHEEL_DIR"
    ls -l "$WHEEL_DIR"
fi

# Repeat after the verbose build output, where the first copy is easy to miss.
if [ -n "$NCCL_WARNING" ]; then
    echo "$NCCL_WARNING" >&2
fi
