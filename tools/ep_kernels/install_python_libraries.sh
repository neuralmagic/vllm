#!/usr/bin/env bash
set -ex

<<<<<<< HEAD
# usage: ./install_python_libraries.sh [options]
#   --workspace <dir>    workspace directory (default: ./ep_kernels_workspace)
#   --mode <mode>        "install" (default) or "wheel"
#   --pplx-ref <commit>  pplx-kernels commit hash
#   --deepep-ref <commit> DeepEP commit hash

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PPLX_COMMIT_HASH=${PPLX_COMMIT_HASH:-"12cecfd"}
DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"73b6ea4"}
NVSHMEM_VER=3.3.9
WORKSPACE=${WORKSPACE:-$(pwd)/ep_kernels_workspace}
MODE=${MODE:-install}

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
        --pplx-ref)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: --pplx-ref requires an argument." >&2
                exit 1
            fi
            PPLX_COMMIT_HASH="$2"
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

=======
# prepare workspace directory
WORKSPACE=$1
if [ -z "$WORKSPACE" ]; then
    export WORKSPACE=$(pwd)/ep_kernels_workspace
fi

if [ ! -d "$WORKSPACE" ]; then
    mkdir -p $WORKSPACE
fi

# configurable pip command (default: pip3)
PIP_CMD=${PIP_CMD:-pip3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

>>>>>>> 31ad8d5ae (Deepep)
# install dependencies if not installed
$PIP_CMD install cmake torch ninja

# build nvshmem
pushd $WORKSPACE
mkdir -p nvshmem_src
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
tar -xvf nvshmem_src_3.2.5-1.txz -C nvshmem_src --strip-components=1
pushd nvshmem_src
wget https://github.com/deepseek-ai/DeepEP/raw/main/third-party/nvshmem.patch
git init
git apply -vvv nvshmem.patch

# assume CUDA_HOME is set correctly
if [ -z "$CUDA_HOME" ]; then
    echo "CUDA_HOME is not set, please set it to your CUDA installation directory."
    exit 1
fi

# assume TORCH_CUDA_ARCH_LIST is set correctly
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    echo "TORCH_CUDA_ARCH_LIST is not set, please set it to your desired architecture."
    exit 1
fi

# disable all features except IBGDA
export NVSHMEM_IBGDA_SUPPORT=1

export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_IBRC_SUPPORT=0
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0

cmake -G Ninja -S . -B $WORKSPACE/nvshmem_build/ -DCMAKE_INSTALL_PREFIX=$WORKSPACE/nvshmem_install
cmake --build $WORKSPACE/nvshmem_build/ --target install


popd

export CMAKE_PREFIX_PATH=$WORKSPACE/nvshmem_install:$CMAKE_PREFIX_PATH

is_git_dirty() {
    local dir=$1
    pushd "$dir" > /dev/null

    if [ -d ".git" ] && [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        popd > /dev/null
        return 0  # dirty (true)
    else
        popd > /dev/null
        return 1  # clean (false)
    fi
}

# Function to handle git repository cloning with dirty/incomplete checks
clone_repo() {
    local repo_url=$1
    local dir_name=$2
    local key_file=$3
    local commit_hash=$4

    if [ -d "$dir_name" ]; then
        # Check if directory has uncommitted changes (dirty)
        if is_git_dirty "$dir_name"; then
            echo "$dir_name directory is dirty, skipping clone"
        # Check if clone failed (directory exists but not a valid git repo or missing key files)
        elif [ ! -d "$dir_name/.git" ] || [ ! -f "$dir_name/$key_file" ]; then
            echo "$dir_name directory exists but clone appears incomplete, cleaning up and re-cloning"
            rm -rf "$dir_name"
            git clone "$repo_url"
            if [ -n "$commit_hash" ]; then
                cd "$dir_name"
                git checkout "$commit_hash"
                cd ..
            fi
        else
            echo "$dir_name directory exists and appears complete; manually update if needed"
        fi
    else
        git clone "$repo_url"
        if [ -n "$commit_hash" ]; then
            cd "$dir_name"
            git checkout "$commit_hash"
            cd ..
        fi
    fi
}

<<<<<<< HEAD
deepep_cuda13_patch() {
    cuda_version_major=$(${CUDA_HOME}/bin/nvcc --version | egrep -o "release [0-9]+" | cut -d ' ' -f 2)
    if [ ${cuda_version_major} -ge 13 ]; then
        sed -i "s|f'{nvshmem_dir}/include']|f'{nvshmem_dir}/include', '${CUDA_HOME}/include/cccl']|" "setup.py"
    fi
}

do_build() {
    local repo=$1
    local name=$2
    local key=$3
    local commit=$4
    local extra_env=$5

    pushd "$WORKSPACE"
    clone_repo "$repo" "$name" "$key" "$commit"
    cd "$name"

    if [ "$name" == "DeepEP" ]; then
        deepep_cuda13_patch
    fi

    if [ "$MODE" = "install" ]; then
        echo "Installing $name into environment"
        eval "$extra_env" uv pip install --no-build-isolation -vvv .
    else
        echo "Building $name wheel into $WHEEL_DIR"
        eval "$extra_env" uv build --wheel --no-build-isolation -vvv --out-dir "$WHEEL_DIR" .
    fi
    popd
}

# build pplx-kernels
do_build \
    "https://github.com/ppl-ai/pplx-kernels" \
    "pplx-kernels" \
    "setup.py" \
    "$PPLX_COMMIT_HASH" \
    ""

# build DeepEP
do_build \
    "https://github.com/elvircrn/DeepEP" \
    "DeepEP" \
    "setup.py" \
    "main" \
    "export NVSHMEM_DIR=$WORKSPACE/nvshmem; "

if [ "$MODE" = "wheel" ]; then
    echo "All wheels written to $WHEEL_DIR"
    ls -l "$WHEEL_DIR"
fi
=======
# build and install pplx, require pytorch installed
pushd $WORKSPACE
clone_repo "https://github.com/ppl-ai/pplx-kernels" "pplx-kernels" "setup.py" "c336faf"
cd pplx-kernels
$PIP_CMD install --no-build-isolation -vvv -e .
popd
>>>>>>> 31ad8d5ae (Deepep)

# build and install deepep, require pytorch installed
pushd $WORKSPACE
clone_repo "https://github.com/elvircrn/DeepEP" "DeepEP" "setup.py"
cd DeepEP
export NVSHMEM_DIR=$WORKSPACE/nvshmem_install
$PIP_CMD install --no-build-isolation -vvv -e .
popd
