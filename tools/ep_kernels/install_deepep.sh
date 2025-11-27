#!/usr/bin/env bash
set -ex

PIP_CMD="uv pip"

# prepare workspace directory
WORKSPACE=$1
if [ -z "$WORKSPACE" ]; then
    export WORKSPACE=$(pwd)/ep_kernels_workspace
fi

# build and install deepep, require pytorch installed
pushd $WORKSPACE
cd DeepEP
git pull origin main
export NVSHMEM_DIR=$WORKSPACE/nvshmem_install
$PIP_CMD install --no-build-isolation -vvv -e .
popd
