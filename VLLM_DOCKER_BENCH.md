## Benchmarking Using Docker Benchmarking Infrastructure

Describe how to run vllm benchmarks locally using the scripts under .buildkite directory. The reference HOWTO for
running the vllm buildkite benchmarks can be found at https://github.com/vllm-project/vllm/issues/8176


#### Download the necessary huggingface models

Create a models directory in your HOME directory.

$ mkdir ~/models

Create a venv so we can use the huggingface download utils safely

$ cd ~/models
$ python3 -m venv test
$ source test/bin/activate
$ pip3 install huggingface[cli]

Download the models as,

$ huggingface-cli download <model-tag> --local-dir ./<last-part-of-model-tag>

`huggingface-cli download meta-llama/Meta-Llama3-8B` --local-dir ./Meta-Llama3-8B

#### Build the docker image

`docker build -t vllm_bench -f Dockerfile.bench --build-arg torch_cuda_arch_list="9.0;9.0a"  --build-arg UID=$(id -u) --build-arg GID=$(id -g) .`

The docker build does the following steps,
  1. Sets up the build environments. i.e. installs the required vllm requirements*.txt
  2. Builds a pip wheel off the current state of the vllm directory
      - Passing the appropriate `torch_cuda_arch_list` is important. This is not set by default as
        default values encompasses all of the supported CUDA arch and the build times can be intolerable.
  3. Sets up the user home directory with the same UID and GID of the host user.

### Start the docker image

`docker run --shm-size=2g -it -d --net host --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus all -v $HOME/models:/models -v `pwd`/.buildkite:/home/docker-user/.buildkite -v `pwd`/benchmarks:/home/docker-user/benchmarks  --name vllm_bench vllm_bench:latest `

Note that we are mounting 3 volumes, 
 - The models folder where the models reside
 - The .buildkite folder where all the scripts are
 - The benchmarks folder where the client scripts are and where the results will be stored 

### Enter the docker shell

`docker exec -it vllm_bench /bin/bash`

### Start benchmarks
- Export your HF_TOKEN
- Export CUDA_VISIBLE_DEVICES

`VLLM_SOURCE_CODE_LOC=$(pwd) bash .buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh  <test-description-json>`
where <test-description-json> could be /home/docker-user/.buildkite/nightly-benchmarks/tests/nightly-tests-tp1.json - Make sure to give the absolute path.

## TRTLLM

There is also a Dockerfile.trtllm docker file for benchmarking TRTLLM

### Build the Docker image
```bash
docker build -t trtllm_bench -f Dockerfile.trtllm --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
``` 

### Run the docker image

```bash
GPUS='"device=0"'
GPUS='"device=0,1,2,3"'
docker run --shm-size=2g  -it -d --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus $GPUS -e HF_TOKEN=$HF_TOKEN -v $(pwd)/models:/models -v $(pwd)/.buildkite:/home/docker-user/.buildkite -v $(pwd)/benchmarks:/home/docker-user/benchmarks  --name trtllm_bench trtllm_bench:latest
```

### Enter the docker shell

```bash
docker exec -it trtllm_bench /bin/bash
```

### Start benchmarks
- Export your HF_TOKEN
- Export CUDA_VISIBLE_DEVICES

```bash
CONFIG_PATH=/home/docker-user/.buildkite/trt-configs/llama-8b.json
VLLM_SOURCE_CODE_LOC=$(pwd) bash .buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh $CONFIG_PATH
```


# ROB VERSION

### Build the Docker image
```bash
docker build -t trtllm_bench -f Dockerfile.trtllm --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
``` 

### Run the docker image

```bash
GPUS='"device=0"'
GPUS='"device=0,1,2,3"'
docker run --shm-size=2g  -it -d --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus $GPUS -e HF_TOKEN=$HF_TOKEN -v $(pwd)/models:/models -v $(pwd)/.buildkite:/home/docker-user/.buildkite -v $(pwd)/benchmarks:/home/docker-user/benchmarks  --name trtllm_bench trtllm_bench:latest
```

### Enter the docker shell

```bash
docker exec -it trtllm_bench /bin/bash
```

### Build TRT Engine

```bash
CONFIG_PATH=/home/docker-user/.buildkite/trt-configs/llama-8b.json

bash .buildkite/nightly-benchmarks/scripts/build-trt.sh $CONFIG_PATH
bash .buildkite/nightly-benchmarks/scripts/launch-trt.sh $CONFIG_PATH
```