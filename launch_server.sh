rm -rf /opt/tritonserver/backends/vllm
mkdir -p /opt/tritonserver/backends/vllm
cp -r /work/src/* /opt/tritonserver/backends/vllm

tritonserver --model-repository /work/model_repository