import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401
from vllm import envs


def test_awq_dequantize_opcheck():
    envs.VLLM_USE_TRITON_AWQ = False
    qweight = torch.randint(-2000000000,
                            2000000000, (8192, 256),
                            device='cuda',
                            dtype=torch.int32)
    scales = torch.rand((64, 2048), device='cuda', dtype=torch.float16)
    zeros = torch.empty((64, 256), device='cuda', dtype=torch.int32)
    split_k_iters = 0
    thx = 0
    thy = 0
    opcheck(torch.ops._C.awq_dequantize,
            (qweight, scales, zeros, split_k_iters, thx, thy))


def test_awq_gemm_opcheck():
    envs.VLLM_USE_TRITON_AWQ = False
    input = torch.rand((2, 8192), device='cuda', dtype=torch.float16)
    qweight = torch.randint(-2000000000,
                            2000000000, (8192, 256),
                            device='cuda',
                            dtype=torch.int32)
    scales = torch.randint(-2000000000,
                           2000000000, (64, 256),
                           device='cuda',
                           dtype=torch.int32)
    qzeros = torch.empty((64, 2048), device='cuda', dtype=torch.float16)
    split_k_iters = 8
    opcheck(torch.ops._C.awq_gemm,
            (input, qweight, qzeros, scales, split_k_iters))
