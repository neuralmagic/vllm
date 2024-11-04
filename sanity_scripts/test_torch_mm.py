import torch
import warnings
from vllm.model_executor.layers.sparsity.utils.cusparse_2_4_utils import (
    compress_to_torch_sparse_semi_structured_mat,
    generate_pruned_semi_structured_mat,
)

# Constants
N = 2048
M = 2560
P = 2048

def main(dtype=torch.float16):
    """
   Test torch.mm with packed and unpacked tensors.
    
    :params: dtype (torch.dtype): The data type for the tensors.
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    print(f"===== {dtype} =====")
    
    # Generate tensors
    sparse_tensor = generate_pruned_semi_structured_mat(N, M, dtype=dtype).cuda()  # N, M
    inputs_dense = torch.rand(N, M, dtype=dtype).cuda()  # N, M

    # Test matrix multiplication with unpacked sparse tensor
    try:
        torch.mm(sparse_tensor, inputs_dense)  # Should fail as dimensions are not compatible
        print(f"Torch mm passed for unpacked case {sparse_tensor.shape=}, {inputs_dense.shape=}")
    except Exception as e:
        print(f"Torch mm failed for unpacked case {sparse_tensor.shape=}, {inputs_dense.shape=}")

    # Test matrix multiplication with packed tensor
    try:
        sparse_tensor_packed = compress_to_torch_sparse_semi_structured_mat(sparse_tensor)  # N, M
        torch.mm(sparse_tensor_packed, inputs_dense)  # Passes even though dimensions are not compatible
        print(f"Torch mm passed for packed case {sparse_tensor_packed.shape=}, {inputs_dense.shape=}")
    except Exception as e:
        print(f"Torch mm failed for packed case {sparse_tensor_packed.shape=}, {inputs_dense.shape=}")

if __name__ == "__main__":
    main(dtype=torch.float16)
    main(dtype=torch.bfloat16)
