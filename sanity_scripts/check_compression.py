"""
We are seeing some discrepancies between torch and vllm after decompression.
This script was created to check if the compression and decompression methods
are working as expected in vllm.

The script does the following:

- Checks CompressedTensors compression and decompression methods
against Alex's methods, from vllm (which should be identical)

- Next it checks the serialization and deserialization of the compressed weights

as of Monday, Nov 4th, 2024, the script is working as expected and does not fail
"""


from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import sparse_semi_structured_to_dense_cutlass, sparse_semi_structured_from_dense_cutlass
from vllm.model_executor.layers.sparsity.utils.cusparse_2_4_utils import generate_pruned_semi_structured_mat

import torch
from safetensors.torch import save_file, load_file

def sparse_compressor():
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors import CompressionFormat

    sparse_24_compressor = BaseCompressor.load_from_registry(
        CompressionFormat.sparse_24.value
    )

    return sparse_24_compressor


def check_compression(sparse_24_tensor, compressor=sparse_compressor()):

    alex_compressed, alex_meta = sparse_semi_structured_from_dense_cutlass(dense=sparse_24_tensor)
    compressor_result = compressor.compress_weight(name="test.weight", value=sparse_24_tensor)
    compressor_compressed, compressor_meta = compressor_result['test.weight_packed'], compressor_result['test.meta']

    # Bring the compressed tensors to the same device
    alex_compressed = alex_compressed.to(compressor_compressed.device)
    alex_meta = alex_meta.to(compressor_meta.device)

    # Check that the compressed tensors are the same
    assert torch.equal(alex_compressed, compressor_compressed)
    assert torch.equal(alex_meta, compressor_meta)

    # Check that the decompressed tensors are the same
    alex_decompressed = sparse_semi_structured_to_dense_cutlass(sparse=alex_compressed, meta_reordered=alex_meta)
    compressor_decompressed = compressor.decompress_weight(
        weight_data=dict(weight_packed=compressor_compressed, meta=compressor_meta)
    )

    assert torch.equal(alex_decompressed, compressor_decompressed)
    assert torch.equal(sparse_24_tensor, compressor_decompressed)
    assert torch.equal(sparse_24_tensor, alex_decompressed)


def check_serialization(sparse_24_tensor, compressor=sparse_compressor()):
    compressor_result = compressor.compress_weight(name="test.weight", value=sparse_24_tensor)
    
    # Serialize the compressed tensors
    save_path = "compressed_test.pt"
    save_file(compressor_result, save_path)

    # Deserialize the compressed tensors
    loaded_compressor_result = load_file(save_path)

    # Check that the compressed tensors are the same
    assert torch.equal(compressor_result['test.weight_packed'], loaded_compressor_result['test.weight_packed'])
    assert torch.equal(compressor_result['test.meta'], loaded_compressor_result['test.meta'])


if __name__ == "__main__":
    compressor = sparse_compressor()
    dtypes = [
        # torch.float16, 
        torch.bfloat16, 
        # torch.int8,
        # torch.float, # failing but expected
        # torch.float8_e4m3fn # not supported by alex's function
        ]
    M, K = 2048, 2048

    for dtype in dtypes:
        sparse_24_tensor = generate_pruned_semi_structured_mat(M, K, dtype).to("cpu")
        check_compression(sparse_24_tensor, compressor=compressor)
        print(f"Compression check for {dtype} successful!")

        check_serialization(sparse_24_tensor, compressor=compressor)
        print(f"Serialization check for {dtype} successful!")

