from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter
import torch
from typing import List, Callable, Optional
from compressed_tensors.compressors import ModelCompressor
from torch.nn import Parameter
from vllm.model_executor.layers.sparsity.utils.cusparse_2_4_utils import (
    compress_to_torch_sparse_semi_structured_mat,
    semi_structured_sparse_dense_gemm,
    semi_structured_sparse_dense_gemm_scaled,
    )

__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    def __init__(self, model_compressor: Optional[ModelCompressor] = None):
        self.model_compressor = model_compressor
        
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                    output_partition_sizes: List[int],
                    input_size_per_partition: int,
                    params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
    
        
        # assume fp8 for now
        weight_dtype = torch.float8_e4m3fn

        # packed dim is dim 1/along input dim
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition // 2,
            dtype=weight_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)


        meta_dtype_map = {
            torch.int8: torch.int32,
            torch.float8_e4m3fn: torch.int32,
            torch.half: torch.int16,
            torch.bfloat16: torch.int16,
            torch.float16: torch.int16,
            torch.float: torch.int16,
            torch.float32: torch.int16,
            torch.int32: torch.int16,
        }

        meta_dtype = meta_dtype_map[weight.dtype]
        
        meta_input_size = (
            input_size_per_partition // 32 
            if meta_dtype == torch.int32
            else input_size_per_partition // 16
             )

        # meta dim changes based on dtype
        meta = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes), 
            meta_input_size,
            dtype=meta_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        # assume per tensor static quantization
        weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)
         # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)
        
        input_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
        input_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("input_scale", input_scale)


        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("meta", meta)
        

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Any preprocessing for the kernel
        # e.g mapp per tensor scales to channel
        # apply marlin format to the weights before kernel call
        # decompress

        if hasattr(layer, "weight_packed"):
            weight = layer.weight_packed.data
            meta = layer.meta.data

            # decompress

            weight_data = {
                "weight_packed": weight,
                "meta": meta
            }

            decompressed_weight = self.model_compressor.sparsity_compressor.decompress_weight(weight_data)
            decompressed_weight = decompressed_weight.t().contiguous()
            compressed = compress_to_torch_sparse_semi_structured_mat(decompressed_weight)
            layer.weight_packed = Parameter(compressed, requires_grad=False)
            
             

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        weight = layer.weight_packed.data
        input_scale = layer.input_scale.data
        weight_scale = layer.weight_scale.data


        # apply the kernel
        output =  semi_structured_sparse_dense_gemm_scaled(
            a_packed=weight,
            b_dense=x,
            scale_a=weight_scale,
            scale_b=input_scale,
            bias=bias
        )        
        return output.t().contiguous()



                