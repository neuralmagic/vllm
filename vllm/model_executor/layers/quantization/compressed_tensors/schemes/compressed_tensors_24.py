from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.parameter import ModelWeightParameter, ChannelQuantScaleParameter
import torch
from typing import List, Callable, Optional
from compressed_tensors.compressors import ModelCompressor
from torch.nn import Parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import sparse_semi_structured_to_dense_cutlass, sparse_semi_structured_from_dense_cutlass
from vllm import _custom_ops as ops
from typing import Tuple

__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    def __init__(
            self, 
            model_compressor: Optional[ModelCompressor] = None, 
            layer_name: Optional[str] = None,
            quantized: bool = False,
            do_decompress: bool = False,
            ):
        
        self.model_compressor = model_compressor
        self.layer_name = layer_name
        self.quantized = quantized
        self.do_decompress = do_decompress

    @classmethod
    def get_min_capability(cls) -> int:
        """
        Since this scheme uses the cutlass library with FP8, it requires
        a minimum capability of 90

        :return: The minimum capability required for this scheme
        """
        return 90

    def create_weights(
            self, 
            layer: torch.nn.Module, 
            input_size: int,
            output_partition_sizes: List[int],
            input_size_per_partition: int,
            params_dtype: torch.dtype, weight_loader: Callable,
            **kwargs
            ):
        layer.logical_widths = output_partition_sizes
        self.params_dtype=params_dtype

        # parameter to store uncompressed weight or decompressed weight
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes),
                             input_size_per_partition,
                             dtype=params_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        
        if self.do_decompress:
            # store compression specific things to be used
            # later during decompression

            bits_per_weight_element = weight.itemsize * 8 
            meta_dtype = torch.int32 if bits_per_weight_element == 8 else torch.int16

            # compressed weight for 2:4 sparse
            weight_packed = ModelWeightParameter(data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader
                )

            meta_input_size = (
                input_size_per_partition // 32
                if bits_per_weight_element == 8
                else input_size_per_partition // 16
            )
            # meta tensor for 2:4 decompression
            meta = ModelWeightParameter(data=torch.empty(
                sum(output_partition_sizes), 
                meta_input_size,
                dtype=meta_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader)

            layer.register_parameter("weight_packed", weight_packed)
            layer.register_parameter("meta", meta)

        if self.quantized:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                dtype=torch.float32),
                                output_dim=0,
                                weight_loader=weight_loader)

            layer.register_parameter("weight_scale", weight_scale)

        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Apply any transformations to the weights after loading
        them from disk

        :param layer: The layer with the weights to be processed
        """

        decompressed_weight = (
            layer.weight if not self.do_decompress
            else self._decompress_24_weight(layer.weight_packed.data, layer.meta.data)
        )
        w_compressed, meta = ops.cutlass_compress_entry(decompressed_weight)
        layer.weight = torch.nn.Parameter(w_compressed, requires_grad=False)
        layer.meta = torch.nn.Parameter(meta, requires_grad=False)
        

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns the output tensor for the layer with 2:4 
        sparse compressed weights, given the input tensor
        and bias

        :param layer: The layer with 2:4 sparse compressed 
            weights to be used for the computation
        :param x: The input tensor to the layer
        :param bias: The bias to be added to the output tensor
        :return: The output tensor of the layer 
        """

        PAD_MULTIPLE = 16
        remainder = x.shape[0] % 16
        pad_size = PAD_MULTIPLE - remainder if remainder > 0 else 0

        q_input, input_scale = ops.scaled_fp8_quant(
            x, pad_to_multiple=PAD_MULTIPLE, use_per_token_if_dynamic=True)

        out = ops.cutlass_scaled_sparse_mm(
            a=layer.weight,
            e=layer.meta,
            b=q_input.t(),
            scale_a=layer.weight_scale,
            scale_b=input_scale,
            out_dtype=self.params_dtype,
            bias=bias
        )

        out = out.t()
        if pad_size > 0:
            out = out[:-pad_size,:].contiguous()
        else:
            out = out.contiguous()
        
        return out
    

    def _decompress_24_weight(self, weight_packed: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        qkv_sizes = [2048, 256, 256]
        gate_up_sizes = [5632, 5632]
        split_weights = None 
        split_meta = None

        def _process_split(input_weight, input_meta):
            weight_data = {
                "weight_packed": input_weight,
                "meta": input_meta
            }
            decompress = self.model_compressor.sparsity_compressor.decompress_weight(weight_data)
            return decompress

        print(self.layer_name)
        if "qkv" in self.layer_name:
            split_weights = torch.split(weight_packed, qkv_sizes)
            split_meta = torch.split(meta, qkv_sizes)
        elif "gate_up" in self.layer_name:
            split_weights = torch.split(weight_packed, gate_up_sizes)
            split_meta = torch.split(meta, gate_up_sizes)

        if split_weights:
            all_compress = []
            for i in range(len(split_weights)):
                print(split_weights[i].shape, split_meta[i].shape)
                compress_i = _process_split(split_weights[i], split_meta[i])
                all_compress.append(compress_i)

            decompressed = torch.cat(all_compress)
        else:
            decompressed = _process_split(weight_packed, meta)

        return decompressed



def check_24(tensor):
    new_tensor = tensor.view(-1, 4)    
    zero_counts = (new_tensor == 0).sum(dim=1)
    return (zero_counts >= 2).all().item()

