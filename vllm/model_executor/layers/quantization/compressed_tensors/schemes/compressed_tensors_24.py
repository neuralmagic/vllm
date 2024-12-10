from typing import Any, Callable, Dict, List, Optional

import torch
from compressed_tensors import ModelCompressor, CompressionFormat
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)
from compressed_tensors.utils import combine_shards

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter,
                                           _ColumnvLLMParameter as ColumnvLLMParameter,
                                           BitMaskShapeParameter,)

__all__ = ["CompressedTensors24"]


class CompressedTensors24(CompressedTensorsScheme):

    def __init__(self,
                 quantized: bool = False,
                 weight_quant: Optional[QuantizationArgs] = None,
                 input_quant: Optional[QuantizationArgs] = None,
                 model_compression_config: Optional[Dict[str, Any]] = None,
                 layer_name: Optional[str] = None # TODO: Remove
                 ):

        self.quantized = quantized
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.model_compressor = (
            ModelCompressor.from_compression_config(model_compression_config)
            if model_compression_config is not None else None)
        self.layer_name = layer_name # TODO: Remove
        self.do_sparse_decompress = self.model_compressor is not None

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        print("Creating weights for:", self.layer_name)
        self.output_dtype = params_dtype

        # This is needed for tensor scale 
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        self.weights_dtype: torch.dtype = self._get_params_dtype(params_dtype)

        # parameter to store uncompressed weight
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=self.weights_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        
        if self.do_sparse_decompress:
            sparsity_config = self.model_compressor.sparsity_config
            if sparsity_config is not None and sparsity_config.format != CompressionFormat.sparse_bitmask.value:
                raise ValueError("CompressedTensors24 only supports sparse_bitmask compression format")
            
            # register compression specific parameters

            shape = BitMaskShapeParameter(data=torch.empty(2 * len(output_partition_sizes), 1, dtype=torch.uint64), 
                                      weight_loader=weight_loader)
            compressed = ColumnvLLMParameter(data=torch.empty(
            sum(output_partition_size * input_size_per_partition  for output_partition_size in output_partition_sizes) // 2, 1,
            dtype=self.weights_dtype),
                                      output_dim=0,
                                      weight_loader=weight_loader)
            bitmask = ModelWeightParameter(data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 8,
                dtype=torch.uint8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
            row_offsets = ColumnvLLMParameter(data=torch.empty(
                sum(output_partition_sizes), 1, dtype=torch.uint64),
                output_dim=0,
                weight_loader=weight_loader,
            )

            layer.register_parameter("shape", shape)
            layer.register_parameter("compressed", compressed)
            layer.register_parameter("bitmask", bitmask)
            layer.register_parameter("row_offsets", row_offsets)

        # Check if quantized, not just 2:4 Sparse
        if self.quantized:
            if (self.weight_quant and self.weight_quant.strategy
                    == QuantizationStrategy.CHANNEL.value):
                weight_scale = ChannelQuantScaleParameter(
                    data=torch.empty((sum(output_partition_sizes), 1),
                                     dtype=torch.float32),
                    output_dim=0,
                    weight_loader=weight_loader)
            else:
                assert (self.weight_quant and self.weight_quant.strategy
                        == QuantizationStrategy.TENSOR.value)
                weight_scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes),
                                     dtype=torch.float32),
                    weight_loader=weight_loader)

            layer.register_parameter("weight_scale", weight_scale)

            # input quant will be non-none
            if self.input_quant and not self.input_quant.dynamic:
                # register input quant scale
                assert (self.input_quant.strategy ==
                        QuantizationStrategy.TENSOR.value)
                input_scale = BasevLLMParameter(data=torch.empty(
                    1, dtype=torch.float32),
                                                weight_loader=weight_loader)

                layer.register_parameter("input_scale", input_scale)

        else:
            # for sparse-only, pass in 1 for weight/input scales
            weight_scale = torch.nn.Parameter(data=torch.ones(
                1, dtype=torch.float32),
                                              requires_grad=False)
            input_scale = torch.nn.Parameter(data=torch.ones(
                1, dtype=torch.float32),
                                             requires_grad=False)
            layer.register_parameter("input_scale", input_scale)
            layer.register_parameter("weight_scale", weight_scale)

        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Compress weights after loading. Store compressed weight and meta
            tensor
        
        :post-condition: layer.w_compressed and layer.meta are
            set to the compressed weight and meta tensor in the
            format expected by the Cutlass kernels
        :param layer: The layer with the weights to be processed
        
        """
        print("Processing weights for:", self.layer_name)

        if self.do_sparse_decompress:
            layer.weight.data = self._decompress_bitmask_compressed_weight(
                layer.compressed, layer.shape, layer.bitmask, layer.row_offsets, layer=layer)

        # torch.compile workaround
        if hasattr(layer, "input_scale"):
            layer.input_scale = torch.nn.Parameter(layer.input_scale.data,
                                                   requires_grad=False)

        if self.weight_quant:
            if self.weight_quant.strategy == QuantizationStrategy.TENSOR.value:
                layer.weight_scale = torch.nn.Parameter(convert_to_channelwise(
                    weight_scale=layer.weight_scale,
                    logical_widths=layer.logical_widths),
                                                        requires_grad=False)
            else:
                # torch.compile workaround
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data, requires_grad=False)

        w_compressed, meta = ops.cutlass_compress_entry(layer.weight.data)
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
        if self.quantized:
            scale = None
            if hasattr(layer, "input_scale"):
                scale = layer.input_scale

            if self.weights_dtype == torch.int8:
                ops_output = ops.scaled_int8_quant(x, scale=scale)
                q_input = ops_output[0]
                input_scale = ops_output[1]
            else:
                assert self.weights_dtype == torch.float8_e4m3fn
                if scale is not None:
                    q_input, input_scale = ops.scaled_fp8_quant(x, scale=scale)
                else:
                    q_input, input_scale = ops.scaled_fp8_quant(
                        x, use_per_token_if_dynamic=True)

        else:
            # Not quantized, nothing to do with the input_scales, use as is
            input_scale = layer.input_scale
            q_input = x

        out = ops.cutlass_scaled_sparse_mm(a=layer.weight,
                                           e=layer.meta,
                                           b=q_input.t(),
                                           scale_a=layer.weight_scale,
                                           scale_b=input_scale,
                                           out_dtype=self.output_dtype,
                                           bias=bias)
        assert out.is_contiguous()
        return out

    def _get_params_dtype(self, params_dtype: torch.dtype) -> torch.dtype:
        if not self.quantized:
            return params_dtype

        assert self.weight_quant is not None
        assert self.input_quant is not None

        is_8_bits = self.weight_quant.num_bits == self.input_quant.num_bits == 8

        if not is_8_bits:
            raise ValueError("Cutlass only supports 8-bit quantization")

        if (self.weight_quant.type == QuantizationType.FLOAT
                and self.input_quant.type == QuantizationType.FLOAT):
            return torch.float8_e4m3fn

        if (self.weight_quant.type == QuantizationType.INT
                and self.input_quant.type == QuantizationType.INT):
            return torch.int8

        raise ValueError("Quantization type not supported by Cutlass")
    
    def _decompress_bitmask_compressed_weight(self, compressed, shape, bitmask, row_offsets, layer):
        print("Decompressing weights for:", self.layer_name)
        split_weights = None 
        def _process_split(bitmask_compressed_weight, shape, bitmask, row_offsets):
            weight_data = {
                "compressed": bitmask_compressed_weight,
                "shape": shape,
                "bitmask": bitmask,
                "row_offsets": row_offsets
            }
            decompress = self.model_compressor.sparsity_compressor.decompress_weight(weight_data)
            return decompress
        if isinstance(layer, (QKVParallelLinear, MergedColumnParallelLinear)):
            split_weights = torch.split(compressed, [partition_width * layer.input_size_per_partition // 2 for partition_width in layer.logical_widths])
            split_bitmask = torch.split(bitmask, layer.logical_widths)
            split_row_offsets = torch.split(row_offsets, layer.logical_widths)
            split_shape = torch.split(shape, [2] * len(layer.logical_widths))
        if split_weights:
            all_compress = []
            for i in range(len(split_weights)):
                compress_i = _process_split(split_weights[i], split_shape[i], split_bitmask[i], split_row_offsets[i])
                all_compress.append(compress_i)
            decompressed = combine_shards(all_compress)
        else:
            decompressed = self.model_compressor.sparsity_compressor.decompress_weight({
            "compressed": compressed,
            "shape": shape,
            "bitmask": bitmask,
            "row_offsets": row_offsets
        })
        return decompressed
        



def check_24(tensor):
    new_tensor = tensor.view(-1, 4)
    zero_counts = (new_tensor == 0).sum(dim=1)
    return (zero_counts >= 2).all().item()
