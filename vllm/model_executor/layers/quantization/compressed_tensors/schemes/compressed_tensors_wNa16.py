from typing import Callable, List, Optional, Tuple, Type

import torch

from abc import ABC

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear, marlin_make_empty_g_idx, marlin_make_workspace,
    marlin_permute_scales, replace_tensor, verify_marlin_supported,
    verify_marlin_supports_shape)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.scalar_type import scalar_types

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128,
}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class QuantLinearKernel(ABC):
    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError("apply_weights not implemented")
    
    @classmethod
    def can_implement(cls,
                      full_weight_shape: Tuple[int, int], # [in, out]
                      partition_weight_shape: Tuple[int, int],
                      quant_type: ScalarType,
                      act_type: torch.dtype,
                      group_size: int,
                      zero_points: bool, 
                      act_reordering: bool) -> Tuple[bool, str]:
        raise NotImplementedError("can_implement not implemented")
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("apply_weights not implemented")

class MarlinKernel(QuantLinearKernel):
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 80
    
    @classmethod
    def can_implement(cls,
                      full_weight_shape: Tuple[int, int], # [in, out]
                      partition_weight_shape: Tuple[int, int],
                      quant_type: ScalarType,
                      act_type: torch.dtype,
                      group_size: int,
                      zero_points: bool, 
                      act_reordering: bool) -> Tuple[bool, str]:
        
        if zero_points:
            return False, "Zero points currently not supported by "\
                          " Compressed Tensors + Marlin. (Kernel supports it"\
                          " but CompressedTensorsWNA16 does not so support has"\
                          " not been addes to MarlinWNA16Kernel yet"
    
        if quant_type not in query_marlin_supported_quant_types(zero_points):
            return False, f"Quant type ({quant_type}) not supported by Marlin,"\
                           " supported types are: "\
                           f"{query_marlin_supported_quant_types(zero_points)}"
        
        if group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return False, f"Group size ({group_size}) not supported by Marlin,"\
                            " supported group sizes are: "\
                            f"{MARLIN_SUPPORTED_GROUP_SIZES}"
        
        return check_marlin_supports_shape(
            partition_weight_shape[0], 
            partition_weight_shape[1], 
            full_weight_shape[1], 
            group_size)
    
    # Checkpoints are serialized in compressed-tensors format, which is
    # different from marlin format. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight_packed.device

        # Allocate marlin workspace.
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Act-order not supported in compressed-tensors yet, so set to empty.
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.weight_zp = marlin_make_empty_g_idx(device)

        # Repack weights from compressed-tensors format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.weight_packed.t().contiguous(),
            perm=layer.g_idx_sort_indices,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=layer.weight_type.size_bits)
        replace_tensor(layer, "weight_packed", marlin_qweight)

        # Permute scales from compressed-tensors format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.weight_scale.squeeze().t().contiguous(),
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=layer.group_size)
        replace_tensor(layer, "weight_scale", marlin_scales)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        print("running marlin")
        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.weight_packed,
            weight_scale=layer.weight_scale,
            weight_zp=layer.weight_zp,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            wtype=layer.weight_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            is_k_full=True,
            bias=bias)


class MacheteKernel(QuantLinearKernel):
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 90
    
    @classmethod
    def can_implement(cls,
                    full_weight_shape: Tuple[int, int], # [in, out]
                    partition_weight_shape: Tuple[int, int],
                    quant_type: ScalarType,
                    act_type: torch.dtype,
                    group_size: int,
                    zero_points: bool, 
                    act_reordering: bool) -> Tuple[bool, str]:
        if act_reordering:
            return False, "Act reordering currently not supported by Machete"
        
        if zero_points:
            return False, "Zero points currently not supported by "\
                          " Compressed Tensors + Machete. (Kernel supports it"\
                          " but CompressedTensorsWNA16 does not so support has"\
                          " not been addes to MacheteWNA16Kernel yet"
        
        if quant_type not in query_machete_supported_quant_types(zero_points):
            return False, f"Quant type ({quant_type}) not supported by "\
                           "Machete, supported types are: "\
                           f"{query_machete_supported_quant_types(zero_points)}"
        
        if group_size not in MACHETE_SUPPORTED_GROUP_SIZES:
            return False, f"Group size ({group_size}) not supported by "\
                            "Machete, supported group sizes are: "\
                            f"{MACHETE_SUPPORTED_GROUP_SIZES}"
        
        return check_machete_supports_shape(
            partition_weight_shape[0], 
            partition_weight_shape[1])
    

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from marlin format. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        machete_qweight = ops.machete_prepack_B(
            layer.weight_packed.t(),
            layer.weight_type
        )
        replace_tensor(layer, "weight_packed", machete_qweight)
        replace_tensor(layer, "weight_scale", 
                       layer.weight_scale.clone().t())
        

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert layer.weight_scale.dtype == x.dtype
        # print("running machete")
        # print(layer.weight_packed.shape)
        # print(layer.weight_scale.dtype, x.dtype, layer.group_size)
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer.output_size_per_partition, )

        output = ops.machete_gemm(
            a=x_2d,
            b_q=layer.weight_packed,
            b_type=layer.weight_type,
            b_zeros=None,
            b_scales=layer.weight_scale,
            b_group_size=layer.group_size
        )
        
        if bias is not None:
            output.add_(bias)  # In-place add
            
        return output.reshape(out_shape)

class CompressedTensorsWNA16(CompressedTensorsScheme):
    
    # in order of priority (i.e. performance if available)
    possible_kernels: List[Type[QuantLinearKernel]] = [
        MacheteKernel,
        #MarlinKernel,
    ]

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None):

        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError("Marlin kernels require group quantization or "
                             "channelwise quantization, but found no group "
                             "size and strategy is not channelwise.")

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")

        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[num_bits]

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, 
                       output_size: int, 
                       input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, 
                       weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        
        failure_reasons = []
        
        full_weight_shape = (input_size, output_size)
        partition_weight_shape = \
            (input_size_per_partition, output_size_per_partition)

        capability = current_platform.get_device_capability()
        capability = capability[0] * 10 + capability[1]

        for kernel in self.possible_kernels:
            if kernel.get_min_capability() > capability:
                failure_reasons.append(
                    (kernel.__name__, 
                     f"requires capability {kernel.get_min_capability()}, "
                     f"current capability is {capability}"))
            
            can_implement, failure_reason = kernel.can_implement(
                    full_weight_shape=full_weight_shape,
                    partition_weight_shape=partition_weight_shape,
                    quant_type=self.quant_type,
                    act_type=params_dtype,
                    group_size=self.group_size,
                    zero_points=False,
                    act_reordering=False)

            if can_implement:
                self.kernel = kernel()
                break
            else:
                failure_reasons.append(
                    (kernel.__name__, failure_reason))
                
        if not hasattr(self, "kernel"):
            raise ValueError(
                f"Failed to find a kernel that can implement the "\
                "WNA16 linear layer. Reasons: \n"
                + '\n'.join([f'  {x} cannot implement due to: {r}'
                               for x, r in failure_reasons]))

        # If group_size is -1, we are in channelwise case.
        channelwise = (self.group_size == -1)
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        # In the case of channelwise quantization, we need to replicate the
        # scales across all gpus.
        partition_scales = (row_parallel and not channelwise)

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size)

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=self.pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         input_size_per_partition //
                                         self.pack_factor,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }
        if self.group_size == -1:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)
        else:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.group_size = group_size

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight_packed.device

        # Allocate marlin workspace.
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Act-order not supported in compressed-tensors yet, so set to empty.
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.weight_zp = marlin_make_empty_g_idx(device)
        # Update for kernel
        layer.weight_packed = torch.nn.Parameter(
            layer.weight_packed.t().contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.squeeze().t().contiguous(), requires_grad=False)

        # Repack weights from compressed-tensors format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.weight_packed,
            perm=layer.g_idx_sort_indices,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_type.size_bits)
        replace_tensor(layer, "weight_packed", marlin_qweight)

        # Permute scales from compressed-tensors format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.weight_scale,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=layer.group_size)
        replace_tensor(layer, "weight_scale", marlin_scales)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)