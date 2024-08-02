from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_tensor
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES, apply_gptq_marlin_linear,
    check_marlin_supports_shape, marlin_make_empty_g_idx,
    marlin_make_workspace, marlin_permute_scales,
    query_marlin_supported_quant_types)

from .MPLinearKernel import *


class MarlinLinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        if c.zero_points:
            return False, "Zero points currently not supported by "\
                          " Compressed Tensors + Marlin. (Kernel supports it"\
                          " but CompressedTensorsWNA16 does not so support has"\
                          " not been added to MarlinWNA16Kernel yet"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return False, f"Quant type ({c.weight_type}) not supported by"\
                          f"  Marlin, supported types are: {quant_types}"

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return False, f"Group size ({c.group_size}) not supported by "\
                            "Marlin, supported group sizes are: "\
                            f"{MARLIN_SUPPORTED_GROUP_SIZES}"

        return check_marlin_supports_shape(c.partition_weight_shape[0],
                                           c.partition_weight_shape[1],
                                           c.full_weight_shape[1],
                                           c.group_size)

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight_packed.device
        c = self.config

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace(c.partition_weight_shape[1],
                                               device)

        # Act-order not supported in compressed-tensors yet, so set to empty.
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.weight_zp = marlin_make_empty_g_idx(device)

        # Repack weights from compressed-tensors format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.weight_packed.t().contiguous(),
            perm=layer.g_idx_sort_indices,
            size_k=c.partition_weight_shape[0],
            size_n=c.partition_weight_shape[1],
            num_bits=c.weight_type.size_bits)
        replace_tensor(layer, "weight_packed", marlin_qweight)

        # Permute scales from compressed-tensors format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.weight_scale.squeeze().t().contiguous(),
            size_k=c.partition_weight_shape[0],
            size_n=c.partition_weight_shape[1],
            group_size=c.group_size)
        replace_tensor(layer, "weight_scale", marlin_scales)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.config

        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.weight_packed,
            weight_scale=layer.weight_scale,
            weight_zp=layer.weight_zp,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=True,
            bias=bias)
