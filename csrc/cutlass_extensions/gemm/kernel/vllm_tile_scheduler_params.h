
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

namespace cutlass::gemm::kernel::detail {

struct vLLMPersistentTileSchedulerSm90StreamKParams {
  using UnderlyingParams = PersistentTileSchedulerSm90StreamKParams;
  using ReductionMode = UnderlyingParams::ReductionMode;
  using RasterOrder = UnderlyingParams::RasterOrder;
  using RasterOrderOptions = UnderlyingParams::RasterOrderOptions;
  using DecompositionMode = UnderlyingParams::DecompositionMode;

  UnderlyingParams underlying_params;
  void* barrier_workspace = nullptr;

  // Given the inputs, computes the physical grid we should launch.
  // This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(BatchedGemmCoord problem_shape,
                             GemmCoord cta_shape, GemmCoord cluster_shape,
                             KernelHardwareInfo hw_info, int max_swizzle_size,
                             RasterOrderOptions raster_order_option) {
    return UnderlyingParams::get_grid_shape(
        problem_shape, cta_shape, cluster_shape, hw_info, max_swizzle_size,
        raster_order_option);
  };

  // Version of get_grid_shape that takes in as input the number of CTAs in the
  // M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem
  // and/or CTA shape has rank > 1, for which using CuTe algebra for calculating
  // tile shapes is easiest.
  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(dim3 problem_blocks, GemmCoord cluster_shape,
                             KernelHardwareInfo hw_info, int max_swizzle_size,
                             RasterOrderOptions raster_order_option) {
    return UnderlyingParams::get_grid_shape(problem_blocks, cluster_shape,
                                            hw_info, max_swizzle_size,
                                            raster_order_option);
  };

#if !defined(__CUDACC_RTC__)
  static constexpr auto get_workspace_component_sizes =
      &UnderlyingParams::get_workspace_component_sizes;
#endif  // !defined(__CUDACC_RTC__)
};

}  // namespace cutlass::gemm::kernel::detail