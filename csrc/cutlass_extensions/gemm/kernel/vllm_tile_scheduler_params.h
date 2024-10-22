
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

namespace cutlass::gemm::kernel::detail {

struct vLLMPersistentTileSchedulerSm90StreamKParams
    : PersistentTileSchedulerSm90StreamKParams {
  using UnderlyingParams = PersistentTileSchedulerSm90StreamKParams;
  using ReductionMode = UnderlyingParams::ReductionMode;
  using RasterOrder = UnderlyingParams::RasterOrder;
  using RasterOrderOptions = UnderlyingParams::RasterOrderOptions;
  using DecompositionMode = UnderlyingParams::DecompositionMode;

  void* barrier_workspace_;

  UnderlyingParams::get_grid_shape;
  UnderlyingParams::get_workspace_component_sizes;
  UnderlyingParams::initialize_workspace;
  UnderlyingParams::initialize;
  UnderlyingParams::requires_separate_reduction;

  // clang-format off
  // Get the amount of scratch workspace needed for the kernel. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  static int
  get_workspace_size(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile) {

    dim3 problem_blocks = UnderlyingParams::UnderlyingParams::get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    uint32_t k_tiles_per_output_tile = (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

    return get_workspace_size(
      problem_blocks,
      k_tiles_per_output_tile,
      tile_shape,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      decomposition_mode,
      mma_warp_groups,
      barrier_bits,
      element_accumulator_bits,
      epilogue_subtile
    );
  }

  // Version of get_workspace_size that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  static int
  get_workspace_size(
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    size_t splits,
    size_t max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile = 1) {

    size_t barrier_workspace_size = 0;
    size_t reduction_workspace_size = 0;

    #if !defined(__CUDACC_RTC__)
      get_workspace_component_sizes(
        problem_blocks,
        k_tiles_per_output_tile,
        tile_shape,
        cluster_shape,
        barrier_workspace_size,
        reduction_workspace_size,
        hw_info,
        splits,
        max_swizzle,
        raster_order_option,
        decomposition_mode,
        mma_warp_groups,
        barrier_bits,
        element_accumulator_bits,
        epilogue_subtile
      );
    #endif

    return barrier_workspace_size + reduction_workspace_size;
  }
  // clang-format on

  // clang-format off
  static int
  get_barrier_workspace_size(
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    size_t splits,
    size_t max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile = 1) {

    size_t barrier_workspace_size = 0;
    size_t reduction_workspace_size = 0;

    #if !defined(__CUDACC_RTC__)
      get_workspace_component_sizes(
        problem_blocks,
        k_tiles_per_output_tile,
        tile_shape,
        cluster_shape,
        barrier_workspace_size,
        reduction_workspace_size,
        hw_info,
        splits,
        max_swizzle,
        raster_order_option,
        decomposition_mode,
        mma_warp_groups,
        barrier_bits,
        element_accumulator_bits,
        epilogue_subtile
      );
    #endif

    return barrier_workspace_size;

  }
  // clang-format on

  // clang-format off
  // Version of initialize_workspace that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  static cutlass::Status
  initialize_workspace(
    void* workspace,
    cudaStream_t stream,
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile = 1) {

    return Status::kSuccess;
  }
  // clang-format on
};

}  // namespace cutlass::gemm::kernel::detail