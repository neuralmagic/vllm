
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

namespace cutlass::gemm::kernel::detail {

struct vLLMPersistentTileSchedulerSm90StreamKParams
    : PersistentTileSchedulerSm90StreamKParams {
  using UnderlyingParams = PersistentTileSchedulerSm90StreamKParams;
  using ReductionMode = UnderlyingParams::ReductionMode;
  using RasterOrder = UnderlyingParams::RasterOrder;
  using RasterOrderOptions = UnderlyingParams::RasterOrderOptions;
  using DecompositionMode = UnderlyingParams::DecompositionMode;

  void* barrier_workspace = nullptr;

  UnderlyingParams::get_grid_shape;
  UnderlyingParams::get_workspace_size;
  UnderlyingParams::initialize_workspace;
  UnderlyingParams::initialize;
  UnderlyingParams::requires_separate_reduction;

  // // Given the inputs, computes the physical grid we should launch.
  // // This variant of the method should only be used when
  // // problem_shape and tile_shape contain modes of only rank 1.
  // CUTLASS_HOST_DEVICE
  // static dim3 get_grid_shape(BatchedGemmCoord problem_shape,
  //                            GemmCoord cta_shape, GemmCoord cluster_shape,
  //                            KernelHardwareInfo hw_info, int
  //                            max_swizzle_size, RasterOrderOptions
  //                            raster_order_option) {
  //   return UnderlyingParams::get_grid_shape(
  //       problem_shape, cta_shape, cluster_shape, hw_info, max_swizzle_size,
  //       raster_order_option);
  // };

  // // Version of get_grid_shape that takes in as input the number of CTAs in
  // the
  // // M and N and L dimensions.
  // // This is useful for calculating the tiled shape when a mode of problem
  // // and/or CTA shape has rank > 1, for which using CuTe algebra for
  // calculating
  // // tile shapes is easiest.
  // CUTLASS_HOST_DEVICE
  // static dim3 get_grid_shape(dim3 problem_blocks, GemmCoord cluster_shape,
  //                            KernelHardwareInfo hw_info, int
  //                            max_swizzle_size, RasterOrderOptions
  //                            raster_order_option) {
  //   return UnderlyingParams::get_grid_shape(problem_blocks, cluster_shape,
  //                                           hw_info, max_swizzle_size,
  //                                           raster_order_option);
  // };

  // // clang-format off
  // // Get the amount of scratch workspace needed for the kernel. This variant
  // of the method should only be used when
  // // problem_shape and tile_shape contain modes of only rank 1.
  // // Get the amount of scratch workspace needed for the kernel. This variant
  // of the method should only be used when
  // // problem_shape and tile_shape contain modes of only rank 1.
  // static size_t
  // get_workspace_size(
  //   BatchedGemmCoord problem_shape,
  //   GemmCoord tile_shape,
  //   GemmCoord cluster_shape,
  //   KernelHardwareInfo const& hw_info,
  //   int splits,
  //   int max_swizzle,
  //   RasterOrderOptions raster_order_option,
  //   DecompositionMode decomposition_mode,
  //   uint32_t mma_warp_groups,
  //   uint32_t barrier_bits,
  //   uint32_t element_accumulator_bits,
  //   uint32_t epilogue_subtile,
  //   uint32_t num_accumulator_mtxs) {

  //   dim3 problem_blocks =
  //   UnderlyingParams::UnderlyingParams::get_tiled_cta_shape_mnl(problem_shape,
  //   tile_shape, cluster_shape); uint32_t k_tiles_per_output_tile =
  //   (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

  //   return get_workspace_size(
  //     problem_blocks,
  //     k_tiles_per_output_tile,
  //     tile_shape,
  //     cluster_shape,
  //     hw_info,
  //     splits,
  //     max_swizzle,
  //     raster_order_option,
  //     decomposition_mode,
  //     mma_warp_groups,
  //     barrier_bits,
  //     element_accumulator_bits,
  //     epilogue_subtile,
  //     num_accumulator_mtxs
  //   );
  // }

  // // Version of get_workspace_size that takes in as input the number of CTAs
  // in the M and N dimensions.
  // // This is useful for calculating the tiled shape when a mode of problem
  // and/or CTA shape has rank > 1,
  // // for which using CuTe algebra for calculating tile shapes is easiest.
  // static size_t
  // get_workspace_size(
  //   dim3 problem_blocks,
  //   uint32_t k_tiles_per_output_tile,
  //   GemmCoord tile_shape,
  //   GemmCoord cluster_shape,
  //   KernelHardwareInfo const& hw_info,
  //   int splits,
  //   int max_swizzle,
  //   RasterOrderOptions raster_order_option,
  //   DecompositionMode decomposition_mode,
  //   uint32_t mma_warp_groups,
  //   uint32_t barrier_bits,
  //   uint32_t element_accumulator_bits,
  //   uint32_t epilogue_subtile = 1,
  //   uint32_t num_accumulator_mtxs = 1) {

  //   size_t barrier_workspace_size = 0;
  //   size_t reduction_workspace_size = 0;

  //   #if !defined(__CUDACC_RTC__)
  //     UnderlyingParams::get_workspace_component_sizes(
  //       problem_blocks,
  //       k_tiles_per_output_tile,
  //       tile_shape,
  //       cluster_shape,
  //       barrier_workspace_size,
  //       reduction_workspace_size,
  //       hw_info,
  //       splits,
  //       max_swizzle,
  //       raster_order_option,
  //       decomposition_mode,
  //       mma_warp_groups,
  //       barrier_bits,
  //       element_accumulator_bits,
  //       epilogue_subtile,
  //       num_accumulator_mtxs
  //     );
  //   #endif

  //   return barrier_workspace_size + reduction_workspace_size;
  // }
  // // clang-format on

  // // clang-format off
  // void
  // initialize(
  //   BatchedGemmCoord problem_shape,
  //   GemmCoord tile_shape,
  //   GemmCoord cluster_shape,
  //   KernelHardwareInfo hw_info,
  //   int splits,
  //   int max_swizzle,
  //   RasterOrderOptions raster_order_option,
  //   ReductionMode reduction_mode,
  //   DecompositionMode decomposition_mode,
  //   void* workspace,
  //   const uint32_t epilogue_subtile = 1
  // ) {
  //   dim3 problem_blocks =
  //     UnderlyingParams::UnderlyingParams::get_tiled_cta_shape_mnl(
  //       problem_shape, tile_shape, cluster_shape);

  //   // Number of k tiles in each output tile
  //   uint32_t k_tiles_per_output_tile = (problem_shape.k() + tile_shape.k() -
  //   1) / tile_shape.k();

  //   initialize(
  //     problem_blocks,
  //     k_tiles_per_output_tile,
  //     cluster_shape,
  //     hw_info,
  //     splits,
  //     max_swizzle,
  //     raster_order_option,
  //     reduction_mode,
  //     decomposition_mode,
  //     workspace,
  //     epilogue_subtile
  //   );
  // }

  // // Version of initialize that takes in as input the number of CTAs in the M
  // and N and L dimensions.
  // // This is useful for calculating the tiled shape when a mode of problem
  // and/or CTA shape has rank > 1,
  // // for which using CuTe algebra for calculating tile shapes is easiest.
  // void
  // initialize(
  //   dim3 problem_blocks,
  //   uint32_t k_tiles_per_output_tile,
  //   GemmCoord cluster_shape,
  //   KernelHardwareInfo hw_info,
  //   int splits,
  //   int max_swizzle,
  //   RasterOrderOptions raster_order_option,
  //   ReductionMode reduction_mode,
  //   DecompositionMode decomposition_mode,
  //   void* workspace,
  //   const uint32_t epilogue_subtile = 1
  // ) {
  //   UnderlyingParams underlying_params;
  //   underlying_params.initialize(
  //     problem_blocks,
  //     k_tiles_per_output_tile,
  //     cluster_shape,
  //     hw_info,
  //     splits,
  //     max_swizzle,
  //     raster_order_option,
  //     reduction_mode,
  //     decomposition_mode,
  //     workspace,
  //     epilogue_subtile
  //   );
  // }
  // // clang-format on

  // // clang-format off
  // // Initialize the workspace to be used for the kernel. This variant of the
  // method should only be used when
  // // problem_shape and tile_shape contain modes of only rank 1.
  // static cutlass::Status
  // initialize_workspace(
  //   void* workspace,
  //   cudaStream_t stream,
  //   BatchedGemmCoord problem_shape,
  //   GemmCoord tile_shape,
  //   GemmCoord cluster_shape,
  //   KernelHardwareInfo const& hw_info,
  //   int splits,
  //   int max_swizzle,
  //   RasterOrderOptions raster_order_option,
  //   DecompositionMode decomposition_mode,
  //   uint32_t mma_warp_groups,
  //   uint32_t barrier_bits,
  //   uint32_t element_accumulator_bits,
  //   uint32_t epilogue_subtile,
  //   CudaHostAdapter* cuda_adapter = nullptr) {

  //   dim3 problem_blocks =
  //   UnderlyingParams::UnderlyingParamsget_tiled_cta_shape_mnl(problem_shape,
  //   tile_shape, cluster_shape); uint32_t k_tiles_per_output_tile =
  //   (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

  //   return initialize_workspace(
  //     workspace,
  //     stream,
  //     problem_blocks,
  //     k_tiles_per_output_tile,
  //     tile_shape,
  //     cluster_shape,
  //     hw_info,
  //     splits,
  //     max_swizzle,
  //     raster_order_option,
  //     decomposition_mode,
  //     mma_warp_groups,
  //     barrier_bits,
  //     element_accumulator_bits,
  //     epilogue_subtile,
  //     1,
  //     cuda_adapter
  //   );
  // }

  // // Version of initialize_workspace that takes in as input the number of
  // CTAs in the M and N dimensions.
  // // This is useful for calculating the tiled shape when a mode of problem
  // and/or CTA shape has rank > 1,
  // // for which using CuTe algebra for calculating tile shapes is easiest.
  // static cutlass::Status
  // initialize_workspace(
  //   void* workspace,
  //   cudaStream_t stream,
  //   dim3 problem_blocks,
  //   uint32_t k_tiles_per_output_tile,
  //   GemmCoord tile_shape,
  //   GemmCoord cluster_shape,
  //   KernelHardwareInfo const& hw_info,
  //   int splits,
  //   int max_swizzle,
  //   RasterOrderOptions raster_order_option,
  //   DecompositionMode decomposition_mode,
  //   uint32_t mma_warp_groups,
  //   uint32_t barrier_bits,
  //   uint32_t element_accumulator_bits,
  //   uint32_t epilogue_subtile = 1,
  //   uint32_t num_accumulator_mtxs = 1,
  //   CudaHostAdapter* cuda_adapter = nullptr) {

  //   #if !defined(__CUDACC_RTC__)
  //     uint64_t barrier_workspace_size = 0;
  //     uint64_t reduction_workspace_size = 0;

  //     UnderlyingParams::get_workspace_component_sizes(
  //       problem_blocks,
  //       k_tiles_per_output_tile,
  //       tile_shape,
  //       cluster_shape,
  //       barrier_workspace_size,
  //       reduction_workspace_size,
  //       hw_info,
  //       splits,
  //       max_swizzle,
  //       raster_order_option,
  //       decomposition_mode,
  //       mma_warp_groups,
  //       barrier_bits,
  //       element_accumulator_bits,
  //       epilogue_subtile,
  //       num_accumulator_mtxs
  //     );

  //     if (barrier_workspace_size > 0) {
  //       if (workspace == nullptr) {
  //         return Status::kErrorWorkspaceNull;
  //       }

  //       // Only the barrier workspace needs to be cleared for stream-K.
  //       // Barrier workspace follows reduction workspace.
  //       uint8_t* barrier_workspace = reinterpret_cast<uint8_t*>(workspace) +
  //       reduction_workspace_size; return
  //       zero_workspace(static_cast<void*>(barrier_workspace),
  //       barrier_workspace_size, stream, cuda_adapter);
  //     }
  //   #endif // !defined(__CUDACC_RTC__)

  //   return Status::kSuccess;
  // }
  // // clang-format on
};

}  // namespace cutlass::gemm::kernel::detail