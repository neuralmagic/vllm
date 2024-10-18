#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/arch/arch.h"

#include "cutlass_extensions/gemm/kernel/vllm_tile_scheduler_params.h"

namespace cutlass::gemm {

struct vLLMStreamKSchedulerWithReset {};

namespace kernel::detail {

#define ALIAS_FN(name) using UnderlyingScheduler::name;

template <class TileShape, class ClusterShape>
class vLLMPersistentTileSchedulerSm90StreamKWithReset
    : PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape> {
 private:
  using UnderlyingScheduler =
      PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;

 private:
  using UnderlyingArguments = typename UnderlyingScheduler::Arguments;
  using UnderlyingParams = typename UnderlyingScheduler::Params;

 public:
  using RasterOrder = typename UnderlyingScheduler::RasterOrder;
  using RasterOrderOptions = typename UnderlyingScheduler::RasterOrderOptions;
  // Use a dummy barrier manager to simply get the type used to store the
  // barrier
  using BarrierType = typename NamedBarrierManager<1>::T;

  using Params = vLLMPersistentTileSchedulerSm90StreamKParams;
  using ReductionMode = typename Params::ReductionMode;
  using DecompositionMode = typename Params::DecompositionMode;

  using WorkTileInfo = typename UnderlyingScheduler::WorkTileInfo;
  using Arguments = typename UnderlyingScheduler::Arguments;

  using UnderlyingScheduler::can_implement;
  using UnderlyingScheduler::compute_epilogue;
  using UnderlyingScheduler::fetch_next_work;
  using UnderlyingScheduler::get_grid_shape;
  using UnderlyingScheduler::get_work_k_tile_count;
  using UnderlyingScheduler::get_work_k_tile_start;
  using UnderlyingScheduler::initial_work_tile_info;
  using UnderlyingScheduler::valid_warpgroup_in_work_tile;

  CUTLASS_DEVICE
  static void assign_work(Params const& params, uint64_t linear_idx,
                          WorkTileInfo& work_tile_info) {
    UnderlyingScheduler::assign_work(params, linear_idx, work_tile_info);
  };

  // Performs the reduction across splits for a given output tile.
  template <class FrgTensorC>
  CUTLASS_DEVICE static void fixup(Params const& params,
                                   WorkTileInfo const& work_tile_info,
                                   FrgTensorC& accumulators,
                                   uint32_t num_barriers,
                                   uint32_t barrier_idx) {
    UnderlyingScheduler::fixup(params, work_tile_info, accumulators,
                               num_barriers, barrier_idx);
  }

  CUTLASS_HOST_DEVICE
  static bool requires_separate_reduction(Params const& params) {
    return params.requires_separate_reduction();
  }

  CUTLASS_HOST_DEVICE
  vLLMPersistentTileSchedulerSm90StreamKWithReset() {};

  CUTLASS_HOST_DEVICE
  vLLMPersistentTileSchedulerSm90StreamKWithReset(Params const& params_)
      : UnderlyingScheduler(params_) {}

  // clang-format off
  // Taken directly from: cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp
  template <class ProblemShape>
  static Params
  to_underlying_arguments(
    ProblemShape problem_shape,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo const& hw_info,
    Arguments const& args,
    void* workspace,
    const uint32_t epilogue_subtile = 1) {

    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    dim3 problem_blocks = UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    Params params;
    params.initialize(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(cluster_shape),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.reduction_mode,
      args.decomposition_mode,
      workspace,
      epilogue_subtile
    );
    return params;
  }
  // clang-format on

  // clang-format off
  // Taken directly from: cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp
  template <class ProblemShape, class ElementAccumulator>
  static int
  get_workspace_size(
    Arguments const& args,
    ProblemShape problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t mma_warp_groups,
    const uint32_t epilogue_subtile = 1) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    ClusterShape cluster_shape;
    TileShape tile_shape;

    dim3 problem_blocks = UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    return Params::get_workspace_size(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cluster_shape),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.decomposition_mode,
      mma_warp_groups,
      sizeof_bits<BarrierType>::value,
      sizeof_bits<ElementAccumulator>::value,
      epilogue_subtile
    );
  }
  // clang-format on

  // clang-format off
  // Taken directly from: cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp
  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(
    Arguments const& args,
    void* workspace,
    cudaStream_t stream,
    ProblemShape const& problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t mma_warp_groups,
    const uint32_t epilogue_subtile = 1,
    CudaHostAdapter* cuda_adapter = nullptr) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    ClusterShape cluster_shape;
    TileShape tile_shape;

    dim3 problem_blocks = UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    return Params::initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cluster_shape),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.decomposition_mode,
      mma_warp_groups,
      sizeof_bits<BarrierType>::value,
      sizeof_bits<ElementAccumulator>::value,
      epilogue_subtile,
      1,
      cuda_adapter
    );
  }
  // clang-format on
};

template <class TileShape, class ClusterShape>
struct TileSchedulerSelector<vLLMStreamKSchedulerWithReset, arch::Sm90,
                             TileShape, ClusterShape> {
  using Scheduler =
      vLLMPersistentTileSchedulerSm90StreamKWithReset<TileShape, ClusterShape>;
};

};  // namespace kernel::detail

};  // namespace cutlass::gemm