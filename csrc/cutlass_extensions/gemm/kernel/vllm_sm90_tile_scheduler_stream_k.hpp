#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/arch/arch.h"

#include "cutlass_extensions/gemm/kernel/vllm_tile_scheduler_params.h"

namespace cutlass::gemm {

struct vLLMStreamKSchedulerWithReset {};

namespace kernel::detail {

template <class TileShape, class ClusterShape>
class vLLMPersistentTileSchedulerSm90StreamKWithReset {
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

  CUTLASS_HOST_DEVICE static constexpr auto can_implement =
      &UnderlyingScheduler::can_implement;

  CUTLASS_HOST_DEVICE static constexpr auto get_work_k_tile_count =
      &UnderlyingScheduler::get_work_k_tile_count;

  CUTLASS_HOST_DEVICE static constexpr auto get_work_k_tile_start =
      &UnderlyingScheduler::get_work_k_tile_start;

  CUTLASS_HOST_DEVICE static constexpr auto assign_work =
      &UnderlyingScheduler::assign_work;

  CUTLASS_HOST_DEVICE static constexpr auto fixup = &UnderlyingScheduler::fixup;

  // Sink scheduler params as a member
  Params scheduler_params;

  CUTLASS_HOST_DEVICE
  vLLMPersistentTileSchedulerSm90StreamKWithReset() {};

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
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
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

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
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
    const uint32_t epilogue_subtile = 1) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    ClusterShape cluster_shape;
    TileShape tile_shape;

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
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
      epilogue_subtile
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