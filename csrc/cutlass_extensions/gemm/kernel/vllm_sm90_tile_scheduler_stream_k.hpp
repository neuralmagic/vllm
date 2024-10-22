#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/arch/arch.h"

#include "cute/config.hpp"

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
  // using Arguments = typename UnderlyingScheduler::Arguments;

  struct Arguments : UnderlyingArguments {
    void* barrier_workspace;
  };

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

  CUTLASS_HOST_DEVICE
  static bool requires_separate_reduction(Params const& params) {
    return params.requires_separate_reduction();
  }

  CUTLASS_HOST_DEVICE
  vLLMPersistentTileSchedulerSm90StreamKWithReset() {}

  CUTLASS_HOST_DEVICE
  vLLMPersistentTileSchedulerSm90StreamKWithReset(Params const& params_)
      : UnderlyingScheduler(params_) {
    // if (params_.reduction_mode_ != ReductionMode::Deterministic) {
    //   CUTE_RUNTIME_ASSERT(
    //       "vLLMPersistentTileSchedulerSm90StreamKWithReset "
    //       "currently only supports deterministic reduction.");
    // }
  }

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
    params.barrier_workspace_ = args.barrier_workspace;
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
  static int
  get_barrier_workspace_size(
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

    return Params::get_barrier_workspace_size(
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

  // clang-format off

  // Performs the reduction across splits for a given output tile.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
    static constexpr uint32_t Offset = static_cast<int>(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
    static constexpr uint32_t MaxNumNamedBarriers = 2;
    using BarrierManager = NamedBarrierManager<NumThreadsPerWarpGroup, Offset, MaxNumNamedBarriers>;
    return fixup_helper<FrgTensorC, BarrierManager>(
      params, work_tile_info, accumulators, num_barriers, barrier_idx);
  }

  // Helper for performing the reduction across splits for a given output tile.
  template <class FrgTensorC, class BarrierManager>
  CUTLASS_DEVICE
  static void
  fixup_helper(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx,
    uint32_t num_accumulator_mtxs = 1) {

    using ElementAccumulator = typename FrgTensorC::value_type;

    if (!requires_fixup(params, work_tile_info)) {
      return;
    }
    auto tile_idx = output_tile_index(params, work_tile_info);

    // Index of the lock on which to wait
    auto lock_idx = (tile_idx * num_barriers) + barrier_idx;

    auto reduction_tile_idx = tile_idx;
    auto [first_peer_id, my_peer_id, last_peer_id] = tile_peer_range(params, tile_idx, static_cast<uint32_t>(work_tile_info.K_idx));
    auto reduction_peer_offset = 0;
    if (params.requires_separate_reduction()) {
      // If separate reduction is to be performed, each stream-K unit writes its partials
      // to a separate portion of the workspace. There are as many of these portions as there
      // are peers for a given output tile, so we multiply the tile index by the maximum peer count.
      reduction_tile_idx *= Params::max_peers_per_tile(params.sk_units_, params.sk_tiles_);
      reduction_peer_offset = my_peer_id * cute::size<0>(TileShape{}) * cute::size<1>(TileShape{});
    }

    // Reductions use BlockStripedReduce with a width of BarrierManager::ThreadCount under the hood.
    // Thus, the start of the reduction space is the same across all threads in a warp group.
    int reduction_offset =
      (cute::size<0>(TileShape{}) * cute::size<1>(TileShape{}) * reduction_tile_idx * num_accumulator_mtxs) +
      reduction_peer_offset +
      (size(accumulators) * barrier_idx * BarrierManager::ThreadCount);

    ElementAccumulator* group_reduction_workspace = reinterpret_cast<ElementAccumulator*>(params.reduction_workspace_) + reduction_offset;

    using AccumulatorArrayT = Array<typename FrgTensorC::value_type, size(FrgTensorC{})>;
    using BlockStripedReduceT = BlockStripedReduce<BarrierManager::ThreadCount, AccumulatorArrayT>;

    AccumulatorArrayT* reduction_workspace_array = reinterpret_cast<AccumulatorArrayT*>(group_reduction_workspace);
    AccumulatorArrayT* accumulator_array = reinterpret_cast<AccumulatorArrayT*>(accumulators.data());

    int barrier_group_thread_idx = threadIdx.x % BarrierManager::ThreadCount;

    // The number of tiles for which reduction is required is either:
    //   (a) the total number of output tiles (in the case of split-K)
    //   (b) the number of stream-K tiles (potentially multiplied by peer count if using separate reduction)
    // To calculate the total number of output tiles in the split-K case, we
    // note that, in the split-K case, the units_per_problem_ member of Params will be
    // the total number of output tiles.
    uint32_t reduction_tiles = 0;
    if (params.divmod_splits_.divisor > 1) {
      reduction_tiles = params.units_per_problem_;
    }
    else if (params.requires_separate_reduction()) {
      reduction_tiles = params.sk_tiles_ * Params::max_peers_per_tile(params.sk_units_, params.sk_tiles_);
    }
    else {
      reduction_tiles = params.sk_tiles_;
    }

    auto reduction_workspace_size = Params::get_reduction_workspace_size(
      reduction_tiles, to_gemm_coord(TileShape{}), sizeof_bits<ElementAccumulator>::value, num_accumulator_mtxs);
    BarrierType* lock_workspace = reinterpret_cast<BarrierType*>(params.barrier_workspace_);

    if (work_tile_info.is_reduction_unit()) {
      plus<AccumulatorArrayT> add_fragments;
      auto peer_offset = size(accumulators) * num_barriers * BarrierManager::ThreadCount;

      // Wait until the peers collaborating on this output tile have all written
      // their accumulators to workspace.
      uint32_t num_peers = last_peer_id - first_peer_id + 1;
      BarrierManager::wait_eq_reset(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, num_peers);

      // Load the first peer's data
      BlockStripedReduceT::load(*accumulator_array, reduction_workspace_array, barrier_group_thread_idx);

      for (int i = 1; i < num_peers; ++i) {
        // Load peer fragment
        AccumulatorArrayT addend_fragment;
        auto peer_reduction_workspace = reinterpret_cast<AccumulatorArrayT*>(group_reduction_workspace + (i * peer_offset));

        BlockStripedReduceT::load(addend_fragment, peer_reduction_workspace, barrier_group_thread_idx);

        // Add peer fragment
        *accumulator_array = add_fragments(*accumulator_array, addend_fragment);
      }
    }
    else if (!compute_epilogue(work_tile_info, params)) {
      if (params.requires_separate_reduction() || work_tile_info.K_idx == 0) {
        // The first peer initializes the workspace partials in the non-separate-reduction case,
        // and all peers write to their own location in workspace when using separate reduction
        BlockStripedReduceT::store(reduction_workspace_array, *accumulator_array, barrier_group_thread_idx);
      }
      else {
        // Wait until the preceding split added its accumulators
        BarrierManager::wait_eq(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, work_tile_info.K_idx);

        // Perform reduction in workspace
        BlockStripedReduceT::reduce(reduction_workspace_array, *accumulator_array, barrier_group_thread_idx);
      }

      // If separate reduction is being performed, each participating stream-K unit increments the barrier
      // by only 1. Otherwise, increment by the K tile count that this unit has processed.
      int32_t increment = params.requires_separate_reduction() ? 1 : work_tile_info.k_tile_count;

      // Signal our arrival
      BarrierManager::arrive_inc(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, increment);
    }
    else {
      // We only support ReductionMode::Deterministic
      //if (params.reduction_mode_ == ReductionMode::Deterministic) {
      // Wait until the preceding split added its accumulators
      if (work_tile_info.is_final_split(params.divmod_tiles_per_output_tile_.divisor)) {
        BarrierManager::wait_eq_reset(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, work_tile_info.K_idx);
      } else {
        BarrierManager::wait_eq(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, work_tile_info.K_idx);
      }
      //}

      // The block computing the final split for the tile adds previously-reduced partials
      // to its accumulators and computes the epilogue.
      BlockStripedReduceT::load_add(*accumulator_array, reduction_workspace_array, barrier_group_thread_idx);
    }
  }

  // Returns the starting and ending peer ID of this tile
  CUTLASS_HOST_DEVICE
  static auto
  tile_peer_range(Params const& params, uint32_t tile_idx, uint32_t cur_k_tile) {
    auto tile_idx_in_cluster_path = params.div_cluster_size(tile_idx);
    auto start_k_tile = params.divmod_tiles_per_output_tile_.divisor * tile_idx_in_cluster_path;
    auto end_k_tile = start_k_tile + params.divmod_tiles_per_output_tile_.divisor - 1;
    auto big_unit_k_tiles = params.big_units_ * (params.divmod_k_tiles_per_sk_unit_.divisor + 1);

    auto adjust_unit = [&](uint32_t k_tile, uint32_t unit_idx, uint32_t k_tiles_per_unit) {
      auto unit_k_start = unit_idx * k_tiles_per_unit;
      auto unit_k_end = unit_k_start + k_tiles_per_unit;
      if (k_tile - start_k_tile < Params::min_iters_per_sk_unit_ &&
          unit_k_end - start_k_tile < Params::min_iters_per_sk_unit_) {
        // k_tile is within the first min_iters_per_sk_unit_ K tiles of this output tile,
        // and the stream-K unit computes fewer than min_iters_per_sk_unit_ K tiles for this
        // output tile. This work will thus be subsumed by the next stream-K unit.
        ++unit_idx;
      }

      if (end_k_tile + 1 - k_tile < Params::min_iters_per_sk_unit_ &&
          end_k_tile + 1 - unit_k_start < Params::min_iters_per_sk_unit_) {
        // k_tile is within the last min_iters_per_sk_unit_ K tiles of this output tile,
        // and the stream-K unit computes fewer than min_iters_per_sk_unit_ K tiles for this
        // output tile. This work will thus be subsumed by the previous stream-K unit.
        --unit_idx;
      }

      return unit_idx;
    };

    // Lambda to find the ID of the stream-K unit that computes this K tile
    auto find_unit = [&](uint32_t k_tile) {
      if (k_tile < big_unit_k_tiles) {
        // The tile is within the "big unit range"
        auto unit_idx = params.divmod_k_tiles_per_sk_big_unit_.divide(k_tile);
        return static_cast<uint64_t>(adjust_unit(k_tile, unit_idx, params.divmod_k_tiles_per_sk_big_unit_.divisor));
      }
      else {
        // The tile is after the "big unit range." Account for this by finding the "normal unit"
        // that it belongs to, and then offsetting by the number of big units
        auto unit_idx = params.divmod_k_tiles_per_sk_unit_.divide(k_tile - big_unit_k_tiles) + params.big_units_;
        return static_cast<uint64_t>(adjust_unit(k_tile, unit_idx, params.divmod_k_tiles_per_sk_unit_.divisor));
      }
    };

    return cute::make_tuple(find_unit(start_k_tile), find_unit(cur_k_tile), find_unit(end_k_tile));
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