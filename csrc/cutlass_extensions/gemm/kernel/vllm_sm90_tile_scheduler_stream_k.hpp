#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/arch/arch.h"

#include "cute/config.hpp"

#include "cutlass_extensions/gemm/kernel/vllm_tile_scheduler_params.h"

namespace cutlass::gemm {

struct vLLMStreamKSchedulerWithReset {};

namespace kernel::detail {

// clang-format off
// Users are not supposed to use this class directly.
// This is a CRTP base class for the actual tile schedulers.
template<class Subclass>
class StaticPersistentTileScheduler32bit {
  //
  // Data members
  //

private:
  uint64_t current_work_linear_idx_;
  uint64_t total_grid_size_;

public:
  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      return is_valid_tile;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, false};
    }

    CUTLASS_HOST_DEVICE
    bool
    is_final_split(uint32_t k_tiles_per_output_tile) const {
      return true;
    }

    CUTLASS_HOST_DEVICE
    int32_t
    reduction_subtile_idx() const {
      return -1;
    }
  };

  using Params = PersistentTileSchedulerSm90Params32bit;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
public:
  struct Arguments {
    int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic;
  };

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
    ProblemShapeMNKL problem_shape_mnkl,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    [[maybe_unused]] KernelHardwareInfo const& hw_info,
    Arguments const& arguments,
    [[maybe_unused]] void* workspace=nullptr,
    [[maybe_unused]] const uint32_t epilogue_subtile = 1) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order
    );

    return params;
  }

  CUTLASS_HOST_DEVICE
  static bool
  can_implement(Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  StaticPersistentTileScheduler32bit() { }

  CUTLASS_DEVICE explicit StaticPersistentTileScheduler32bit(Params const& params_) : scheduler_params(params_) {
    // MSVC requires protecting use of CUDA-specific nonstandard syntax,
    // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
    if (params_.raster_order_ == RasterOrder::AlongN) {
      current_work_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
    }
    else {
      current_work_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
    }

    total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);
#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape cluster_shape) {
    return get_current_work();
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(uint64_t linear_idx) const {
    if (linear_idx >= scheduler_params.blocks_per_problem_) {
      return WorkTileInfo::invalid_work_tile();
    }

    // Map worker's linear index into the CTA tiled problem shape to the corresponding MNL indices
    int work_idx_l, remainder;
    scheduler_params.divmod_batch_(work_idx_l, remainder, linear_idx);

    int blk_per_grid_dim = scheduler_params.divmod_cluster_shape_minor_.divide(remainder);

    auto [work_idx_m, work_idx_n] = Subclass::get_work_idx_m_and_n(blk_per_grid_dim,
                                                         scheduler_params.divmod_cluster_shape_major_,
                                                         scheduler_params.divmod_cluster_shape_minor_,
                                                         scheduler_params.divmod_cluster_blk_major_,
                                                         scheduler_params.log_swizzle_size_,
                                                         scheduler_params.raster_order_);

    return {work_idx_m, work_idx_n, static_cast<int32_t>(work_idx_l), true};
  }

  CUTLASS_DEVICE
  void
  advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count);
  }

  // Computes the linear index within a batch given M and N tile offsets within the batch.
  // This essentially inverts the mapping performed in get_work_idx_m_and_n
  static CUTLASS_DEVICE
  uint64_t
  get_linear_idx_from_m_and_n(
    int32_t tile_m,
    int32_t tile_n,
    FastDivmodU64Pow2 const& divmod_cluster_shape_major,
    FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
    FastDivmod const& divmod_cluster_blk_major,
    int32_t log_swizzle_size,
    RasterOrder raster_order) {

    auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();

    uint64_t minor_work_idx, major_work_idx, cluster_minor_offset;
    if (raster_order == RasterOrder::AlongN) {
      minor_work_idx = static_cast<uint64_t>(tile_m);
      major_work_idx = static_cast<uint64_t>(tile_n);
      cluster_minor_offset = cta_m_in_cluster;
    }
    else {
      major_work_idx = static_cast<uint64_t>(tile_m);
      minor_work_idx = static_cast<uint64_t>(tile_n);
      cluster_minor_offset = cta_n_in_cluster;
    }

    uint64_t cluster_idx_minor, cluster_idx_major, cluster_major_offset;
    cluster_idx_minor = divmod_cluster_shape_minor.divide(minor_work_idx - cluster_minor_offset);
    divmod_cluster_shape_major(cluster_idx_major, cluster_major_offset, major_work_idx);

    uint64_t cluster_idx_minor_div_swizzle = cluster_idx_minor >> log_swizzle_size;
    uint64_t offset = cluster_idx_minor & ((1 << log_swizzle_size) - 1);

    uint64_t extra = cluster_idx_minor_div_swizzle * divmod_cluster_blk_major.divisor + cluster_idx_major;

    uint64_t cluster_id = (extra << log_swizzle_size) | offset;
    return (cluster_id * divmod_cluster_shape_major.divisor + cluster_major_offset) * divmod_cluster_shape_minor.divisor + cluster_minor_offset;
  }

  // Given the inputs, computes the total number of output blocks over which this problem will compute. 
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, BlockShape cta_shape, ClusterShape cluster_shape) {
    auto cta_m = cute::size(cute::ceil_div(cute::shape<0>(problem_shape_mnkl), cute::shape<0>(cta_shape)));
    auto cta_n = cute::size(cute::ceil_div(cute::shape<1>(problem_shape_mnkl), cute::shape<1>(cta_shape)));

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(problem_shape_mnkl),
      to_gemm_coord(cluster_shape),
      cta_m, cta_n
    );
  }
  // Kernel helper function to get next work ID
  template <class WorkIdPipeline, class WorkIdPipelineState>
  CUTLASS_DEVICE
  auto
  fetch_next_work(
    WorkTileInfo work_tile_info,
    WorkIdPipeline& work_id_pipeline,
    WorkIdPipelineState work_id_pipe_consumer_state) {
      WorkTileInfo new_work_tile_info;
      advance_to_next_work();
      new_work_tile_info = get_current_work();

    // Return true to indicate that the WorkID pipeline state should be advanced
    return cute::make_tuple(new_work_tile_info, true);
  }

  CUTLASS_DEVICE
  static auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info) {
    // Get every cta coord in three dimensions of the cluster
    auto [cta_m_in_cluster, cta_n_in_cluster, cta_l_in_cluster] = cute::block_id_in_cluster();
    return make_coord(
      work_tile_info.M_idx + static_cast<int32_t>(cta_m_in_cluster),
      work_tile_info.N_idx + static_cast<int32_t>(cta_n_in_cluster),
      _,
      work_tile_info.L_idx + static_cast<int32_t>(cta_l_in_cluster)
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    ProblemShapeMNKL problem_shape_mnk,
    BlockShape cta_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    Arguments arguments,
    bool truncate_by_problem_size=true) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape_mnk, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape);

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    Params const& params,
    ProblemShapeMNKL problem_shape_mnk,
    BlockShape cta_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape_mnk, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape);

    Arguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.log_swizzle_size_;
    }
    args.raster_order = params.raster_order_ == RasterOrder::AlongN ? RasterOrderOptions::AlongN : RasterOrderOptions::AlongM;

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      args.max_swizzle_size,
      args.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  // Convert CTA-level work tile info to cluster-level tile coord
  CUTLASS_DEVICE
  cute::Coord<int,int,int,int>
  tile_info_to_coord_mnkl(WorkTileInfo work_tile_info) const {
    // TileScheduler works at CTA-level, kernel works at cluster-level
    int m_coord = idx2crd(work_tile_info.M_idx / scheduler_params.cluster_shape_m_,
                          scheduler_params.problem_tiles_m_);
    int n_coord = idx2crd(work_tile_info.N_idx / scheduler_params.cluster_shape_n_,
                          scheduler_params.problem_tiles_n_);
    int l_coord = idx2crd(work_tile_info.L_idx,
                          scheduler_params.problem_tiles_l_);
    return make_coord(m_coord, n_coord, _, l_coord);
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the basic tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&) {
    return true;
  }

  // Performs the reduction across splits for a given output tile. Since this scheduler does
  // not split output tiles, no reduction is needed.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(Params const&, WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) {}

  // Performs the reduction across splits for a given output tile. No fixup is required for
  // work units returned by this scheduler.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  fixup(WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) const { }

  // Returns whether the current WorkTileInfo passed in should continue to be used. Since
  // this scheduler only schedules work in units of single, full output tiles, the WorkTileInfo
  // passed in should not be used after having been processed.
  CUTLASS_DEVICE
  static bool
  continue_current_work(WorkTileInfo&) {
    return false;
  }

  template <class ProblemShape, class TileShape>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape problem_shape, TileShape tile_shape) {
    // All work units returned by this scheduler cover the entire K iteration
    // space of the output tile assigned to the work unit.
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const&) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  CUTLASS_DEVICE
  static bool
  need_separate_reduction(Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  bool
  is_work_tile_for_reduction(WorkTileInfo const& work_tile_info, Params const& params) {
    return false;
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  separate_reduction(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
  }

  // Shares the accumulator set with peers in the global workspace
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  share(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
  }

  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return true;
  }

  CUTLASS_DEVICE
  static bool
  requires_separate_reduction(Params const& params) {
    return false;
  }
public:
  // Sink scheduler params as a member
  Params scheduler_params;
};

// Persistent Thread Block (TB) scheduler
class PersistentTileSchedulerSm9032bit:
public StaticPersistentTileScheduler32bit<PersistentTileSchedulerSm9032bit> {

  using BaseScheduler = StaticPersistentTileScheduler32bit<PersistentTileSchedulerSm9032bit>;
public:
  using StaticPersistentTileScheduler32bit::StaticPersistentTileScheduler32bit;
  using Params = PersistentTileSchedulerSm90Params32bit;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  using Arguments = BaseScheduler::Arguments;

  // get work_idx_m, work_idx_n from blk_per_grid_dim while applying swizzle
  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmod const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order) {

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();
    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = cta_m_in_cluster;
    }
    else {
      cluster_minor_offset = cta_n_in_cluster;
    }

    int cluster_idx_minor, cluster_idx_major;

    int cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor +
                                               cluster_minor_offset);
    auto major_work_idx = static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor +
                                               cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx};
    }
    else {
      return {major_work_idx, minor_work_idx};
    }

  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static int
  get_workspace_size(Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape, KernelHardwareInfo const&,
    uint32_t, const uint32_t = 1) {
    return Status::kSuccess;
  }

};


// Persistent Thread Block (TB) scheduler leveraging stream-K decomposition
template <
  class TileShape,
  class ClusterShape
>
class PersistentTileSchedulerSm90StreamK32bit {
  //
  // Data members
  //

private:
  using UnderlyingScheduler = PersistentTileSchedulerSm9032bit;

private:
  using UnderlyingArguments = typename UnderlyingScheduler::Arguments;
  using UnderlyingParams = typename UnderlyingScheduler::Params;

  uint64_t current_work_linear_idx_ = 0;

public:

  using RasterOrder = UnderlyingScheduler::RasterOrder;
  using RasterOrderOptions = UnderlyingScheduler::RasterOrderOptions;
  // Use a dummy barrier manager to simply get the type used to store the barrier
  using BarrierType = typename NamedBarrierManager<1>::T;

  using Params = PersistentTileSchedulerSm90StreamKParams32bit<false>;
  using ReductionMode = Params::ReductionMode;
  using DecompositionMode = Params::DecompositionMode;

  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t K_idx = 0;
    int32_t L_idx = 0;

    // Number of k tiles to compute for this unit of work. For stream-K, this
    // can indicate the number of K tiles across multiple output tiles.
    uint32_t k_tile_count = 0;

    // Number of k tiles remaining for the work unit as a whole
    uint32_t k_tile_remaining = 0;

    // Whether this unit of work is the final split for the given tile
    bool is_separate_reduction = false;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      // A work tile that computes no K tiles is invalid unless it is a separate-reduction work tile
      // (which only performs reduction and epilogue)
      return k_tile_count > 0 || is_separate_reduction;
    }

    CUTLASS_HOST_DEVICE
    bool
    is_reduction_unit() const {
      return is_separate_reduction;
    }

    CUTLASS_HOST_DEVICE
    int32_t
    reduction_subtile_idx() const {
      // For separate reduction units, the K_idx of the work tile is unused.
      // Therefore, we override it to contain the subtile of that the reduction
      // unit operates on.
      return is_reduction_unit() ? K_idx : -1;
    }

    CUTLASS_HOST_DEVICE
    void
    setup_separate_reduction(int32_t epilogue_subtile_idx) {
      // Set the epilogue subtile in the K_idx, since this is otherwise unused
      // by separate reduction units.
      K_idx = epilogue_subtile_idx;

      is_separate_reduction = true;
      k_tile_count = 0;
      // Clean up remaining k tiles
      k_tile_remaining = 0;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, -1, 0};
    }

    CUTLASS_HOST_DEVICE
    bool
    is_final_split(uint32_t k_tiles_per_output_tile) const {
      return (K_idx + k_tile_count) == k_tiles_per_output_tile;
    }
  };

  struct Arguments {

    Arguments() = default;
    Arguments(Arguments const&) = default;
    Arguments(Arguments&&) = default;

    CUTLASS_HOST_DEVICE
    Arguments&
    operator=(Arguments const& args) {
      splits = args.splits;
      max_swizzle_size = args.max_swizzle_size;
      raster_order = args.raster_order;
      reduction_mode = args.reduction_mode;
      decomposition_mode = args.decomposition_mode;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    Arguments&
    operator=(Arguments&& args) noexcept {
      splits = args.splits;
      max_swizzle_size = args.max_swizzle_size;
      raster_order = args.raster_order;
      reduction_mode = args.reduction_mode;
      decomposition_mode = args.decomposition_mode;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    Arguments(int splits_) : splits(splits_) {}

    CUTLASS_HOST_DEVICE
    Arguments(int splits_, int max_swizzle_size_, RasterOrderOptions raster_order_, DecompositionMode decomposition_mode_) :
      splits(splits_),
      max_swizzle_size(max_swizzle_size_),
      raster_order(raster_order_),
      decomposition_mode(decomposition_mode_) {}

    // The splitting factor to be used in a split-K decomposition of the problem.
    // If this is set to a value greater than 1, stream-K decomposition logic
    // is bypassed in favor of a split-K decomposition.
    int splits = 1;
    int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic;
    ReductionMode reduction_mode = ReductionMode::Deterministic;
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic;
  };

  // Sink scheduler params as a member
  Params scheduler_params;

  //
  // Methods
  //

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

  static bool
  can_implement(Arguments const& args) {
    // Split count > 1 is only valid for heuristic and split-K decomposition modes
    return (args.splits == 1 ||
            args.decomposition_mode == DecompositionMode::Heuristic ||
            args.decomposition_mode == DecompositionMode::SplitK);
  }

  CUTLASS_HOST_DEVICE
  PersistentTileSchedulerSm90StreamK32bit() { };

  CUTLASS_HOST_DEVICE
  PersistentTileSchedulerSm90StreamK32bit(Params const& params_) : scheduler_params(params_) {
    if (params_.raster_order_ == RasterOrder::AlongN) {
      current_work_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
    }
    else {
      current_work_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
    }
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_, scheduler_params);
  }

  CUTLASS_DEVICE
  static WorkTileInfo
  get_current_work_for_linear_idx(uint64_t linear_idx, Params const& params) {
    // The maximum number of work units is units_per_problem_ * splits_.
    // The multiplication by splits_ is used for handling split-K, in which
    // units_per_problem_ is equal to the total number of output tiles. To account
    // for the fact that we have splits_ peers per output tile, we multiply this
    // value by splits_. For stream-K, this multiplication ends up being a no-op
    // because splits_ is set to 1 for stream-K.
    if(linear_idx >= (params.units_per_problem_ * params.divmod_splits_.divisor + params.separate_reduction_units_)) {
      // Invalid work. Return an empty result.
      return WorkTileInfo::invalid_work_tile();
    }

    WorkTileInfo work_tile_info;
    assign_work(params, linear_idx, work_tile_info);
    return work_tile_info;
  }

  // Returns whether the current work_tile_info passed in should continue to be used. This
  // occurs only in the stream-K decomposition with stream-K work units, which encompass
  // work over multiple output tiles. If the current work_tile_info should continue to be
  // used, it is updated to advance to the next output tile it should cover.
  CUTLASS_DEVICE
  bool
  continue_current_work(WorkTileInfo& work_tile_info) const {
    return continue_current_work_for_linear_idx(
      current_work_linear_idx_, work_tile_info, scheduler_params);
  }

  CUTLASS_DEVICE
  static bool
  continue_current_work_for_linear_idx(
    uint64_t linear_idx,
    WorkTileInfo& work_tile_info,
    Params const& params) {

    work_tile_info.k_tile_remaining -= work_tile_info.k_tile_count;

    if (work_tile_info.k_tile_remaining == 0) {
      return false;
    }
    assign_work(params, linear_idx, work_tile_info);
    return work_tile_info.is_valid();
  }

  CUTLASS_DEVICE
  void
  advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z) * uint64_t(advance_count);
  }

  // Given the inputs, computes the total number of output blocks this problem will compute over
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShape problem_shape_mnkl, TileShape cta_shape, ClusterShape cluster_shape) {
    return UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape);
  }

  // Given the cluster shape, computes the physical grid we should launch.
  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    ProblemShape problem_shape,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    Arguments arguments) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order
    );
  }

  // Returns whether fixup is needed for `work_tile_info`.
  CUTLASS_HOST_DEVICE
  static bool
  requires_fixup(Params const& params, WorkTileInfo const& work_tile_info) {
    // Fixup is not needed for invalid or data-parallel tiles
    return work_tile_info.is_valid() && work_tile_info.k_tile_count != params.divmod_tiles_per_output_tile_.divisor;
  }

  CUTLASS_HOST_DEVICE
  static bool
  requires_separate_reduction(Params const& params) {
    return params.requires_separate_reduction();
  }

  // When the work tile is not special for reduction, it's valid. Otherwise need to skip
  // global loading that producer warpgroup do, also math computation that consumer warpgroup do.
  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return !work_tile_info.is_reduction_unit();
  }

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
    BarrierType* lock_workspace = reinterpret_cast<BarrierType*>(
      reinterpret_cast<uint8_t*>(params.reduction_workspace_) + reduction_workspace_size);

    if (work_tile_info.is_reduction_unit()) {
      plus<AccumulatorArrayT> add_fragments;
      auto peer_offset = size(accumulators) * num_barriers * BarrierManager::ThreadCount;

      // Wait until the peers collaborating on this output tile have all written
      // their accumulators to workspace.
      uint32_t num_peers = last_peer_id - first_peer_id + 1;
      BarrierManager::wait_eq(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, num_peers);

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
      if (params.reduction_mode_ == ReductionMode::Deterministic) {
        // Wait until the preceding split added its accumulators
        BarrierManager::wait_eq(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, work_tile_info.K_idx);
      }
      else {
        // Wait unitl the first split has stored its accumulators
        BarrierManager::wait_lt(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, 1);
      }

      // The block computing the final split for the tile adds previously-reduced partials
      // to its accumulators and computes the epilogue.
      BlockStripedReduceT::load_add(*accumulator_array, reduction_workspace_array, barrier_group_thread_idx);
    }
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the case of stream-K, this should only occur if the work is marked as the final split.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const& work_tile_info, Params const& params) {
    // `is_final_split` will be set to `true` for the following scenarios, all of which must compute the epilogue:
    //  1. The tile is computed in data-parallel mode
    //  2. The tile is computed in split-/stream-K mode and this work unit represents the final split of the tile
    //  3. The tile is computed in split-/stream-K mode and separate reduction is used, and this is a separate reduction unit
    return work_tile_info.is_valid() &&
            (work_tile_info.is_final_split(params.divmod_tiles_per_output_tile_.divisor) &&
             !params.requires_separate_reduction()) || work_tile_info.is_separate_reduction;
  }

  // Returns the linearized index of the output tile corresponding to the tile with offset [L, M, K]
  CUTLASS_DEVICE
  static int
  output_tile_index(Params const& params, WorkTileInfo const& work_tile_info) {
    uint64_t linear_idx_in_batch = UnderlyingScheduler::get_linear_idx_from_m_and_n(
      work_tile_info.M_idx, work_tile_info.N_idx,
      params.divmod_cluster_shape_major_,
      params.divmod_cluster_shape_minor_,
      params.divmod_cluster_blk_major_,
      params.log_swizzle_size_,
      params.raster_order_
    );

    uint64_t tiles_mn = params.divmod_batch_.divisor;
    return tiles_mn * work_tile_info.L_idx + linear_idx_in_batch;
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t
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
      epilogue_subtile,
      1,
      cuda_adapter
    );
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape, TileShape) {
    return work_tile_info.k_tile_count;
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const& work_tile_info) {
    return work_tile_info.K_idx;
  }

  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  auto
  fetch_next_work(WorkTileInfo work_tile_info) {
    if (continue_current_work(work_tile_info)) {
      return work_tile_info;
    }

    advance_to_next_work();
    return get_current_work();
  }

  // Returns the initial work tile info that will be computed over
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape) {
    return get_current_work();
  }

private:
  // Sets the current stream-K work to compute within work_tile_info. If new_unit is true, work_tile_info
  // is populated as a new unit of work. Otherwise, state existing in work_tile_info (e.g., remaining
  // iterations) is used to find the next tile in the current work unit.
  CUTLASS_DEVICE
  static void
  assign_work(
    Params const& params,
    uint64_t linear_idx,
    WorkTileInfo& work_tile_info) {

    auto [cta_m_in_cluster_, cta_n_in_cluster_, _] = cute::block_id_in_cluster();
    uint64_t cta_m_in_cluster = static_cast<uint64_t>(cta_m_in_cluster_);
    uint64_t cta_n_in_cluster = static_cast<uint64_t>(cta_n_in_cluster_);
    uint64_t output_tile_id = linear_idx;
    if (linear_idx >= params.units_per_problem_ * params.divmod_splits_.divisor) {
      // Separate-reduction work
      auto cluster_size = params.get_cluster_size();
      // Divide up the linearized separate reduction units into clusters
      auto cluster_linear_reduction_unit_idx = params.div_cluster_size((linear_idx - params.units_per_problem_));
      int32_t cluster_tile_idx, epi_subtile_idx;
      params.divmod_epilogue_subtile_(cluster_tile_idx, epi_subtile_idx, cluster_linear_reduction_unit_idx);
      // Bring the linearized tile ID back into the space of tiles, rather than clusters
      output_tile_id = cluster_tile_idx * cluster_size;

      work_tile_info.setup_separate_reduction(epi_subtile_idx);
    }
    else if (linear_idx >= params.sk_units_ && params.divmod_splits_.divisor == 1) {
      // Data-parallel work
      output_tile_id = linear_idx - params.sk_units_ + params.sk_tiles_;
      work_tile_info.K_idx = 0;
      work_tile_info.k_tile_count = params.divmod_tiles_per_output_tile_.divisor;
      work_tile_info.k_tile_remaining = params.divmod_tiles_per_output_tile_.divisor;
    }
    else {
      // In the CUTLASS 2.x implementation of stream K, stream-K work is assigned to each stream-K
      // threadblock individually. For the most part, the set of K iterations corresponding to stream-K
      // work was divided amongst stream-K threadblocks, and a threadblock determined which tile
      // it would compute a (potentially-partial) output tile for based on the space of k iterations
      // assigned to it. This often results in stream-K threadblocks processing tiles with different
      // offsets in the K dimension from one another. This can reduce locality, but is lmitied to the
      // (generally few) waves of threadblocks assigned to compute stream-K work.
      //
      // With the introduction of threadblock clusters, there is additional benefit to maintaining
      // locality in the K dimension: shared portions of operands can be multicasted to threadblocks
      // within a cluster. Thus, we would like to ensure that the assignment of stream-K work to
      // threadblocks respects the ability to perform multicasting.
      //
      // To do so, we divide up the linearized stream-K units into clusters and share the same K
      // offsets for work within clusters.

      auto cluster_linear_work_idx = params.div_cluster_size(linear_idx);

      int32_t group_idx;
      params.divmod_sk_groups_(cluster_linear_work_idx, group_idx, cluster_linear_work_idx);

      // Determine whether we are in a "big group" that will process an additional
      // stream-K cluster tile.
      auto sk_cluster_tiles = params.div_cluster_size(params.sk_tiles_);
      auto sk_cluster_tiles_in_group = params.divmod_sk_groups_.divide(sk_cluster_tiles);
      if (group_idx < params.big_groups_) {
        ++sk_cluster_tiles_in_group;
      }

      // Determine whether we are in a "big unit" within the group, that will process
      // an additional K chunk in the group.
      auto sk_tiles_in_group = sk_cluster_tiles_in_group * params.get_cluster_size();
      auto k_tiles_in_group = sk_tiles_in_group * params.divmod_tiles_per_output_tile_.divisor;
      auto k_tiles_per_unit_in_group = params.divmod_sk_units_per_group_.divide(k_tiles_in_group);
      auto big_units_in_group = params.div_cluster_size(
        k_tiles_in_group - (k_tiles_per_unit_in_group * params.divmod_sk_units_per_group_.divisor));

      int32_t split;
      params.divmod_clusters_mnl_(split, cluster_linear_work_idx, cluster_linear_work_idx);

      bool is_split_k = params.divmod_splits_.divisor > 1;
      auto big_unit_cmp_lhs = is_split_k ? split : cluster_linear_work_idx;
      auto big_unit_cmp_rhs = is_split_k ? params.big_units_ : big_units_in_group;
      auto linear_idx_mult = is_split_k ? params.divmod_tiles_per_output_tile_.divisor : k_tiles_per_unit_in_group;
      auto k_tiles_per_split = is_split_k ? params.divmod_k_tiles_per_sk_unit_.divisor : k_tiles_per_unit_in_group;

      // Determine the starting k iteration computed by this stream-K work unit
      uint32_t unit_iter_start = (linear_idx_mult * cluster_linear_work_idx) +
                                 (k_tiles_per_split * split);

      // Adjust the starting position and number of k iterations for "big units," which
      // compute one extra iteration. If there are any big units, they will be the first
      // in the linearized ID space.
      auto k_tiles_in_my_split = k_tiles_per_split;
      if (big_unit_cmp_lhs < big_unit_cmp_rhs) {
        // Since the "big units" are the first units in the linearized ID space, each
        // of the units preceding this big unit computed one extra iteration. Thus,
        // we must offset our start iteration by the number of units that precede
        // the current unit in the linearized ID space.
        unit_iter_start += big_unit_cmp_lhs;
        ++k_tiles_in_my_split;
      }
      else {
        // Increment by one for each of the big clusters (since all big units precede this unit)
        unit_iter_start += big_unit_cmp_rhs;
      }

      if (!is_split_k) {
        // Adjust the unit starting position and number of tiles to avoid
        // computing splits of size less than min_iters_per_sk_unit_
        int unused, start_tile_k_tile;
        params.divmod_tiles_per_output_tile_(unused, start_tile_k_tile, unit_iter_start);
        if (start_tile_k_tile < Params::min_iters_per_sk_unit_) {
          // Starting K tile is in range [0, Params::min_iters_per_sk_unit_), which means that another
          // stream-K unit will be computing a split with fewer than Params::min_iters_per_sk_unit_ K tiles.
          // Adjust our work to take over these K tiles.
          unit_iter_start -= start_tile_k_tile;
          k_tiles_in_my_split += start_tile_k_tile;
        }
        else if (start_tile_k_tile > (params.divmod_tiles_per_output_tile_.divisor - Params::min_iters_per_sk_unit_)) {
          // Starting K tile is within the final Params::min_iters_per_sk_unit_ K tiles of some output tile,
          // which means that this unit will compute a split with fewer than Params::min_iters_per_sk_unit_ K tiles.
          // Adjust our work to shed these K tiles to a neighboring stream-K unit that will compute more consecutive K tiles.
          auto adjustment_tiles = (params.divmod_tiles_per_output_tile_.divisor - start_tile_k_tile);
          unit_iter_start += adjustment_tiles;
          k_tiles_in_my_split -= adjustment_tiles;
        }
        else if (params.ktile_start_alignment_count == 2 && start_tile_k_tile % 2 != 0) {
          // ktile for each SM start from even number
          // If start from odd number ktile within the output tile
          //    now start at the ktile one before my initial ktile start (take one ktile from prev sm)
          // if end on odd number ktile within the output tile
          //    now end at ktile that one before my ktile end (give one ktile to next sm)
          unit_iter_start -= 1;
          k_tiles_in_my_split += 1;
        }
      }

      if (work_tile_info.k_tile_count == 0) {
        // This is a new unit

        if (!is_split_k) {
          //
          // Adjust the unit ending position and number of tiles to avoid
          // computing splits of size less than min_iters_per_sk_unit_
          //

          // Begin by assuming that no adjustment is needed
          auto initial_unit_iter_end = unit_iter_start + k_tiles_in_my_split;

          int unused, end_tile_k_tile;
          params.divmod_tiles_per_output_tile_(unused, end_tile_k_tile, initial_unit_iter_end);

          if (end_tile_k_tile < Params::min_iters_per_sk_unit_) {
            // Ending K tile is within the first Params::min_iters_per_sk_unit_ K tiles of some output tile,
            // which means that this unit will compute a split with fewer than Params::min_iters_per_sk_unit_ K tiles.
            // Adjust our work to shed these K tiles to a neighboring stream-K unit that will compute more consecutive K tiles.
            k_tiles_in_my_split -= end_tile_k_tile;
          }
          else if (end_tile_k_tile > (params.divmod_tiles_per_output_tile_.divisor - Params::min_iters_per_sk_unit_)) {
            // Ending K tile is within the final Params::min_iters_per_sk_unit_ K tiles of some output tile,
            // which means that some other unit will compute a split with fewer than Params::min_iters_per_sk_unit_ K tiles.
            // Adjust our work to take on these K tiles.
            k_tiles_in_my_split += (params.divmod_tiles_per_output_tile_.divisor - end_tile_k_tile);
          }
          else if (params.ktile_start_alignment_count == 2 && end_tile_k_tile % 2 != 0) {
            // ktile for each SM start from even number
            // If start from odd number ktile within the output tile
            //    now start at the ktile one before my initial ktile start (take one ktile from prev sm)
            // If end on odd number ktile within the output tile,
            //    now end at ktile that one before my ktile end (give one ktile to next sm)
            k_tiles_in_my_split -= 1;
          }
        }

        work_tile_info.k_tile_remaining = k_tiles_in_my_split;
      }

      uint32_t unit_iter_end = unit_iter_start + work_tile_info.k_tile_remaining - 1;

      // Find the output tile corresponding to the final k tile covered by this
      // work unit. Stream-K work units will work backwards in terms of the tiles they
      // are responsible computing. This is beneficial because the final (partial)
      // tile computed by a stream-K block is typically the beginning of the output
      // tile, while the beginning (partial) tile is typically the ending of another
      // output tile. Since ending portions of an output tile must reduce across
      // other work units computing portions of that output tile, it is preferable
      // for them to be computed later, so as to reduce the likelihood of blocking
      // on other work.

      auto output_tile_id_in_group = params.divmod_tiles_per_output_tile_.divide(unit_iter_end);
      uint32_t output_tile_iter_start = output_tile_id_in_group * params.divmod_tiles_per_output_tile_.divisor;
      uint32_t output_tile_iter_end = output_tile_iter_start + params.divmod_tiles_per_output_tile_.divisor;

      // Convert the output tile from the linearized space within each group to the
      // overall linearized space.
      output_tile_id = (output_tile_id_in_group * params.divmod_sk_groups_.divisor) + group_idx;

      // Bring the linearized tile ID back into the space of tiles, rather than clusters
      output_tile_id *= params.get_cluster_size();

      // The final linearized tile ID is in units of the cluster dimension over which we rasterize.
      if (params.raster_order_ == RasterOrder::AlongN) {
        output_tile_id += cta_n_in_cluster * params.divmod_cluster_shape_minor_.divisor;
      }
      else {
        output_tile_id += cta_m_in_cluster * params.divmod_cluster_shape_minor_.divisor;
      }

      // The unit's starting k iteration in the current tile is either the starting
      // iteration for the tile as a whole, or the starting k iteration for the unit
      // as a whole (if the latter is greater than the former).
      uint32_t tile_iter_start = max(output_tile_iter_start, unit_iter_start);

      // Similarly, the unit's ending k iteration (exclusive) is either the end of
      // the current tile it is assigned, or the ending iteration of the unit as a whole
      // (if the latter is less than the former).
      uint32_t tile_iter_end = min(output_tile_iter_end, unit_iter_end + 1);

      // Set the k offset to be the starting k tile for this output tile
      work_tile_info.K_idx = static_cast<int32_t>(tile_iter_start - output_tile_iter_start);
      work_tile_info.k_tile_count = tile_iter_end - tile_iter_start;
    }

    int32_t work_idx_l, remainder;
    params.divmod_batch_(work_idx_l, remainder, output_tile_id);

    uint64_t cta_per_grid_dim = params.divmod_cluster_shape_minor_.divide(remainder);

    auto [work_idx_m, work_idx_n] = UnderlyingScheduler::get_work_idx_m_and_n(
                                          cta_per_grid_dim,
                                          params.divmod_cluster_shape_major_,
                                          params.divmod_cluster_shape_minor_,
                                          params.divmod_cluster_blk_major_,
                                          params.log_swizzle_size_,
                                          params.raster_order_
                                        );

    // Set the M, N, and L block offsets
    work_tile_info.M_idx = work_idx_m;
    work_tile_info.N_idx = work_idx_n;
    work_tile_info.L_idx = static_cast<int32_t>(work_idx_l);
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
};
// clang-format on

template <class TileShape, class ClusterShape>
class vLLMPersistentTileSchedulerSm90StreamKWithReset
    : PersistentTileSchedulerSm90StreamK32bit<TileShape, ClusterShape> {
 private:
  using UnderlyingScheduler =
      PersistentTileSchedulerSm90StreamK32bit<TileShape, ClusterShape>;

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

    // if (work_tile_info.is_reduction_unit()) {
    //   plus<AccumulatorArrayT> add_fragments;
    //   auto peer_offset = size(accumulators) * num_barriers * BarrierManager::ThreadCount;

    //   // Wait until the peers collaborating on this output tile have all written
    //   // their accumulators to workspace.
    //   uint32_t num_peers = last_peer_id - first_peer_id + 1;
    //   BarrierManager::wait_eq_reset(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, num_peers);

    //   // Load the first peer's data
    //   BlockStripedReduceT::load(*accumulator_array, reduction_workspace_array, barrier_group_thread_idx);

    //   for (int i = 1; i < num_peers; ++i) {
    //     // Load peer fragment
    //     AccumulatorArrayT addend_fragment;
    //     auto peer_reduction_workspace = reinterpret_cast<AccumulatorArrayT*>(group_reduction_workspace + (i * peer_offset));

    //     BlockStripedReduceT::load(addend_fragment, peer_reduction_workspace, barrier_group_thread_idx);

    //     // Add peer fragment
    //     *accumulator_array = add_fragments(*accumulator_array, addend_fragment);
    //   }
    // }
    // else 
    if (!compute_epilogue(work_tile_info, params)) {
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