#pragma once

#include <vector>
#include <hip/hip_runtime.h>
#include "quick_reduce_impl.cuh"

#define HIP_CHECK(err)                                                     \
  do {                                                                     \
    hipError_t err_ = (err);                                               \
    if (err_ != hipSuccess) {                                              \
      std::printf("HIP error %d at %s:%d. %s\n", err_, __FILE__, __LINE__, \
                  hipGetErrorString(err_));                                \
      throw std::runtime_error("HIP error");                               \
    }                                                                      \
  } while (0)

namespace quickreduce {
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

static constexpr int kOneShotAllreduceMaxElemsWorldSize2 = 8192 * 12;
static constexpr int kOneShotAllreduceMaxElemsWorldSize4 = 8192 * 8;
static constexpr int kOneShotAllreduceMaxElemsWorldSize8 = 8192 * 4;
static constexpr long kOneShotAllreduceMaxSize =
    std::max(kOneShotAllreduceMaxElemsWorldSize2 * 2,
             std::max(kOneShotAllreduceMaxElemsWorldSize4 * 4,
                      kOneShotAllreduceMaxElemsWorldSize8 * 8)) *
    sizeof(half);

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_one_shot__ static void
allreduce_prototype_oneshot(T const* A, T* B, int N, int rank,
                            uint8_t** dbuffer_list, long data_offset,
                            int flag_color) {
  AllReduceKernel::run(A, B, N, rank, dbuffer_list, data_offset, flag_color);
}

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_twoshot(T const* A, T* B, int N, int num_blocks, int rank,
                            uint8_t** dbuffer_list, long data_offset,
                            int flag_color) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  while (block < num_blocks) {
    AllReduceKernel::run(A, B, N, block, num_blocks, rank, dbuffer_list,
                         data_offset, flag_color);
    block += grid;
  }
}

#define ONESHOT_DISPATCH()                                                  \
  if (world_size == 2) {                                                    \
    using AllReduceKernel = AllReduceOneshot<T, 2>;                         \
    hipLaunchKernelGGL((allreduce_prototype_oneshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockOneShot), 0, stream, A, B, N, \
                       rank, dbuffer_list, data_offset, flag_color);        \
  } else if (world_size == 4) {                                             \
    using AllReduceKernel = AllReduceOneshot<T, 4>;                         \
    hipLaunchKernelGGL((allreduce_prototype_oneshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockOneShot), 0, stream, A, B, N, \
                       rank, dbuffer_list, data_offset, flag_color);        \
  } else if (world_size == 8) {                                             \
    using AllReduceKernel = AllReduceOneshot<T, 8>;                         \
    hipLaunchKernelGGL((allreduce_prototype_oneshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockOneShot), 0, stream, A, B, N, \
                       rank, dbuffer_list, data_offset, flag_color);        \
  }

#define TWOSHOT_DISPATCH(__codec)                                           \
  if (world_size == 2) {                                                    \
    using LineCodec = __codec<T, 2>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec>;                 \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color);                                         \
  } else if (world_size == 4) {                                             \
    using LineCodec = __codec<T, 4>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec>;                 \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color);                                         \
  } else if (world_size == 8) {                                             \
    using LineCodec = __codec<T, 8>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec>;                 \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color);                                         \
  }

struct DeviceComms {
  // Workgroup scope = Tile = (256 threads x 16B x 8 atoms)
  static long constexpr kTileSize = 256 * 16 * 8;

  // Max problem size is 8GB (in bytes)
  static long constexpr kMaxProblemSize = 8589934592;
  static long constexpr kMaxTiles = kMaxProblemSize / kTileSize;

  // Max TP-8
  static int constexpr kMaxWorldSize = 8;

  bool initialized = false;
  int flag_color = 1;
  int world_size;
  int rank;

  uint8_t* dbuffer;
  uint8_t** dbuffer_list;
  hipIpcMemHandle_t buffer_ipc_handle;
  std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;
  std::vector<uint8_t*> buffer_list;
  long data_offset;

  DeviceComms() : initialized(false), world_size(1), rank(0) {}
  ~DeviceComms() { destroy(); }

  void init(int world_size, int rank) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;

    // Allocate buffer size for worst case: Twoshot FP16 2-stage buffer.
    long flags_buffer_size = 2 * world_size * kMaxTiles * sizeof(int);
    static constexpr long data_buffer_size =
        std::max(2 * kMaxProblemSize, kOneShotAllreduceMaxSize);
    long total_buffer_size = flags_buffer_size + data_buffer_size;
    data_offset = flags_buffer_size;
    HIP_CHECK(hipExtMallocWithFlags((void**)&dbuffer, total_buffer_size,
                                    hipDeviceMallocUncached));

    // Clear the flags buffer.
    HIP_CHECK(hipMemset(dbuffer, 0, flags_buffer_size));

    // Device-side list of IPC buffers.
    buffer_list.resize(world_size);
    HIP_CHECK(hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*)));

    // Create IPC handles for rank's communication buffer.
    all_buffer_ipc_handles.resize(world_size);
    HIP_CHECK(hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer));

    initialized = true;
  }
  int get_world_size() { return world_size; }
  int get_rank() { return rank; }
  bool status() { return initialized; }
  hipIpcMemHandle_t const get_handle() { return buffer_ipc_handle; }

  void destroy() {
    if (initialized) {
      for (int i = 0; i < world_size; i++) {
        if (i != rank) {
          HIP_CHECK(hipIpcCloseMemHandle(dbuffer_list[i]));
        }
      }

      HIP_CHECK(hipFree(dbuffer));
      HIP_CHECK(hipFree(dbuffer_list));

      initialized = false;
    }
  }

  void open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles) {
    assert(ipc_handles.size() == all_buffer_ipc_handles.size());
    for (int i = 0; i < world_size; i++) {
      all_buffer_ipc_handles[i] = ipc_handles[i];
    }

    // Open device memory access to the IPC communication buffers.
    // Note: For our own rank, we do not need to open a handle.
    for (int i = 0; i < world_size; i++) {
      if (i != rank) {
        HIP_CHECK(hipIpcOpenMemHandle((void**)&buffer_list[i],
                                      all_buffer_ipc_handles[i],
                                      hipIpcMemLazyEnablePeerAccess));
      } else {
        buffer_list[i] = dbuffer;
      }
    }

    HIP_CHECK(hipMemcpy(dbuffer_list, buffer_list.data(),
                        world_size * sizeof(uint8_t*), hipMemcpyHostToDevice));
  }

  template <typename T>
  void allreduce(T const* A, T* B, int N, bool quantized, hipStream_t stream) {
    if (world_size != 2 && world_size != 4 && world_size != 8) {
      throw std::runtime_error("All Reduce not supported for world_size = " +
                               std::to_string(world_size));
    }

    // Configuration.
    long msg_size = N * sizeof(T);
    bool use_one_shot_allreduce =
        (world_size == 2 and N <= kOneShotAllreduceMaxElemsWorldSize2) or
        (world_size == 4 and N <= kOneShotAllreduceMaxElemsWorldSize4) or
        (world_size == 8 and N <= kOneShotAllreduceMaxElemsWorldSize8);
    if (use_one_shot_allreduce) {
      // Each thread processes blocks out of 4 elements
      unsigned long num_blocks = divceil(N, (4 * kThreadsOneShot));
      unsigned long grid = min(kMaxNumBlocks, num_blocks);
      ONESHOT_DISPATCH()
    } else {
      unsigned long num_blocks = divceil(msg_size, kTileSize);
      unsigned long grid = min(kMaxNumBlocks, num_blocks);

      if (quantized) {
        TWOSHOT_DISPATCH(CodecQ4Symm)
      } else {
        TWOSHOT_DISPATCH(CodecFP16)
      }
    }
    HIP_CHECK(cudaGetLastError());
    // Rotate the flag color.
    flag_color++;
  }
};

}  // namespace quickreduce