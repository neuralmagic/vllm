// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#if defined(O_DIRECT)
constexpr int kODirectFlag = O_DIRECT;
#else
constexpr int kODirectFlag = 0;
#endif

extern "C" {

namespace {

// Returns 0 on success, or the std::error_code's POSIX-compatible value on
// failure, mirroring the errno convention used by the syscalls below.
inline int ensure_parent_dirs(const std::string& path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) {
    return 0;
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  return ec ? ec.value() : 0;
}

// Core single-block store: src/size are raw pointer + byte count. Returns 0
// on success, or the errno of the failing step on failure -- captured
// before any subsequent cleanup call can overwrite it. On failure, the temp
// file is removed.
inline int _store_block(const char* tmp_path, const char* dest_path,
                        const char* src, size_t size, bool use_o_direct) {
  if (access(dest_path, F_OK) == 0) {
    return 0;  // Already present.
  }

  if (const int err = ensure_parent_dirs(dest_path); err != 0) {
    return err;
  }

  const int o_direct_flag = use_o_direct ? kODirectFlag : 0;
  const int fd = open(
      tmp_path, O_CREAT | O_EXCL | O_WRONLY | O_TRUNC | o_direct_flag, 0644);
  if (fd < 0) {
    return errno;
  }

  const ssize_t written = write(fd, src, size);
  if (written < 0 || static_cast<size_t>(written) != size) {
    const int err = written < 0 ? errno : EIO;
    close(fd);  // Best-effort cleanup; the real error is already captured.
    unlink(tmp_path);
    return err;
  }

  if (close(fd) != 0) {
    const int err = errno;
    unlink(tmp_path);
    return err;
  }

  if (rename(tmp_path, dest_path) != 0) {
    const int err = errno;
    unlink(tmp_path);
    return err;
  }

  return 0;
}

// Core single-block load: dst/size are raw pointer + byte count. Returns 0
// on success, or the errno of the failing step on failure. Removes the source
// file ONLY on a provable short read (the read completed but returned fewer
// bytes than requested): stores are atomic, so a too-short file is genuine
// corruption. Open failures and read errors (bytes_read < 0) are
// transient/ambiguous and leave the file untouched; a close failure after a
// full read is harmless and does not fail the load.
inline int _load_block(const char* source_path, char* dst, size_t size,
                       bool use_o_direct) {
  const int o_direct_flag = use_o_direct ? kODirectFlag : 0;
  const int fd = open(source_path, O_RDONLY | o_direct_flag, 0);
  if (fd < 0) {
    return errno;
  }

  const ssize_t bytes_read = read(fd, dst, size);
  if (bytes_read < 0) {
    // Transient read error: leave the file untouched.
    const int err = errno;
    close(fd);
    return err;
  }
  if (static_cast<size_t>(bytes_read) < size) {
    // Provable short read: the block is genuinely corrupt, so remove it.
    close(fd);
    unlink(source_path);
    return EIO;
  }

  // A close error after a successful full read is harmless: the data is
  // already in the destination buffer, so the load succeeds.
  close(fd);
  return 0;
}

inline void _batch_lookup(const std::vector<const char*>& paths,
                          std::vector<int>& exists_flags) {
  for (size_t i = 0; i < paths.size(); i++) {
    exists_flags[i] = (access(paths[i], F_OK) == 0) ? 1 : 0;
  }
}

// Helper: extract a list[str] of length n into a vector<const char*>.
// Returns false and sets a Python exception on error.
inline bool extract_str_list(PyObject* list, Py_ssize_t n,
                             std::vector<const char*>& out) {
  for (Py_ssize_t i = 0; i < n; i++) {
    out[i] = PyUnicode_AsUTF8AndSize(PyList_GetItem(list, i), nullptr);
    if (out[i] == nullptr) {
      return false;
    }
  }
  return true;
}

// Helper: extract a Py_buffer per element of a list[bytes-like] of length n.
// On success, `out` holds n acquired buffers (caller must PyBuffer_Release
// each). On failure, any buffers already acquired are released before
// returning false, and a Python exception is set.
inline bool extract_buffer_list(PyObject* list, Py_ssize_t n, int flags,
                                std::vector<Py_buffer>& out) {
  for (Py_ssize_t i = 0; i < n; i++) {
    if (PyObject_GetBuffer(PyList_GetItem(list, i), &out[i], flags) != 0) {
      for (Py_ssize_t j = 0; j < i; j++) {
        PyBuffer_Release(&out[j]);
      }
      return false;
    }
  }
  return true;
}

inline void release_buffer_list(std::vector<Py_buffer>& buffers) {
  for (auto& buf : buffers) {
    PyBuffer_Release(&buf);
  }
}

// ---------------------------------------------------------------------------
// Work-stealing pool: a per-tier Pool holds a load WorkQueue and a store
// WorkQueue (each its own mutex) plus a ResultQueue (its own mutex). Worker
// threads (real Python threading.Thread objects) call wait_and_run() which
// releases the GIL once and drains items until both queues are empty,
// pushing one Result per item. Python drains ResultQueue asynchronously via
// pop_all_results(), independent of worker activity.
// ---------------------------------------------------------------------------

struct WorkItem {
  int64_t job_id;
  int32_t index;  // position within the job's original paths/offsets list.
  std::string path;
  size_t offset;
};

struct Result {
  int64_t job_id;
  int32_t index;
  int err;  // errno of the failing step, or 0 on success.
  double transfer_time;
};

struct WorkQueue {
  std::mutex mutex;
  std::deque<WorkItem> items;
};

struct ResultQueue {
  std::mutex mutex;
  std::deque<Result> items;
};

struct Pool {
  WorkQueue load_q;
  WorkQueue store_q;
  ResultQueue results;
  Py_buffer buffer;  // primary_kv_view, pinned for the pool's lifetime.
  size_t block_size = 0;
  bool use_o_direct = true;
  // Set by destroy_pool(); get_pool() rejects further use. Lets the pinned
  // buffer be released deterministically at shutdown() while the capsule's
  // own destructor (which may run later, at an unpredictable GC/refcount
  // time) still safely deletes the Pool exactly once and skips re-releasing
  // the buffer.
  bool destroyed = false;
  // Idle signaling only (never guards load_q/store_q/results themselves):
  // lets a worker with no work block in wait_and_run() instead of spinning,
  // while still waking instantly on new work or request_stop().
  std::mutex signal_mutex;
  std::condition_variable signal_cv;
  bool stop = false;
};

constexpr const char* kPoolCapsuleName = "vllm.fs_io_C.Pool";

// Enqueues *n* items (paths[i], offsets[i]) tagged with job_id and index i
// into *q*. Holds q.mutex for the whole batch so the job's items appear
// contiguously and atomically from a consumer's point of view.
inline void enqueue_items(WorkQueue& q, int64_t job_id,
                          std::vector<std::string>& paths,
                          const std::vector<int64_t>& offsets) {
  std::lock_guard<std::mutex> lock(q.mutex);
  for (size_t i = 0; i < paths.size(); i++) {
    q.items.push_back(WorkItem{job_id, static_cast<int32_t>(i),
                               std::move(paths[i]),
                               static_cast<size_t>(offsets[i])});
  }
}

inline void _pop_all_job_items_locked(WorkQueue& q, int64_t job_id,
                                      std::vector<WorkItem>& out) {
  while (!q.items.empty() && q.items.front().job_id == job_id) {
    out.push_back(std::move(q.items.front()));
    q.items.pop_front();
  }
}

// Pops one item, preferring *primary* then falling back to *secondary*.
// Sets *is_load* to whichever queue the item actually came from (which may
// differ from the caller's own priority when falling back).
inline bool pop_items(WorkQueue& primary, WorkQueue& secondary,
                      bool primary_is_load, std::vector<WorkItem>& out,
                      bool& is_load) {
  {
    std::lock_guard<std::mutex> lock(primary.mutex);
    if (!primary.items.empty()) {
      is_load = primary_is_load;
      bool const is_store = !is_load;
      if (is_store) {
        _pop_all_job_items_locked(primary, primary.items.front().job_id, out);
      } else {
        out.push_back(std::move(primary.items.front()));
        primary.items.pop_front();
      }
      return true;
    }
  }
  std::lock_guard<std::mutex> lock(secondary.mutex);
  if (!secondary.items.empty()) {
    is_load = !primary_is_load;
    bool const is_store = !is_load;
    if (is_store) {
      _pop_all_job_items_locked(secondary, secondary.items.front().job_id, out);
    } else {
      out.push_back(std::move(secondary.items.front()));
      secondary.items.pop_front();
    }
    return true;
  }
  return false;
}

inline bool queue_group_nonempty(WorkQueue& primary, WorkQueue& secondary) {
  {
    std::lock_guard<std::mutex> lock(primary.mutex);
    if (!primary.items.empty()) return true;
  }
  std::lock_guard<std::mutex> lock(secondary.mutex);
  return !secondary.items.empty();
}

// A per-thread random suffix, generated once and reused for every temp file
// this worker thread ever writes -- mirrors io.py's _get_tmp_suffix(). job_id
// + index would not be unique across pools/processes (e.g. two independent
// FileSystemTierManager instances, or two ranks sharing a network
// filesystem, can both legitimately run job_id=1/index=0 at the same time),
// so uniqueness has to come from randomness instead.
inline const std::string& tmp_suffix() {
  thread_local std::string suffix;
  if (suffix.empty()) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, (uint64_t{1} << 63) - 1);
    suffix = "_" + std::to_string(dist(gen)) + ".tmp";
  }
  return suffix;
}

// Performs the raw I/O for one item directly against the pool's pinned
// buffer. No Python API calls: safe to run with the GIL released.
inline int do_io(Pool& pool, const WorkItem& item, bool is_load) {
  const size_t buf_len = static_cast<size_t>(pool.buffer.len);
  // Reject out-of-bounds offsets before touching the buffer at all -- an
  // unchecked offset+block_size here would be a silent heap overflow
  // (read or write) instead of a clean per-item failure.
  if (item.offset > buf_len || pool.block_size > buf_len - item.offset) {
    return ERANGE;
  }
  char* block_ptr = static_cast<char*>(pool.buffer.buf) + item.offset;
  if (is_load) {
    return _load_block(item.path.c_str(), block_ptr, pool.block_size,
                       pool.use_o_direct);
  }
  const std::string tmp_path = item.path + tmp_suffix();
  return _store_block(tmp_path.c_str(), item.path.c_str(), block_ptr,
                      pool.block_size, pool.use_o_direct);
}

inline Pool* get_pool(PyObject* capsule) {
  if (!PyCapsule_CheckExact(capsule)) {
    PyErr_SetString(PyExc_TypeError, "expected a fs_io pool capsule");
    return nullptr;
  }
  void* ptr = PyCapsule_GetPointer(capsule, kPoolCapsuleName);
  if (ptr == nullptr) {
    PyErr_SetString(PyExc_ValueError, "invalid fs_io pool handle");
    return nullptr;
  }
  Pool* pool = static_cast<Pool*>(ptr);
  if (pool->destroyed) {
    PyErr_SetString(PyExc_ValueError, "fs_io pool handle already destroyed");
    return nullptr;
  }
  return pool;
}

void destroy_pool_capsule(PyObject* capsule) {
  // Do not go through get_pool(): a pool destroyed via destroy_pool() must
  // still be freed here exactly once when the capsule itself is deallocated.
  Pool* pool =
      static_cast<Pool*>(PyCapsule_GetPointer(capsule, kPoolCapsuleName));
  if (pool != nullptr) {
    if (!pool->destroyed) {
      PyBuffer_Release(&pool->buffer);
    }
    delete pool;
  }
}

}  // namespace

/// @brief Create a work-stealing I/O pool bound to one shared KV buffer.
/// @param primary_kv_view writable bytes-like object; pinned for the pool's
///                         lifetime (one PyObject_GetBuffer call total).
/// @param block_size       size in bytes of each block.
/// @param use_o_direct     whether to open files with O_DIRECT (default True).
/// @return PyCapsule opaque pool handle.
static PyObject* create_pool(PyObject* /*self*/, PyObject* args) {
  PyObject* buffer_obj = nullptr;
  long long block_size = 0;
  int use_o_direct = 1;
  if (!PyArg_ParseTuple(args, "OL|p", &buffer_obj, &block_size,
                        &use_o_direct)) {
    return nullptr;
  }

  auto* pool = new Pool();
  pool->block_size = static_cast<size_t>(block_size);
  pool->use_o_direct = use_o_direct != 0;

  if (PyObject_GetBuffer(buffer_obj, &pool->buffer, PyBUF_WRITABLE) != 0) {
    delete pool;
    return nullptr;
  }

  PyObject* capsule =
      PyCapsule_New(pool, kPoolCapsuleName, destroy_pool_capsule);
  if (capsule == nullptr) {
    PyBuffer_Release(&pool->buffer);
    delete pool;
    return nullptr;
  }
  return capsule;
}

/// @brief Explicitly release a pool's pinned buffer.
/// @note Safe to call even though the underlying Pool object is not freed
///       here: the capsule's destructor frees it later (deterministically,
///       once the last Python reference to the capsule is dropped) and
///       skips releasing the buffer again, since `destroyed` is now true.
///       Calling destroy_pool() a second time raises, same as any other
///       operation on an already-destroyed handle.
static PyObject* destroy_pool(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  Pool* pool = get_pool(capsule);
  if (pool == nullptr) {
    return nullptr;
  }
  PyBuffer_Release(&pool->buffer);
  pool->destroyed = true;
  Py_RETURN_NONE;
}

/// @brief Enqueue every (path, offset) pair of a job into the load or store
///        WorkQueue in one locked batch.
static PyObject* push_items(PyObject* args, bool is_load) {
  PyObject* capsule = nullptr;
  long long job_id = 0;
  PyObject* paths_obj = nullptr;
  PyObject* offsets_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OLO!O!", &capsule, &job_id, &PyList_Type,
                        &paths_obj, &PyList_Type, &offsets_obj)) {
    return nullptr;
  }
  Pool* pool = get_pool(capsule);
  if (pool == nullptr) return nullptr;

  const Py_ssize_t n = PyList_Size(paths_obj);
  if (PyList_Size(offsets_obj) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "paths and offsets must have the same length");
    return nullptr;
  }

  std::vector<std::string> paths(n);
  for (Py_ssize_t i = 0; i < n; i++) {
    const char* s =
        PyUnicode_AsUTF8AndSize(PyList_GetItem(paths_obj, i), nullptr);
    if (s == nullptr) return nullptr;
    paths[i] = s;  // own copy; the Python list may be freed after we return.
  }

  std::vector<int64_t> offsets(n);
  for (Py_ssize_t i = 0; i < n; i++) {
    const long long off = PyLong_AsLongLong(PyList_GetItem(offsets_obj, i));
    if (off == -1 && PyErr_Occurred()) return nullptr;
    offsets[i] = off;
  }

  enqueue_items(is_load ? pool->load_q : pool->store_q, job_id, paths, offsets);
  pool->signal_cv.notify_all();
  Py_RETURN_NONE;
}

static PyObject* push_load(PyObject* /*self*/, PyObject* args) {
  return push_items(args, /*is_load=*/true);
}

static PyObject* push_store(PyObject* /*self*/, PyObject* args) {
  return push_items(args, /*is_load=*/false);
}

/// @brief Check whether there is any claimable work for a worker with the
///        given priority (own queue, or the other queue as fallback).
/// @note Takes the same mutexes as push/pop; a bare unsynchronized read of
///       the deques while another thread mutates them would be a data race.
static PyObject* queue_nonempty(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  int load_priority = 0;
  if (!PyArg_ParseTuple(args, "Op", &capsule, &load_priority)) {
    return nullptr;
  }
  Pool* pool = get_pool(capsule);
  if (pool == nullptr) return nullptr;

  WorkQueue& primary = load_priority ? pool->load_q : pool->store_q;
  WorkQueue& secondary = load_priority ? pool->store_q : pool->load_q;
  const bool nonempty = queue_group_nonempty(primary, secondary);
  if (nonempty) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

// How long a worker with no work stays parked inside wait_and_run() before
// returning to Python, so a steady stream of small jobs doesn't force a
// GIL acquisition (Python call return + re-invoke) between every one.
constexpr auto kIdleTimeout = std::chrono::seconds(30);

/// @brief Drain and execute work items for a worker with the given
///        priority, blocking (without spinning) whenever both queues run
///        dry, for up to kIdleTimeout since the last time work was found.
/// @note Releases the GIL once for the entire call -- no Python API call
///       happens between free_gil and acquire_gil. Each item's Result is
///       pushed to ResultQueue as soon as that item finishes, so the
///       Python scheduler thread can observe partial progress at any time
///       via pop_all_results(), independent of when this call returns.
///       request_stop() wakes an idling worker immediately.
static PyObject* wait_and_run(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  int load_priority = 0;
  if (!PyArg_ParseTuple(args, "Op", &capsule, &load_priority)) {
    return nullptr;
  }
  Pool* pool = get_pool(capsule);
  if (pool == nullptr) return nullptr;

  WorkQueue& primary = load_priority ? pool->load_q : pool->store_q;
  WorkQueue& secondary = load_priority ? pool->store_q : pool->load_q;

  Py_BEGIN_ALLOW_THREADS const auto deadline =
      std::chrono::steady_clock::now() + kIdleTimeout;
  while (true) {
    bool is_load = false;
    std::vector<WorkItem> items;
    while (pop_items(primary, secondary, load_priority != 0, items, is_load)) {
      for (auto const& item : items) {
        const auto start = std::chrono::steady_clock::now();
        const int err = do_io(*pool, item, is_load);
        const double transfer_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count();
        Result result{item.job_id, item.index, err, transfer_time};
        {
          std::lock_guard<std::mutex> lock(pool->results.mutex);
          pool->results.items.push_back(result);
        }
      }
      items.clear();
    }

    std::unique_lock<std::mutex> signal_lock(pool->signal_mutex);
    if (pool->stop) break;
    const bool woke_for_work_or_stop = pool->signal_cv.wait_until(
        signal_lock, deadline,
        [&] { return pool->stop || queue_group_nonempty(primary, secondary); });
    if (!woke_for_work_or_stop) break;  // idle timeout: nothing to do.
    if (pool->stop) break;
    // Otherwise new work arrived within the window -- loop back and drain.
  }
  Py_END_ALLOW_THREADS

      Py_RETURN_NONE;
}

/// @brief Drain every currently-available Result into a Python list.
/// @note Locks ResultQueue's mutex for the swap only; marshaling to Python
///       tuples happens afterwards, unlocked, with the GIL held throughout.
static PyObject* pop_all_results(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  Pool* pool = get_pool(capsule);
  if (pool == nullptr) return nullptr;

  std::deque<Result> drained;
  {
    std::lock_guard<std::mutex> lock(pool->results.mutex);
    drained.swap(pool->results.items);
  }

  PyObject* out = PyList_New(static_cast<Py_ssize_t>(drained.size()));
  if (out == nullptr) return nullptr;
  Py_ssize_t idx = 0;
  for (const Result& r : drained) {
    PyObject* tup = PyTuple_New(4);
    if (tup == nullptr) {
      Py_DECREF(out);
      return nullptr;
    }
    PyTuple_SetItem(tup, 0, PyLong_FromLongLong(r.job_id));
    PyTuple_SetItem(tup, 1, PyLong_FromLong(r.index));
    PyTuple_SetItem(tup, 2, PyLong_FromLong(r.err));
    PyTuple_SetItem(tup, 3, PyFloat_FromDouble(r.transfer_time));
    PyList_SetItem(out, idx++, tup);
  }
  return out;
}

/// @brief Clear both work queues (load and store). Used by shutdown(): any
///        worker blocked inside wait_and_run() simply finds nothing left to
///        pop and returns; no separate C-side stop flag is needed.
static PyObject* clear_work_queue(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  Pool* pool = get_pool(capsule);
  if (pool == nullptr) return nullptr;

  {
    std::lock_guard<std::mutex> lock(pool->load_q.mutex);
    pool->load_q.items.clear();
  }
  {
    std::lock_guard<std::mutex> lock(pool->store_q.mutex);
    pool->store_q.items.clear();
  }
  Py_RETURN_NONE;
}

/// @brief Wake every worker idling inside wait_and_run() so it returns
///        immediately instead of waiting out its idle timeout. Used by
///        shutdown(); does not itself clear the work queues.
static PyObject* request_stop(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  Pool* pool = get_pool(capsule);
  if (pool == nullptr) return nullptr;

  {
    std::lock_guard<std::mutex> lock(pool->signal_mutex);
    pool->stop = true;
  }
  pool->signal_cv.notify_all();
  Py_RETURN_NONE;
}

/// @brief Check file existence for a batch of paths.
/// @param paths list[str] – absolute paths to check.
/// @return list[bool] – True if the corresponding path exists, False otherwise.
/// @note Releases the GIL for the entire batch. File existence via access(2).
static PyObject* batch_lookup(PyObject* /*self*/, PyObject* args) {
  PyObject* path_list;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &path_list)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(path_list);
  std::vector<const char*> paths(n);
  for (Py_ssize_t i = 0; i < n; i++) {
    paths[i] = PyUnicode_AsUTF8AndSize(PyList_GetItem(path_list, i), nullptr);
    if (paths[i] == nullptr) {
      return nullptr;
    }
  }

  std::vector<int> exists_flags(n);
  {
    Py_BEGIN_ALLOW_THREADS _batch_lookup(paths, exists_flags);
    Py_END_ALLOW_THREADS
  }

  PyObject* result = PyList_New(n);
  if (result == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    PyList_SetItem(result, i, PyBool_FromLong(exists_flags[i]));
  }
  return result;
}

/// @brief Store a batch of blocks, each from its own buffer, to disk.
/// @param tmp_paths    list[str] – one temp path per block.
/// @param dest_paths   list[str] – one destination path per block.
/// @param buffers      list[bytes-like] – one source buffer per block.
/// @param use_o_direct bool – whether to open files with O_DIRECT
///                     (default True). Ignored where O_DIRECT is unsupported
///                     by the platform.
/// @note Releases the GIL for the entire batch. Raises on first error.
static PyObject* batch_store_block(PyObject* /*self*/, PyObject* args) {
  PyObject* tmp_paths_obj = nullptr;
  PyObject* dest_paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;
  int use_o_direct = 1;

  if (!PyArg_ParseTuple(args, "O!O!O!|p", &PyList_Type, &tmp_paths_obj,
                        &PyList_Type, &dest_paths_obj, &PyList_Type,
                        &buffers_obj, &use_o_direct)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(tmp_paths_obj);
  if (PyList_Size(dest_paths_obj) != n || PyList_Size(buffers_obj) != n) {
    PyErr_SetString(
        PyExc_ValueError,
        "tmp_paths, dest_paths and buffers must have the same length");
    return nullptr;
  }

  std::vector<const char*> tmp_paths(n);
  std::vector<const char*> dest_paths(n);

  if (!extract_str_list(tmp_paths_obj, n, tmp_paths)) return nullptr;
  if (!extract_str_list(dest_paths_obj, n, dest_paths)) return nullptr;

  std::vector<Py_buffer> buffers(n);
  if (!extract_buffer_list(buffers_obj, n, PyBUF_SIMPLE, buffers)) {
    return nullptr;
  }

  Py_ssize_t failed_index = -1;
  int failure_errno = 0;

  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      const char* buf = static_cast<const char*>(buffers[i].buf);
      const int err =
          _store_block(tmp_paths[i], dest_paths[i], buf,
                       static_cast<size_t>(buffers[i].len), use_o_direct);
      if (err != 0) {
        failed_index = i;
        failure_errno = err;
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  if (failed_index >= 0) {
    // PyErr_SetFromErrnoWithFilename() reads the errno to format exception.
    errno = failure_errno;
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError,
                                          dest_paths[failed_index]);
  }

  Py_RETURN_NONE;
}

/// @brief Load a batch of blocks from disk, each into its own buffer.
/// @param source_paths list[str] – one source path per block.
/// @param buffers      list[writable bytes-like] – one destination buffer
///                     per block.
/// @param use_o_direct bool – whether to open files with O_DIRECT
///                     (default True). Ignored where O_DIRECT is unsupported
///                     by the platform.
/// @note Releases the GIL for the entire batch. Raises on first error.
static PyObject* batch_load_block(PyObject* /*self*/, PyObject* args) {
  PyObject* source_paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;
  int use_o_direct = 1;

  if (!PyArg_ParseTuple(args, "O!O!|p", &PyList_Type, &source_paths_obj,
                        &PyList_Type, &buffers_obj, &use_o_direct)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(source_paths_obj);
  if (PyList_Size(buffers_obj) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "source_paths and buffers must have the same length");
    return nullptr;
  }

  std::vector<const char*> source_paths(n);
  if (!extract_str_list(source_paths_obj, n, source_paths)) return nullptr;

  std::vector<Py_buffer> buffers(n);
  if (!extract_buffer_list(buffers_obj, n, PyBUF_WRITABLE, buffers)) {
    return nullptr;
  }

  Py_ssize_t failed_index = -1;
  int failure_errno = 0;

  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      char* buf = static_cast<char*>(buffers[i].buf);
      const int err =
          _load_block(source_paths[i], buf, static_cast<size_t>(buffers[i].len),
                      use_o_direct);
      if (err != 0) {
        failed_index = i;
        failure_errno = err;
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  if (failed_index >= 0) {
    // PyErr_SetFromErrnoWithFilename() reads the errno to format exception.
    errno = failure_errno;
    PyErr_SetFromErrnoWithFilename(PyExc_OSError, source_paths[failed_index]);
    // Attach the number of blocks that loaded before the failure so the tier
    // can keep them (partial success). failed_index == count of blocks read OK.
    PyObject *etype, *evalue, *etb;
    PyErr_Fetch(&etype, &evalue, &etb);
    PyErr_NormalizeException(&etype, &evalue, &etb);
    if (evalue != nullptr) {
      PyObject* num = PyLong_FromSsize_t(failed_index);
      if (num != nullptr) {
        PyObject_SetAttrString(evalue, "num_succeeded", num);
        Py_DECREF(num);
      }
    }
    PyErr_Restore(etype, evalue, etb);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyMethodDef fs_io_C_methods[] = {
    {"batch_lookup", batch_lookup, METH_VARARGS,
     "batch_lookup(paths: list[str]) -> list[bool]\n"
     "\n"
     "Check file existence for a batch of paths."},
    {"batch_store_block", batch_store_block, METH_VARARGS,
     "batch_store_block(tmp_paths: list[str], dest_paths: list[str],\n"
     "                  buffers: list[bytes-like],\n"
     "                  use_o_direct: bool = True) -> None\n"
     "\n"
     "Store a batch of blocks, each from its own buffer, to disk. Raises on "
     "first error."},
    {"batch_load_block", batch_load_block, METH_VARARGS,
     "batch_load_block(source_paths: list[str],\n"
     "                 buffers: list[writable bytes-like],\n"
     "                 use_o_direct: bool = True) -> None\n"
     "\n"
     "Load a batch of blocks from disk into corresponding buffers. "
     "Raises on first error."},
    {"create_pool", create_pool, METH_VARARGS,
     "create_pool(primary_kv_view: writable bytes-like, block_size: int,\n"
     "            use_o_direct: bool = True) -> capsule\n"
     "\n"
     "Create a work-stealing I/O pool bound to one shared KV buffer, "
     "pinned for the pool's lifetime."},
    {"destroy_pool", destroy_pool, METH_VARARGS,
     "destroy_pool(pool: capsule) -> None\n"
     "\n"
     "Release the pool's pinned buffer immediately."},
    {"push_load", push_load, METH_VARARGS,
     "push_load(pool: capsule, job_id: int, paths: list[str],\n"
     "          offsets: list[int]) -> None\n"
     "\n"
     "Enqueue every block of a load job into the pool's load WorkQueue."},
    {"push_store", push_store, METH_VARARGS,
     "push_store(pool: capsule, job_id: int, paths: list[str],\n"
     "           offsets: list[int]) -> None\n"
     "\n"
     "Enqueue every block of a store job into the pool's store WorkQueue."},
    {"queue_nonempty", queue_nonempty, METH_VARARGS,
     "queue_nonempty(pool: capsule, load_priority: bool) -> bool\n"
     "\n"
     "Check whether a worker with the given priority has any claimable "
     "work (own queue, or the other queue as fallback)."},
    {"wait_and_run", wait_and_run, METH_VARARGS,
     "wait_and_run(pool: capsule, load_priority: bool) -> None\n"
     "\n"
     "Drain and execute work items for a worker with the given priority, "
     "blocking without spinning whenever both queues run dry, for up to "
     "a 30s idle timeout since work was last found. Releases the GIL for "
     "the whole call; results are pushed to the pool's ResultQueue as "
     "each item finishes. request_stop() wakes an idling call "
     "immediately."},
    {"pop_all_results", pop_all_results, METH_VARARGS,
     "pop_all_results(pool: capsule) -> list[tuple[int, int, int, float]]\n"
     "\n"
     "Drain every currently-available (job_id, index, errno, "
     "transfer_time) result. errno == 0 means success."},
    {"clear_work_queue", clear_work_queue, METH_VARARGS,
     "clear_work_queue(pool: capsule) -> None\n"
     "\n"
     "Clear both the load and store WorkQueues. Used by shutdown()."},
    {"request_stop", request_stop, METH_VARARGS,
     "request_stop(pool: capsule) -> None\n"
     "\n"
     "Wake every worker idling inside wait_and_run() immediately, "
     "instead of waiting out its idle timeout. Used by shutdown()."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef fs_io_C_module = {
    PyModuleDef_HEAD_INIT, "fs_io_C", "Filesystem helpers for KV offload", -1,
    fs_io_C_methods,
};

PyMODINIT_FUNC PyInit_fs_io_C(void) { return PyModule_Create(&fs_io_C_module); }

}  // extern "C"
