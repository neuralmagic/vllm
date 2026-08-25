// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

enum TaskStatus : int8_t {
  kStatusSuccess = 0,
  kStatusFailed = 1,
};

// Publishes status[i] with release semantics: everything the calling thread
// wrote before this call (in particular, the block's data via _store_block /
// _load_block) is guaranteed visible to any thread that later reads
// status[i] with an acquire load and observes this value.
inline void set_status(int8_t* status, Py_ssize_t i, TaskStatus value) {
  std::atomic_ref<int8_t>(status[i]).store(value, std::memory_order_release);
}

// Helper: acquire a writable int8, length-n buffer for the status array.
// Returns false and sets a Python exception on error.
inline bool extract_status_buffer(PyObject* status_obj, Py_ssize_t n,
                                  Py_buffer& out) {
  if (PyObject_GetBuffer(status_obj, &out, PyBUF_WRITABLE) != 0) {
    return false;
  }
  if (out.len != n || out.itemsize != 1) {
    PyBuffer_Release(&out);
    PyErr_SetString(PyExc_ValueError,
                    "status buffer must be a writable int8 array of length n");
    return false;
  }
  return true;
}

// A single failed task's OS-level error, recorded without touching the
// Python C-API so it can be populated while the GIL is released.
struct Failure {
  Py_ssize_t index;
  int err;
};

// Set once in PyInit_fs_io_C; a subclass of OSError raised when one or more
// tasks in a batch fail. The message summarizes every failure (path, errno,
// strerror) so a single log line captures the whole batch.
PyObject* g_batch_io_error = nullptr;

// Raises g_batch_io_error with a message summarizing every failure. Always
// returns nullptr so callers can `return raise_batch_io_error(...);`.
inline PyObject* raise_batch_io_error(const std::vector<const char*>& paths,
                                      const std::vector<Failure>& failures) {
  std::string message = "batch I/O failed for " +
                        std::to_string(failures.size()) + "/" +
                        std::to_string(paths.size()) + " blocks: ";
  for (size_t i = 0; i < failures.size(); i++) {
    const Failure& f = failures[i];
    if (i > 0) {
      message += ", ";
    }
    message += paths[f.index];
    message += " (errno ";
    message += std::to_string(f.err);
    message += ": ";
    message += strerror(f.err);
    message += ")";
  }
  PyErr_SetString(g_batch_io_error, message.c_str());
  return nullptr;
}

}  // namespace

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
/// @param use_o_direct bool – whether to open files with O_DIRECT. Ignored
///                     where O_DIRECT is unsupported by the platform.
/// @param status       writable int8 array, length == len(tmp_paths). Caller
///                     must pre-initialize every entry (e.g. to -1); this
///                     call updates each entry with a TaskStatus value as the
///                     function processes it.
/// @note Releases the GIL for the entire batch. Processes every block even
///       if some fail; raises BatchIOError (see raise_batch_io_error) if any
///       block failed, after status has been fully populated.
static PyObject* batch_store_block(PyObject* /*self*/, PyObject* args) {
  PyObject* tmp_paths_obj = nullptr;
  PyObject* dest_paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;
  PyObject* status_obj = nullptr;
  int use_o_direct = 1;

  if (!PyArg_ParseTuple(args, "O!O!O!pO", &PyList_Type, &tmp_paths_obj,
                        &PyList_Type, &dest_paths_obj, &PyList_Type,
                        &buffers_obj, &use_o_direct, &status_obj)) {
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

  Py_buffer status_buf;
  if (!extract_status_buffer(status_obj, n, status_buf)) {
    release_buffer_list(buffers);
    return nullptr;
  }
  int8_t* status = static_cast<int8_t*>(status_buf.buf);
  std::vector<Failure> failures;

  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      const char* buf = static_cast<const char*>(buffers[i].buf);
      const int err =
          _store_block(tmp_paths[i], dest_paths[i], buf,
                       static_cast<size_t>(buffers[i].len), use_o_direct);

      if (err == 0) {
        set_status(status, i, kStatusSuccess);
      } else {
        set_status(status, i, kStatusFailed);
        failures.push_back({i, err});
      }
    }
    Py_END_ALLOW_THREADS
  }

  PyBuffer_Release(&status_buf);
  release_buffer_list(buffers);

  if (!failures.empty()) {
    return raise_batch_io_error(dest_paths, failures);
  }

  Py_RETURN_NONE;
}

/// @brief Load a batch of blocks from disk, each into its own buffer.
/// @param source_paths list[str] – one source path per block.
/// @param buffers      list[writable bytes-like] – one destination buffer
///                     per block.
/// @param use_o_direct bool – whether to open files with O_DIRECT. Ignored
///                     where O_DIRECT is unsupported by the platform.
/// @param status       writable int8 array, length == len(source_paths).
///                     Caller must pre-initialize every entry (e.g. to -1);
///                     this call updates each entry with a TaskStatus value
///                     as the function processes it.
/// @note Releases the GIL for the entire batch. Processes every block even
///       if some fail; raises BatchIOError (see raise_batch_io_error) if any
///       block failed, after status has been fully populated.
static PyObject* batch_load_block(PyObject* /*self*/, PyObject* args) {
  PyObject* source_paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;
  PyObject* status_obj = nullptr;
  int use_o_direct = 1;

  if (!PyArg_ParseTuple(args, "O!O!pO", &PyList_Type, &source_paths_obj,
                        &PyList_Type, &buffers_obj, &use_o_direct,
                        &status_obj)) {
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

  Py_buffer status_buf;
  if (!extract_status_buffer(status_obj, n, status_buf)) {
    release_buffer_list(buffers);
    return nullptr;
  }
  int8_t* status = static_cast<int8_t*>(status_buf.buf);
  std::vector<Failure> failures;

  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      char* buf = static_cast<char*>(buffers[i].buf);
      const int err =
          _load_block(source_paths[i], buf, static_cast<size_t>(buffers[i].len),
                      use_o_direct);
      if (err == 0) {
        set_status(status, i, kStatusSuccess);
      } else {
        set_status(status, i, kStatusFailed);
        failures.push_back({i, err});
      }
    }
    Py_END_ALLOW_THREADS
  }

  PyBuffer_Release(&status_buf);
  release_buffer_list(buffers);

  if (!failures.empty()) {
    return raise_batch_io_error(source_paths, failures);
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
     "                  use_o_direct: bool,\n"
     "                  status: writable int8 array of length n) -> None\n"
     "\n"
     "Store a batch of blocks, each from its own buffer, to disk. Writes 0 "
     "(success) or 1 (failed) into status[i] as block i completes. Processes "
     "the whole batch even if some blocks fail; if any failed, raises "
     "BatchIOError summarizing every failed (path, errno, strerror)."},
    {"batch_load_block", batch_load_block, METH_VARARGS,
     "batch_load_block(source_paths: list[str],\n"
     "                 buffers: list[writable bytes-like],\n"
     "                 use_o_direct: bool,\n"
     "                 status: writable int8 array of length n) -> None\n"
     "\n"
     "Load a batch of blocks from disk into corresponding buffers. Writes 0 "
     "(success) or 1 (failed) into status[i] as block i completes. Processes "
     "the whole batch even if some blocks fail; if any failed, raises "
     "BatchIOError summarizing every failed (path, errno, strerror)."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef fs_io_C_module = {
    PyModuleDef_HEAD_INIT, "fs_io_C", "Filesystem helpers for KV offload", -1,
    fs_io_C_methods,
};

PyMODINIT_FUNC PyInit_fs_io_C(void) {
  PyObject* m = PyModule_Create(&fs_io_C_module);
  if (m == nullptr) {
    return nullptr;
  }

  PyObject* exc_type =
      PyErr_NewException("fs_io_C.BatchIOError", PyExc_OSError, nullptr);
  // PyModule_AddObjectRef takes its own reference instead of stealing ours,
  // so exc_type stays valid for our own cache below on success.
  if (exc_type == nullptr ||
      PyModule_AddObjectRef(m, "BatchIOError", exc_type) < 0) {
    Py_XDECREF(exc_type);
    Py_DECREF(m);
    return nullptr;
  }
  g_batch_io_error = exc_type;  // Cached ref, used by raise_batch_io_error.

  return m;
}

}  // extern "C"
