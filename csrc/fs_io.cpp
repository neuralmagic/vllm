// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

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

struct IOResult {
  std::string error_message;
  int errnum = 0;

  void clear() {
    error_message.clear();
    errnum = 0;
  }

  bool has_error() const { return !error_message.empty() || errnum != 0; }

  void maybe_set_errnum() {
    if (!has_error()) {
      errnum = errno;
    }
  }

  void maybe_set_error_message(std::string msg) {
    if (!has_error()) {
      error_message = std::move(msg);
    }
  }

  PyObject* to_py_error(const char* path_for_errno) const {
    if (!error_message.empty()) {
      return PyErr_Format(PyExc_IOError, "%s", error_message.c_str());
    }
    if (errnum != 0) {
      errno = errnum;
      return PyErr_SetFromErrnoWithFilename(PyExc_OSError, path_for_errno);
    }
    return nullptr;
  }
};

inline bool ensure_parent_dirs(const std::string& path, IOResult* io_result) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) {
    return true;
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    io_result->maybe_set_error_message("Failed to create parent dirs for " +
                                       path + ": " + ec.message());
    return false;
  }
  return true;
}

inline int safe_open(const char* path, int flags, mode_t mode,
                     IOResult* result) {
  const int fd = open(path, flags, mode);
  if (fd < 0) {
    result->maybe_set_errnum();
  }
  return fd;
}

inline ssize_t safe_write(int fd, const char* src, size_t size,
                          IOResult* result) {
  const ssize_t written = write(fd, src, size);
  if (written < 0) {
    result->maybe_set_errnum();
  } else if (static_cast<size_t>(written) != size) {
    result->maybe_set_error_message("Short write: expected " +
                                    std::to_string(size) + " bytes, wrote " +
                                    std::to_string(written));
  }
  return written;
}

inline ssize_t safe_read(int fd, char* dst, size_t size, IOResult* result) {
  const ssize_t bytes_read = read(fd, dst, size);
  if (bytes_read < 0) {
    result->maybe_set_errnum();
  } else if (static_cast<size_t>(bytes_read) != size) {
    result->maybe_set_error_message("Short read: expected " +
                                    std::to_string(size) + " bytes, read " +
                                    std::to_string(bytes_read));
  }
  return bytes_read;
}

inline bool safe_close(int fd, IOResult* result) {
  if (close(fd) != 0) {
    result->maybe_set_errnum();
    return false;
  }
  return true;
}

inline bool safe_rename(const char* src, const char* dst, IOResult* result) {
  if (rename(src, dst) != 0) {
    result->maybe_set_errnum();
    return false;
  }
  return true;
}

inline void safe_unlink(const char* path, IOResult* result) {
  if (unlink(path) != 0) {
    result->maybe_set_errnum();
  }
}

// Core single-block store: src/size are raw pointer + byte count.
inline bool _store_block(const char* tmp_path, const char* dest_path,
                         const char* src, size_t size, IOResult* result) {
  result->clear();

  if (access(dest_path, F_OK) == 0) {
    // Already present
    return true;
  }

  if (!ensure_parent_dirs(dest_path, result)) {
    return false;
  }

  const int fd =
      safe_open(tmp_path, O_CREAT | O_EXCL | O_WRONLY | O_TRUNC | kODirectFlag,
                0644, result);
  if (fd < 0) {
    return false;
  }

  safe_write(fd, src, size, result);
  safe_close(fd, result);

  if (!result->has_error()) {
    safe_rename(tmp_path, dest_path, result);
  }

  if (result->has_error()) {
    safe_unlink(tmp_path, result);
    return false;
  }

  return true;
}

// Core single-block load: dst/size are raw pointer + byte count.
inline bool _load_block(const char* source_path, char* dst, size_t size,
                        IOResult* result) {
  result->clear();

  int fd = safe_open(source_path, O_RDONLY | kODirectFlag, 0, result);
  if (fd < 0) {
    unlink(source_path);
    return false;
  }

  safe_read(fd, dst, size, result);
  safe_close(fd, result);

  if (result->has_error()) {
    unlink(source_path);
    return false;
  }

  return true;
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

// Helper: extract a list[int] of length n into a vector<Py_ssize_t>.
// Returns false and sets a Python exception on error.
inline bool extract_ssize_list(PyObject* list, Py_ssize_t n,
                               std::vector<Py_ssize_t>& out) {
  for (Py_ssize_t i = 0; i < n; i++) {
    out[i] = PyLong_AsSsize_t(PyList_GetItem(list, i));
    if (out[i] == -1 && PyErr_Occurred()) {
      return false;
    }
  }
  return true;
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

static PyObject* store_block(PyObject* /*self*/, PyObject* args) {
  const char* tmp_path = nullptr;
  const char* dest_path = nullptr;
  PyObject* buffer_obj = nullptr;
  if (!PyArg_ParseTuple(args, "ssO", &tmp_path, &dest_path, &buffer_obj)) {
    return nullptr;
  }

  Py_buffer view;
  if (PyObject_GetBuffer(buffer_obj, &view, PyBUF_SIMPLE) != 0) {
    return nullptr;
  }

  IOResult io_result;
  bool ok = false;

  {
    Py_BEGIN_ALLOW_THREADS ok =
        _store_block(tmp_path, dest_path, static_cast<const char*>(view.buf),
                     static_cast<size_t>(view.len), &io_result);
    Py_END_ALLOW_THREADS
  }

  PyBuffer_Release(&view);

  if (!ok) {
    return io_result.to_py_error(dest_path);
  }

  Py_RETURN_NONE;
}

static PyObject* load_block(PyObject* /*self*/, PyObject* args) {
  const char* source_path = nullptr;
  PyObject* view_obj = nullptr;
  Py_ssize_t expected_size = 0;
  if (!PyArg_ParseTuple(args, "sOn", &source_path, &view_obj, &expected_size)) {
    return nullptr;
  }
  if (expected_size < 0) {
    PyErr_SetString(PyExc_ValueError, "expected_size must be >= 0");
    return nullptr;
  }

  Py_buffer view;
  if (PyObject_GetBuffer(view_obj, &view, PyBUF_WRITABLE) != 0) {
    return nullptr;
  }

  IOResult io_result;
  bool ok = false;

  {
    Py_BEGIN_ALLOW_THREADS ok =
        _load_block(source_path, static_cast<char*>(view.buf),
                    static_cast<size_t>(expected_size), &io_result);
    Py_END_ALLOW_THREADS
  }

  PyBuffer_Release(&view);

  if (!ok) {
    return io_result.to_py_error(source_path);
  }

  Py_RETURN_NONE;
}

/// @brief Store a batch of blocks from a shared buffer to disk.
/// @param tmp_paths  list[str] – one temp path per block.
/// @param dest_paths list[str] – one destination path per block.
/// @param buf        bytes-like – the shared source buffer.
/// @param offsets    list[int] – byte offset into buf for each block.
/// @param block_size int – byte length of each block.
/// @note Releases the GIL for the entire batch. Raises on first error.
static PyObject* batch_store_block(PyObject* /*self*/, PyObject* args) {
  PyObject* tmp_paths_obj = nullptr;
  PyObject* dest_paths_obj = nullptr;
  PyObject* buffer_obj = nullptr;
  PyObject* offsets_obj = nullptr;
  Py_ssize_t block_size = 0;

  if (!PyArg_ParseTuple(args, "O!O!OO!n", &PyList_Type, &tmp_paths_obj,
                        &PyList_Type, &dest_paths_obj, &buffer_obj,
                        &PyList_Type, &offsets_obj, &block_size)) {
    return nullptr;
  }
  if (block_size < 0) {
    PyErr_SetString(PyExc_ValueError, "block_size must be >= 0");
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(tmp_paths_obj);
  if (PyList_Size(dest_paths_obj) != n || PyList_Size(offsets_obj) != n) {
    PyErr_SetString(
        PyExc_ValueError,
        "tmp_paths, dest_paths and offsets must have the same length");
    return nullptr;
  }

  std::vector<const char*> tmp_paths(n);
  std::vector<const char*> dest_paths(n);
  std::vector<Py_ssize_t> offsets(n);

  if (!extract_str_list(tmp_paths_obj, n, tmp_paths)) return nullptr;
  if (!extract_str_list(dest_paths_obj, n, dest_paths)) return nullptr;
  if (!extract_ssize_list(offsets_obj, n, offsets)) return nullptr;

  Py_buffer view;
  if (PyObject_GetBuffer(buffer_obj, &view, PyBUF_SIMPLE) != 0) {
    return nullptr;
  }

  IOResult io_result;
  const char* failed_path = nullptr;

  {
    Py_BEGIN_ALLOW_THREADS const char* buf = static_cast<const char*>(view.buf);
    for (size_t i = 0; i < static_cast<size_t>(n); i++) {
      if (!_store_block(tmp_paths[i], dest_paths[i], buf + offsets[i],
                        static_cast<size_t>(block_size), &io_result)) {
        failed_path = dest_paths[i];
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  PyBuffer_Release(&view);

  if (failed_path) {
    return io_result.to_py_error(failed_path);
  }

  Py_RETURN_NONE;
}

/// @brief Load a batch of blocks from disk into a shared buffer.
/// @param source_paths list[str] – one source path per block.
/// @param view         writable bytes-like – the shared destination buffer.
/// @param offsets      list[int] – byte offset into view for each block.
/// @param block_size   int – byte length of each block.
/// @note Releases the GIL for the entire batch. Raises on first error.
static PyObject* batch_load_block(PyObject* /*self*/, PyObject* args) {
  PyObject* source_paths_obj = nullptr;
  PyObject* view_obj = nullptr;
  PyObject* offsets_obj = nullptr;
  Py_ssize_t block_size = 0;

  if (!PyArg_ParseTuple(args, "O!OO!n", &PyList_Type, &source_paths_obj,
                        &view_obj, &PyList_Type, &offsets_obj, &block_size)) {
    return nullptr;
  }
  if (block_size < 0) {
    PyErr_SetString(PyExc_ValueError, "block_size must be >= 0");
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(source_paths_obj);
  if (PyList_Size(offsets_obj) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "source_paths and offsets must have the same length");
    return nullptr;
  }

  std::vector<const char*> source_paths(n);
  std::vector<Py_ssize_t> offsets(n);

  if (!extract_str_list(source_paths_obj, n, source_paths)) return nullptr;
  if (!extract_ssize_list(offsets_obj, n, offsets)) return nullptr;

  Py_buffer view;
  if (PyObject_GetBuffer(view_obj, &view, PyBUF_WRITABLE) != 0) {
    return nullptr;
  }

  IOResult io_result;
  const char* failed_path = nullptr;

  {
    Py_BEGIN_ALLOW_THREADS char* buf = static_cast<char*>(view.buf);
    for (size_t i = 0; i < static_cast<size_t>(n); i++) {
      if (!_load_block(source_paths[i], buf + offsets[i],
                       static_cast<size_t>(block_size), &io_result)) {
        failed_path = source_paths[i];
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  PyBuffer_Release(&view);

  if (failed_path) {
    return io_result.to_py_error(failed_path);
  }

  Py_RETURN_NONE;
}

static PyMethodDef fs_io_C_methods[] = {
    {"batch_lookup", batch_lookup, METH_VARARGS,
     "batch_lookup(paths: list[str]) -> list[bool]\n"
     "\n"
     "Check file existence for a batch of paths."},
    {"store_block", store_block, METH_VARARGS,
     "store_block(tmp_path: str, dest_path: str, buf: bytes-like) -> None"},
    {"load_block", load_block, METH_VARARGS,
     "load_block(source_path: str, buf: writable bytes-like, "
     "expected_size: int) -> None"},
    {"batch_store_block", batch_store_block, METH_VARARGS,
     "batch_store_block(tmp_paths: list[str], dest_paths: list[str],\n"
     "                  buf: bytes-like, offsets: list[int],\n"
     "                  block_size: int) -> None\n"
     "\n"
     "Store a batch of blocks from a shared buffer to disk. Raises on first "
     "error."},
    {"batch_load_block", batch_load_block, METH_VARARGS,
     "batch_load_block(source_paths: list[str], view: writable bytes-like,\n"
     "                 offsets: list[int], block_size: int) -> None\n"
     "\n"
     "Load a batch of blocks from disk into a shared buffer. Raises on first "
     "error."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef fs_io_C_module = {
    PyModuleDef_HEAD_INIT, "fs_io_C", "Filesystem helpers for KV offload", -1,
    fs_io_C_methods,
};

PyMODINIT_FUNC PyInit_fs_io_C(void) { return PyModule_Create(&fs_io_C_module); }

}  // extern "C"
