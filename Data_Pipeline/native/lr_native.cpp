#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <unordered_set>
#include <vector>
#include <zlib.h>

// Compute one American Leisen-Reimer tree. Python retains the parameter
// computation (Peizer-Pratt inversion), adjacent step forcing to odd, IV
// solver, and Greek logic.
//
// The lattice backward-induction loop is mathematically identical to the CRR
// kernel; only the (u, d, p, disc) values differ — they are computed in Python
// using the Leisen-Reimer / Peizer-Pratt Method 2 formulas before being
// passed here.
static PyObject* price_single_lr(PyObject*, PyObject* args) {
    double spot, strike, up, down, up_probability, discount;
    int steps, is_call;
    if (!PyArg_ParseTuple(args, "ddddddii", &spot, &strike, &up, &down,
                          &up_probability, &discount, &steps, &is_call)) {
        return nullptr;
    }
    if (steps < 1) {
        PyErr_SetString(PyExc_ValueError, "steps must be positive");
        return nullptr;
    }
    if (!std::isfinite(spot) || !std::isfinite(strike) || !std::isfinite(up) ||
        !std::isfinite(down) || !std::isfinite(up_probability) ||
        !std::isfinite(discount)) {
        return PyFloat_FromDouble(NAN);
    }

    try {
        constexpr int STACK_STEPS = 1024;
        double stack_stock[STACK_STEPS + 1];
        double stack_values[STACK_STEPS + 1];
        std::vector<double> heap_stock;
        std::vector<double> heap_values;
        double* stock_prices = stack_stock;
        double* option_values = stack_values;

        if (steps > STACK_STEPS) {
            heap_stock.resize(static_cast<size_t>(steps) + 1);
            heap_values.resize(static_cast<size_t>(steps) + 1);
            stock_prices = heap_stock.data();
            option_values = heap_values.data();
        }

        for (int index = 0; index <= steps; ++index) {
            const double stock = spot * std::pow(up, steps - index) * std::pow(down, index);
            if (!std::isfinite(stock)) return PyFloat_FromDouble(NAN);
            stock_prices[index] = stock;
            option_values[index] = std::max(
                is_call ? stock - strike : strike - stock, 0.0);
        }

        const double down_probability = 1.0 - up_probability;
        for (int step = steps - 1; step >= 0; --step) {
            for (int index = 0; index <= step; ++index) {
                stock_prices[index] /= up;
                const double continuation = discount *
                    (up_probability * option_values[index] +
                     down_probability * option_values[index + 1]);
                const double exercise = std::max(
                    is_call ? stock_prices[index] - strike : strike - stock_prices[index], 0.0);
                option_values[index] = std::max(continuation, exercise);
            }
        }
        return PyFloat_FromDouble(option_values[0]);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
}


static PyObject* stream_filter_opra_file(PyObject*, PyObject* args) {
    const char* file_path = nullptr;
    PyObject* symbols_obj = nullptr;
    const char* min_exp = nullptr;
    const char* max_exp = nullptr;

    if (!PyArg_ParseTuple(args, "sOzz", &file_path, &symbols_obj, &min_exp, &max_exp)) {
        return nullptr;
    }

    std::vector<std::string> symbols_vec;
    bool has_symbols = false;

    if (symbols_obj != Py_None) {
        if (PyUnicode_Check(symbols_obj)) {
            const char* s = PyUnicode_AsUTF8(symbols_obj);
            if (s && *s) {
                symbols_vec.push_back(s);
                has_symbols = true;
            }
        } else if (PySequence_Check(symbols_obj) || PyAnySet_Check(symbols_obj)) {
            PyObject* iter = PyObject_GetIter(symbols_obj);
            if (!iter) return nullptr;
            PyObject* item = nullptr;
            while ((item = PyIter_Next(iter)) != nullptr) {
                if (PyUnicode_Check(item)) {
                    const char* s = PyUnicode_AsUTF8(item);
                    if (s && *s) symbols_vec.push_back(s);
                }
                Py_DECREF(item);
            }
            Py_DECREF(iter);
            has_symbols = true;
            if (symbols_vec.empty()) {
                return PyBytes_FromStringAndSize("", 0);
            }
        }
    }

    std::string min_exp_str = min_exp ? min_exp : "";
    std::string max_exp_str = max_exp ? max_exp : "";

    std::string matched_output;
    bool open_success = true;

    Py_BEGIN_ALLOW_THREADS
    gzFile gz = gzopen(file_path, "rb");
    if (!gz) {
        open_success = false;
    } else {
        matched_output.reserve(64 * 1024);
        char buffer[256 * 1024];
        std::string remainder;
        remainder.reserve(1024);

        bool header_saved = false;
        bool filter_single = (has_symbols && symbols_vec.size() == 1);
        std::string single_prefix = filter_single ? ("O:" + symbols_vec[0]) : "";
        size_t single_prefix_len = single_prefix.size();

        std::unordered_set<std::string> sym_set;
        if (has_symbols && !filter_single) {
            sym_set.insert(symbols_vec.begin(), symbols_vec.end());
        }
        bool filter_set = !sym_set.empty();

        bool has_dte_filter = (min_exp_str.size() == 6 && max_exp_str.size() == 6);
        const char* min_ptr = has_dte_filter ? min_exp_str.data() : nullptr;
        const char* max_ptr = has_dte_filter ? max_exp_str.data() : nullptr;

        int bytes_read = 0;
        while ((bytes_read = gzread(gz, buffer, sizeof(buffer))) > 0) {
            size_t search_start = 0;
            size_t chunk_len = static_cast<size_t>(bytes_read);

            if (!remainder.empty()) {
                const char* nl = static_cast<const char*>(std::memchr(buffer, '\n', chunk_len));
                if (nl) {
                    size_t first_part_len = nl - buffer;
                    remainder.append(buffer, first_part_len);
                    search_start = first_part_len + 1;

                    if (!header_saved) {
                        matched_output.append(remainder);
                        matched_output.push_back('\n');
                        header_saved = true;
                    } else if (filter_single) {
                        if (remainder.size() > single_prefix_len + 15 &&
                            std::memcmp(remainder.data(), single_prefix.data(), single_prefix_len) == 0) {
                            if (!has_dte_filter) {
                                matched_output.append(remainder);
                                matched_output.push_back('\n');
                            } else {
                                const char* exp = remainder.data() + single_prefix_len;
                                if (std::memcmp(exp, min_ptr, 6) >= 0 && std::memcmp(exp, max_ptr, 6) <= 0) {
                                    matched_output.append(remainder);
                                    matched_output.push_back('\n');
                                }
                            }
                        }
                    } else {
                        if (remainder.size() >= 19 && remainder[0] == 'O' && remainder[1] == ':') {
                            const char* comma = static_cast<const char*>(std::memchr(remainder.data(), ',', remainder.size()));
                            if (comma && (comma - remainder.data()) >= 17) {
                                const char* exp = comma - 15;
                                if (!has_dte_filter || (std::memcmp(exp, min_ptr, 6) >= 0 && std::memcmp(exp, max_ptr, 6) <= 0)) {
                                    if (!filter_set) {
                                        matched_output.append(remainder);
                                        matched_output.push_back('\n');
                                    } else {
                                        std::string sym(remainder.data() + 2, exp - (remainder.data() + 2));
                                        if (sym_set.count(sym)) {
                                            matched_output.append(remainder);
                                            matched_output.push_back('\n');
                                        }
                                    }
                                }
                            }
                        }
                    }
                    remainder.clear();
                } else {
                    remainder.append(buffer, chunk_len);
                    continue;
                }
            }

            const char* ptr = buffer + search_start;
            const char* end_ptr = buffer + chunk_len;

            while (ptr < end_ptr) {
                const char* nl = static_cast<const char*>(std::memchr(ptr, '\n', end_ptr - ptr));
                if (!nl) {
                    remainder.assign(ptr, end_ptr - ptr);
                    break;
                }

                size_t line_len = nl - ptr;
                if (!header_saved) {
                    matched_output.append(ptr, line_len);
                    matched_output.push_back('\n');
                    header_saved = true;
                } else if (filter_single) {
                    if (line_len > single_prefix_len + 15 &&
                        std::memcmp(ptr, single_prefix.data(), single_prefix_len) == 0) {
                        if (!has_dte_filter) {
                            matched_output.append(ptr, line_len);
                            matched_output.push_back('\n');
                        } else {
                            const char* exp = ptr + single_prefix_len;
                            if (std::memcmp(exp, min_ptr, 6) >= 0 && std::memcmp(exp, max_ptr, 6) <= 0) {
                                matched_output.append(ptr, line_len);
                                matched_output.push_back('\n');
                            }
                        }
                    }
                } else {
                    if (line_len >= 19 && ptr[0] == 'O' && ptr[1] == ':') {
                        const char* comma = static_cast<const char*>(std::memchr(ptr, ',', line_len));
                        if (comma && (comma - ptr) >= 17) {
                            const char* exp = comma - 15;
                            if (!has_dte_filter || (std::memcmp(exp, min_ptr, 6) >= 0 && std::memcmp(exp, max_ptr, 6) <= 0)) {
                                if (!filter_set) {
                                    matched_output.append(ptr, line_len);
                                    matched_output.push_back('\n');
                                } else {
                                    std::string sym(ptr + 2, exp - (ptr + 2));
                                    if (sym_set.count(sym)) {
                                        matched_output.append(ptr, line_len);
                                        matched_output.push_back('\n');
                                    }
                                }
                            }
                        }
                    }
                }
                ptr = nl + 1;
            }
        }
        gzclose(gz);

        if (!remainder.empty()) {
            if (filter_single) {
                if (remainder.size() > single_prefix_len + 15 &&
                    std::memcmp(remainder.data(), single_prefix.data(), single_prefix_len) == 0) {
                    if (!has_dte_filter) {
                        matched_output.append(remainder);
                        matched_output.push_back('\n');
                    } else {
                        const char* exp = remainder.data() + single_prefix_len;
                        if (std::memcmp(exp, min_ptr, 6) >= 0 && std::memcmp(exp, max_ptr, 6) <= 0) {
                            matched_output.append(remainder);
                            matched_output.push_back('\n');
                        }
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (!open_success) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, file_path);
        return nullptr;
    }

    return PyBytes_FromStringAndSize(matched_output.data(), matched_output.size());
}

static PyMethodDef methods[] = {
    {"price_single_lr", price_single_lr, METH_VARARGS,
     "Price one American Leisen-Reimer tree using prevalidated parameters."},
    {"stream_filter_opra_file", stream_filter_opra_file, METH_VARARGS,
     "Fast streaming decompression and byte-level line filtering for raw OPRA gzip files."},
    {nullptr, nullptr, 0, nullptr},
};


static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_lr_native", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

PyMODINIT_FUNC PyInit__lr_native(void) {
    return PyModule_Create(&module);
}

