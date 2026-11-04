#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <zlib.h>

inline double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double black_scholes_price(double S, double K, double T, double r, double sigma, bool is_call, double q) {
    if (S <= 0.0 || K <= 0.0 || sigma <= 0.0) return 0.0;
    if (T <= 0.0) {
        return is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }
    const double sqrt_t = std::sqrt(T);
    const double sigma_sqrt_t = sigma * sqrt_t;
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_t;
    const double d2 = d1 - sigma_sqrt_t;

    const double discounted_spot = S * std::exp(-q * T);
    const double discounted_strike = K * std::exp(-r * T);
    if (is_call) {
        return discounted_spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2);
    } else {
        return discounted_strike * norm_cdf(-d2) - discounted_spot * norm_cdf(-d1);
    }
}

inline std::pair<double, double> peizer_pratt(double z, int n) {
    if (z == 0.0) return {0.5, 0.5};
    const double n_d = static_cast<double>(n);
    const double denom = n_d + 1.0 / 3.0 + 0.1 / (n_d + 1.0);
    const double ratio = z / denom;
    const double w = ratio * ratio * (n_d + 1.0 / 6.0);
    if (w > 700.0) {
        return z > 0 ? std::make_pair(1.0 - 1e-15, 1e-15) : std::make_pair(1e-15, 1.0 - 1e-15);
    }
    const double exp_neg_w = std::exp(-w);
    const double sqrt_term = std::sqrt(std::max(0.0, 1.0 - exp_neg_w));
    if (z > 0) {
        const double h = 0.5 + 0.5 * sqrt_term;
        const double one_minus_h = 0.5 * exp_neg_w / (1.0 + sqrt_term);
        return {h, one_minus_h};
    } else {
        const double one_minus_h = 0.5 + 0.5 * sqrt_term;
        const double h = 0.5 * exp_neg_w / (1.0 + sqrt_term);
        return {h, one_minus_h};
    }
}

struct LRParams {
    int n_steps;
    double dt;
    double u;
    double d;
    double p;
    double disc;
    bool valid;
};

inline LRParams lr_parameters(double S, double K, double T, double r, double sigma, double q, int steps) {
    LRParams res = {0, 0, 0, 0, 0, 0, false};
    if (S <= 0.0 || K <= 0.0 || sigma <= 0.0 || T <= 0.0) return res;

    int n_steps = (steps < 3 ? 3 : steps);
    if (n_steps % 2 == 0) n_steps += 1;
    const double dt = T / n_steps;

    const double sqrt_t = std::sqrt(T);
    const double sigma_sqrt_t = sigma * sqrt_t;
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_t;
    const double d2 = d1 - sigma_sqrt_t;

    const auto [p, one_minus_p] = peizer_pratt(d2, n_steps);
    const auto [p_prime, one_minus_p_prime] = peizer_pratt(d1, n_steps);

    if (p <= 0.0 || one_minus_p <= 0.0) return res;

    const double growth = std::exp((r - q) * dt);
    const double u = growth * (p_prime / p);
    const double d = growth * (one_minus_p_prime / one_minus_p);

    if (!std::isfinite(u) || !std::isfinite(d) || u <= 0.0 || d <= 0.0) return res;

    const double disc = std::exp(-r * dt);
    return {n_steps, dt, u, d, p, disc, true};
}

inline double evaluate_lr_tree(double spot, double strike, double up, double down,
                               double up_probability, double discount, int steps,
                               bool is_call, double* stock_prices, double* option_values) {
    for (int index = 0; index <= steps; ++index) {
        const double stock = spot * std::pow(up, steps - index) * std::pow(down, index);
        if (!std::isfinite(stock)) return std::numeric_limits<double>::quiet_NaN();
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
    return option_values[0];
}

inline double american_price_lr(double S, double K, double T, double r, double sigma, bool is_call, double q, int steps,
                                double* stock_prices, double* option_values) {
    if (S <= 0.0 || K <= 0.0 || sigma <= 0.0) return 0.0;
    if (T <= 0.0) {
        return is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }
    if (is_call && q <= 1e-12) {
        return black_scholes_price(S, K, T, r, sigma, is_call, q);
    }
    LRParams params = lr_parameters(S, K, T, r, sigma, q, steps);
    if (!params.valid) return std::numeric_limits<double>::quiet_NaN();

    return evaluate_lr_tree(S, K, params.u, params.d, params.p, params.disc,
                             params.n_steps, is_call, stock_prices, option_values);
}

inline void compute_greeks_single(double S, double K, double T, double r, double sigma, double q, bool is_call, int steps,
                                  double* stock_prices, double* option_values,
                                  double& delta, double& gamma, double& theta, double& vega) {
    if (!std::isfinite(sigma) || !std::isfinite(S) || !std::isfinite(K) || !std::isfinite(T) || !std::isfinite(r)) {
        delta = gamma = theta = vega = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    if (q < 0.0 || !std::isfinite(q)) q = 0.0;
    if (sigma <= 0.0 || S <= 0.0 || K <= 0.0 || T <= 0.0) {
        delta = gamma = theta = vega = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    const double dS = std::max(0.01, 0.01 * S);
    const double d_sigma = std::max(1e-4, 0.01 * sigma);
    const double dT = std::min(1.0 / 365.25, std::max(T / 10.0, 1.0 / 365.25));

    const double base = american_price_lr(S, K, T, r, sigma, is_call, q, steps, stock_prices, option_values);
    const double up = american_price_lr(S + dS, K, T, r, sigma, is_call, q, steps, stock_prices, option_values);
    const double down = american_price_lr(std::max(1e-8, S - dS), K, T, r, sigma, is_call, q, steps, stock_prices, option_values);
    const double vega_up = american_price_lr(S, K, T, r, sigma + d_sigma, is_call, q, steps, stock_prices, option_values);
    const double vega_down = american_price_lr(S, K, T, r, std::max(1e-6, sigma - d_sigma), is_call, q, steps, stock_prices, option_values);
    const double shorter = american_price_lr(S, K, std::max(1e-6, T - dT), r, sigma, is_call, q, steps, stock_prices, option_values);

    if (!std::isfinite(base) || !std::isfinite(up) || !std::isfinite(down) ||
        !std::isfinite(vega_up) || !std::isfinite(vega_down) || !std::isfinite(shorter)) {
        delta = gamma = theta = vega = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    delta = (up - down) / (2.0 * dS);
    gamma = (up - 2.0 * base + down) / (dS * dS);
    theta = (shorter - base) / dT;
    vega = (vega_up - vega_down) / (2.0 * d_sigma);

    if (!std::isfinite(delta) || !std::isfinite(gamma) || !std::isfinite(theta) || !std::isfinite(vega)) {
        delta = gamma = theta = vega = std::numeric_limits<double>::quiet_NaN();
    }
}

// Compute one American Leisen-Reimer tree. Python retains the parameter
// computation (Peizer-Pratt inversion), adjacent step forcing to odd, IV
// solver, and Greek logic.
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

        double price = evaluate_lr_tree(spot, strike, up, down, up_probability, discount, steps,
                                        is_call != 0, stock_prices, option_values);
        return PyFloat_FromDouble(price);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
}

struct ScopedBuffer {
    Py_buffer buf;
    bool active = false;
    ~ScopedBuffer() {
        if (active) {
            PyBuffer_Release(&buf);
        }
    }
};

static PyObject* batch_regular_greeks(PyObject*, PyObject* args) {
    PyObject *spots_obj, *strikes_obj, *times_obj, *rates_obj, *sigmas_obj, *divs_obj, *is_calls_obj;
    PyObject *deltas_obj, *gammas_obj, *thetas_obj, *vegas_obj;
    int steps = 65;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO|i",
                          &spots_obj, &strikes_obj, &times_obj, &rates_obj,
                          &sigmas_obj, &divs_obj, &is_calls_obj,
                          &deltas_obj, &gammas_obj, &thetas_obj, &vegas_obj,
                          &steps)) {
        return nullptr;
    }

    ScopedBuffer b_spots, b_strikes, b_times, b_rates, b_sigmas, b_divs, b_calls;
    ScopedBuffer b_deltas, b_gammas, b_thetas, b_vegas;

    if (PyObject_GetBuffer(spots_obj, &b_spots.buf, PyBUF_SIMPLE) != 0) return nullptr;
    b_spots.active = true;
    if (PyObject_GetBuffer(strikes_obj, &b_strikes.buf, PyBUF_SIMPLE) != 0) return nullptr;
    b_strikes.active = true;
    if (PyObject_GetBuffer(times_obj, &b_times.buf, PyBUF_SIMPLE) != 0) return nullptr;
    b_times.active = true;
    if (PyObject_GetBuffer(rates_obj, &b_rates.buf, PyBUF_SIMPLE) != 0) return nullptr;
    b_rates.active = true;
    if (PyObject_GetBuffer(sigmas_obj, &b_sigmas.buf, PyBUF_SIMPLE) != 0) return nullptr;
    b_sigmas.active = true;
    if (PyObject_GetBuffer(divs_obj, &b_divs.buf, PyBUF_SIMPLE) != 0) return nullptr;
    b_divs.active = true;
    if (PyObject_GetBuffer(is_calls_obj, &b_calls.buf, PyBUF_SIMPLE) != 0) return nullptr;
    b_calls.active = true;

    if (PyObject_GetBuffer(deltas_obj, &b_deltas.buf, PyBUF_WRITABLE) != 0) return nullptr;
    b_deltas.active = true;
    if (PyObject_GetBuffer(gammas_obj, &b_gammas.buf, PyBUF_WRITABLE) != 0) return nullptr;
    b_gammas.active = true;
    if (PyObject_GetBuffer(thetas_obj, &b_thetas.buf, PyBUF_WRITABLE) != 0) return nullptr;
    b_thetas.active = true;
    if (PyObject_GetBuffer(vegas_obj, &b_vegas.buf, PyBUF_WRITABLE) != 0) return nullptr;
    b_vegas.active = true;

    const Py_ssize_t n = b_spots.buf.len / sizeof(double);
    if (b_spots.buf.len % sizeof(double) != 0 ||
        b_strikes.buf.len != b_spots.buf.len ||
        b_times.buf.len != b_spots.buf.len ||
        b_rates.buf.len != b_spots.buf.len ||
        b_sigmas.buf.len != b_spots.buf.len ||
        b_divs.buf.len != b_spots.buf.len ||
        b_calls.buf.len != n * static_cast<Py_ssize_t>(sizeof(int8_t)) ||
        b_deltas.buf.len != b_spots.buf.len ||
        b_gammas.buf.len != b_spots.buf.len ||
        b_thetas.buf.len != b_spots.buf.len ||
        b_vegas.buf.len != b_spots.buf.len) {
        PyErr_SetString(PyExc_ValueError, "Array shapes or byte sizes mismatch");
        return nullptr;
    }

    const double* spots = static_cast<const double*>(b_spots.buf.buf);
    const double* strikes = static_cast<const double*>(b_strikes.buf.buf);
    const double* times = static_cast<const double*>(b_times.buf.buf);
    const double* rates = static_cast<const double*>(b_rates.buf.buf);
    const double* sigmas = static_cast<const double*>(b_sigmas.buf.buf);
    const double* divs = static_cast<const double*>(b_divs.buf.buf);
    const int8_t* is_calls = static_cast<const int8_t*>(b_calls.buf.buf);

    double* deltas = static_cast<double*>(b_deltas.buf.buf);
    double* gammas = static_cast<double*>(b_gammas.buf.buf);
    double* thetas = static_cast<double*>(b_thetas.buf.buf);
    double* vegas = static_cast<double*>(b_vegas.buf.buf);

    constexpr int STACK_STEPS = 1024;
    double stack_stock[STACK_STEPS + 1];
    double stack_values[STACK_STEPS + 1];
    std::vector<double> heap_stock;
    std::vector<double> heap_values;
    double* stock_prices = stack_stock;
    double* option_values = stack_values;

    int resolved_steps = (steps < 3 ? 3 : steps);
    if (resolved_steps % 2 == 0) resolved_steps += 1;

    if (resolved_steps > STACK_STEPS) {
        try {
            heap_stock.resize(static_cast<size_t>(resolved_steps) + 1);
            heap_values.resize(static_cast<size_t>(resolved_steps) + 1);
            stock_prices = heap_stock.data();
            option_values = heap_values.data();
        } catch (const std::bad_alloc&) {
            PyErr_NoMemory();
            return nullptr;
        }
    }

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) {
        int8_t call_flag = is_calls[i];
        if (call_flag != 0 && call_flag != 1) {
            deltas[i] = gammas[i] = thetas[i] = vegas[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        bool is_call = (call_flag == 1);
        compute_greeks_single(spots[i], strikes[i], times[i], rates[i], sigmas[i], divs[i], is_call, resolved_steps,
                              stock_prices, option_values,
                              deltas[i], gammas[i], thetas[i], vegas[i]);
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
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
    {"batch_regular_greeks", batch_regular_greeks, METH_VARARGS,
     "Batch compute regular option Greeks directly in C++."},
    {nullptr, nullptr, 0, nullptr},
};


static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_lr_native", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

PyMODINIT_FUNC PyInit__lr_native(void) {
    return PyModule_Create(&module);
}

