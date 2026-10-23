#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

// Compute one American CRR tree. Python retains the existing parameter,
// adjacent-depth averaging, IV solver, and Greek logic.
static PyObject* price_single_crr(PyObject*, PyObject* args) {
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
        std::vector<double> stock_prices(static_cast<size_t>(steps) + 1);
        std::vector<double> option_values(static_cast<size_t>(steps) + 1);
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

static PyMethodDef methods[] = {
    {"price_single_crr", price_single_crr, METH_VARARGS,
     "Price one American CRR tree using prevalidated parameters."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_crr_native", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

PyMODINIT_FUNC PyInit__crr_native(void) {
    return PyModule_Create(&module);
}
