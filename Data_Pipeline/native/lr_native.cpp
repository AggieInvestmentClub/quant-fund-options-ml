#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

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

static PyMethodDef methods[] = {
    {"price_single_lr", price_single_lr, METH_VARARGS,
     "Price one American Leisen-Reimer tree using prevalidated parameters."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_lr_native", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

PyMODINIT_FUNC PyInit__lr_native(void) {
    return PyModule_Create(&module);
}

