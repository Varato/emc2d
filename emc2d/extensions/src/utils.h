//
// Created by Chen on 14/12/2020.
//

#ifndef EMC2D_UTILS_H
#define EMC2D_UTILS_H

#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
py::array_t<T> make2dArray(size_t n1, size_t n2) {
    // construct row major (c contiguous) numpy array
    py::buffer_info info (
            nullptr,
            sizeof(T),
            py::format_descriptor<T>::format(),
            2,
            {n1, n2},
            {sizeof(T)*n2, sizeof(T)});
    return py::array_t<T>(info);
}

template<typename T>
py::array_t<T> make3dArray(size_t n0, size_t n1, size_t n2) {
    // construct row majoc (c contiguous) numpy array
    py::buffer_info info (
            nullptr,
            sizeof(T),
            py::format_descriptor<T>::format(),
            3,
            {n0, n1, n2},
            {sizeof(T)*n1*n2, sizeof(T)*n2, sizeof(T)});
    return py::array_t<T>(info);
}

#endif //EMC2D_UTILS_H
