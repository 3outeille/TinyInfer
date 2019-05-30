#pragma once

#include <vector>

namespace tinyinfer {
/**
*\brief Shape class to descripe the shape of tensor
*/
    class Shape : public std::vector<size_t> {
    public:
        Shape() {}

        Shape(const std::vector<size_t> &axis_length)
                : std::vector<size_t>(axis_length) {}

        Shape(const Shape &axis_length) : std::vector<size_t>(axis_length) {}
    };
}