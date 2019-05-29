#pragma once

#include <vector>

namespace tinyinfer {
class Shape : public std::vector<size_t> {
 public:
  Shape() {}
  Shape(const std::vector<size_t>& axis_length)
      : std::vector<size_t>(axis_length) {}
  Shape(const Shape& axis_length) : std::vector<size_t>(axis_length) {}
};
}