#pragma once

#include <memory>
#include <set>

#include "input.hpp"
#include "runtime/tensor.hpp"

namespace tinyinfer {
    class Node;

/**
 * \brief Output objects affiliated to Node
 */
    class Output {
    protected:
        Node *m_node;
        size_t m_index;
        std::shared_ptr<runtime::Tensor> m_tensor = nullptr;
        std::set<Input *> m_inputs;

    public:
        Output(Node *node, size_t index);

        std::shared_ptr<Node> get_node() const;

        size_t get_index() const { return m_index; }

        std::shared_ptr<runtime::Tensor> get_tensor_ptr() const { return m_tensor; }

        void set_tensor_ptr(const std::shared_ptr<runtime::Tensor> &tensor);

        void add_input(Input *input);

        void remove_input(Input *input);

        const std::set<Input *> &get_inputs() const { return m_inputs; }

        const Shape &get_shape() const;
    };
}