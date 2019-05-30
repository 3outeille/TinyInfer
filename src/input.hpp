#pragma once

#include "runtime/tensor.hpp"

namespace tinyinfer {
    class Node;

    class Output;

/**
 * \brief Output objects affiliated to Node
 */
    class Input {
        friend class Node;

    protected:
        Node *m_node;
        size_t m_index;
        Output *m_output;

    public:
        // node: owns this input
        // index: position of this input in all inputs
        // output: supplies the value for this input
        Input(Node *node, size_t index, Output &output);

        // return the node that is this input of
        std::shared_ptr<Node> get_node() const;

        // return the position
        size_t get_index() const { return m_index; }

        // return the connected output
        Output &get_output() { return *m_output; }

        // prepare a interface to modify the link directly
        void set_output(Output &output);

        // return the tensor of the connected output
        std::shared_ptr<runtime::Tensor> get_tensor_ptr();

        // return the shape of tensor of the connected output
        const Shape &get_shape() const;
    };
}  // namespace tinyinfer
