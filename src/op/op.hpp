#pragma once

#include "node.hpp"
#include "runtime/tensor.hpp"

namespace tinyinfer {
    namespace op {
// TODO: not clear the difference of roles between Op and Node
/**
 * \brief Root of all op nodes
 */
        class Op : public Node {
        public:
            /**
             * Construct the node with input nodes
             * @param arguments: the input nodes
             */
            Op(const std::string &node_type, const NodeVector &arguments)
                    : Node(node_type, arguments) {}

            /**
             * execute the exact computing and write the result into output
             */
            virtual void forward() = 0;
            //
            // currently do nothing
            /**
             * Validating correctness of the node
             */
            virtual void validate_and_infer() { ; }

            /**
             * Return whether a node has the same type
             * @param op: the op node
             */
            bool is_same_op_type(const std::shared_ptr<Op> &op) const {
                return get_description() == op->get_description();
            }
        };
    }
}