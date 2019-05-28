#pragma once

#include <deque>
#include <vector>

#include "input.hpp"
#include "output.hpp"

namespace tinyinfer
{
    class Node;
    using nodeVector = std::vector<std::shared_ptr<Node>>;
    
    class Node : public std::enable_shared_from_this<Node>
    {
        friend class Input;
        friend class Output;

    protected:
        virtual void validate_and_infer();

        Node(const std::string& node_type, const NodeVector& arguments, size_t output_size = 1);

    public:
        virtual ~Node();

        const std::string& description() const;

        const std::string& get_name() const;

        bool is_same_op_type(const std::shared_ptr<Node>& node) const
        {
            return description() == node->description();
        }

    }
}