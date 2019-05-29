#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "input.hpp"
#include "output.hpp"

namespace tinyinfer {
class Node;
using NodeVector = std::vector<std::shared_ptr<Node>>;

class Node : public std::enable_shared_from_this<Node> {
  friend class Input;
  friend class Output;

 private:
  const std::string m_node_type;
  size_t m_instance_id;
  const std::string m_unique_name;
  static std::atomic<size_t> m_next_instance_id;

  std::deque<Input> m_inputs;
  std::deque<Output> m_outputs;

 protected:
  Node(const std::string& node_type, const NodeVector& arguments);

 public:
  virtual ~Node();

  const std::string& get_description() const { return m_node_type; }
  size_t get_instance_id() const { return m_instance_id; }
  const std::string& get_name() const { return m_unique_name; }

  // get number of inputs/outpus for the op
  size_t get_input_num() const { return m_inputs.size(); }
  size_t get_output_num() const { return m_outputs.size(); }

  // get nodes of providing input
  std::shared_ptr<Node> get_argument(size_t index);
  NodeVector get_arguments();
  // get nodes of using output
  NodeVector get_users() const;

  // get input/output queue
  std::deque<Input>& get_inputs() { return m_inputs; }
  std::deque<Output>& get_outputs() { return m_outputs; }

  const NodeVector& check_args_single_output(const NodeVector& args);
  void virtual forward() = 0;
};
}  // namespace tinyinfer