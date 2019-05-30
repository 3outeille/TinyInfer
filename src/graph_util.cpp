#include <deque>
#include <unordered_map>

#include "graph_util.hpp"

using namespace tinyinfer;

NodeVector tinyinfer::topological_sort(const NodeVector& nodes) {
  std::deque<Node*> indep_nodes;
  std::unordered_map<const Node*, std::shared_ptr<Node>> node_map;
  std::unordered_map<const Node*, size_t> node_dep_count;

  for (auto node : nodes) {
    node_map[node.get()] = node;
    size_t dep_count = node->get_input_num();
    node_dep_count[node.get()] = dep_count;

    if (dep_count == 0) {
      indep_nodes.push_back(node.get());
    }
  }

  NodeVector result;
  while (indep_nodes.size() > 0) {
    auto indep_node = indep_nodes.front();
    result.push_back(node_map[indep_node]);
    indep_nodes.pop_front();

    for (auto user : indep_node->get_users()) {
      if (--node_dep_count[user.get()] == 0) {
        indep_nodes.push_back(user.get());
      }
    }
  }

  assert(result.size() == nodes.size());
  return result;
}
