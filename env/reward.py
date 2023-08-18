from .search_tree import SearchTree


def postorder(graph, root):
    tot = 0
    children = []
    for child in graph.successors(root):
        tot += postorder(graph, child)
        children.append(child)

    graph.nodes[root]['subtree_size'] = tot + 1
    graph.nodes[root]['children'] = children
    return graph.nodes[root]['subtree_size']


class Reward:
    def __init__(self, debug=False):
        self.tree = None
        self.debug_mode = debug

    def before_reset(self, model):
        self.tree = None

    def extract(self, model, done):
        if not done:
            if self.tree is None:
                self.tree = SearchTree(model)
            else:
                self.tree.update_tree(model)
                if self.debug_mode:
                    self.tree.render()
            return None, None

        # instance was pre-solved
        if self.tree is None:
            return []

        node2position = {node: i for i, node in enumerate(self.tree.tree.graph['visited_node_ids'])}

        returns = []
        childrens = []
        for node in self.tree.tree.graph['visited_node_ids']:
            returns.append(-self.tree.tree.nodes[node]['subtree_size'])
    
            # ignore not visited children
            childrens.append([
                node2position[child] for child in self.tree.tree.successors(node) if child in node2position
            ])

        if self.debug_mode:
            print('\nB&B tree:')
            print(f'All nodes saved: {self.tree.tree.nodes()}')
            print(f'Visited nodes: {self.tree.tree.graph["visited_node_ids"]}')
            self.tree.render()
            self.tree.update_tree(model)
            self.tree.render()
            for node in [node for node in self.tree.tree.nodes]:
                if node not in self.tree.tree.graph['visited_node_ids']:
                    self.tree.tree.remove_node(node)
            self.tree.step_idx += 1
            self.tree.render()


        return returns, childrens
