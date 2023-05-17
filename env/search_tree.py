import matplotlib.pyplot as plt
import seaborn as sns
from ordered_set import OrderedSet
from collections import defaultdict
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from ecole.scip import Model


class SearchTree:
    '''
    Tracks SCIP search tree. Call SearchTree.update_tree(ecole.Model) each
    time the ecole environment (and therefore the ecole.Model) is updated.

    N.B. SCIP does not store nodes which were pruned, infeasible, outside
    the search tree's optimality bounds, or which node was optimal, therefore these nodes will not be
    stored in the SearchTree. This is why m.getNTotalNodes() (the total number
    of nodes processed by SCIP) will likely be more than the number of nodes in
    the search tree when an instance is solved.
    '''
    def __init__(self, model: Model):       
        self.tree = nx.DiGraph()
        
        self.tree.graph['root_node'] = None
        self.tree.graph['visited_nodes'] = []
        self.tree.graph['visited_node_ids'] = OrderedSet()
        
        m = model.as_pyscipopt()
        if m.getCurrentNode() is not None:

            self.tree.graph['optimum_nodes'] = [m.getCurrentNode()]
            self.tree.graph['optimum_node_ids'] = OrderedSet([m.getCurrentNode().getNumber()])
            self.init_primal_bound = m.getPrimalbound()
            self.tree.graph['incumbent_primal_bound'] = self.init_primal_bound

            self.tree.graph['fathomed_nodes'] = []
            self.tree.graph['fathomed_node_ids'] = OrderedSet()

            self.prev_primal_bound = None
            self.prev_node_id = None

            self.step_idx = 0

            self.update_tree(model)
            
        else:
            # instance was pre-solved
            pass
            
    def update_tree(self, model: Model):
        '''
        Call this method after each update to the ecole environment. Pass
        the updated ecole.Model, and the B&B tree tracker will be updated accordingly.
        '''
        m = model.as_pyscipopt()
                
        # get current node (i.e. next node to be branched at)
        _curr_node = m.getCurrentNode()
        if _curr_node is not None:
            self.curr_node_id = _curr_node.getNumber()
        else:
            # branching finished, no curr node
            self.curr_node_id = None
        
        if len(self.tree.graph['visited_node_ids']) >= 1:
            self.prev_node_id, self.prev_node = self.tree.graph['visited_node_ids'][-1], self.tree.graph['visited_nodes'][-1]
            
            # check if previous branching at previous node changed global primal bound. If so, set previous node as optimum
            if m.getPrimalbound() < self.tree.graph['incumbent_primal_bound']:
                # branching at previous node led to finding new incumbent solution
                self.tree.graph['optimum_nodes'].append(self.prev_node)
                self.tree.graph['optimum_node_ids'].add(self.prev_node_id)
                self.tree.graph['incumbent_primal_bound'] = m.getPrimalbound()
            
        self.curr_node = {self.curr_node_id: _curr_node}
        if self.curr_node_id is not None:
            if self.curr_node_id not in self.tree.graph['visited_node_ids']:
                self._add_nodes(self.curr_node)
                self.tree.graph['visited_nodes'].append(self.curr_node)
                self.tree.graph['visited_node_ids'].add(self.curr_node_id)
                self.tree.nodes[self.curr_node_id]['step_visited'] = self.step_idx
        
        if self.curr_node_id is not None:
            _parent_node = list(self.curr_node.values())[0].getParent()
            if _parent_node is not None:
                parent_node_id = _parent_node.getNumber()
            else:
                # curr node is root node
                parent_node_id = None
            self.parent_node = {parent_node_id: _parent_node}
        else:
            self.parent_node = {None: None} 
            
        # add open nodes to tree
        open_leaves, open_children, open_siblings = m.getOpenNodes()
        self.open_leaves = {node.getNumber(): node  for node in open_leaves}
        self.open_children = {node.getNumber(): node for node in open_children}
        self.open_siblings = {node.getNumber(): node for node in open_siblings}
        
        self._add_nodes(self.open_leaves)
        self._add_nodes(self.open_children)
        self._add_nodes(self.open_siblings)
        
        # check if previous branching at previous node led to fathoming
        if len(self.tree.graph['visited_node_ids']) > 2 or self.curr_node_id is None:
            if self.curr_node_id is not None:
                # in above code, have added current node to visited node ids, therefore prev node is at idx=-2
                self.prev_node_id, self.prev_node = self.tree.graph['visited_node_ids'][-2], self.tree.graph['visited_nodes'][-2]
            else:
                # branching finished, previous node was fathomed
                self.prev_node_id, self.prev_node = self.tree.graph['visited_node_ids'][-1], self.tree.graph['visited_nodes'][-1]
            if len(list(self.tree.successors(self.prev_node_id))) == 0 and self.prev_node_id != self.curr_node_id:
                # branching at previous node led to fathoming
                self.tree.graph['fathomed_nodes'].append(self.prev_node)
                self.tree.graph['fathomed_node_ids'].add(self.prev_node_id)

        self.step_idx += 1

    def _add_nodes(self, nodes, parent_node_id=None):
        '''Adds nodes if not already in tree.'''
        for node_id, node in nodes.items():
            if node_id not in self.tree:
                # add node
                self.tree.add_node(node_id,
                                   _id=node_id,
                                   lower_bound=node.getLowerbound())

                # add edge
                _parent_node = node.getParent()
                if _parent_node is not None:
                    if parent_node_id is None:
                        parent_node_id = _parent_node.getNumber()
                    else:
                        # parent node id already given
                        pass
                    self.tree.add_edge(parent_node_id,
                                       node_id)
                else:
                    # is root node, has no parent
                    self.tree.graph['root_node'] = {node_id: node}
                    
    def _get_node_groups(self):
        node_groups = defaultdict(lambda: [])
        for node in self.tree.nodes:
            if node not in self.tree.graph['visited_node_ids'] or self.curr_node_id == node:
                node_groups['Unvisited'].append(node)
            else:
                node_groups['Visited'].append(node)
            if node in self.tree.graph['fathomed_node_ids']:
                node_groups['Fathomed'].append(node)
            if node == self.tree.graph['optimum_node_ids'][-1]:
                node_groups['Incumbent'].append(node)
        return node_groups
                                    
    def render(self,
               unvisited_node_colour='#FFFFFF',
               visited_node_colour='#A7C7E7',
               fathomed_node_colour='#FF6961',
               incumbent_node_colour='#C1E1C1',
               next_node_colour='#FFD700',
               node_edge_colour='#000000',
               use_latex_font=False,
               font_scale=0.75,
               context='paper',
               style='ticks'
              ):
        '''Renders B&B search tree.'''
        if use_latex_font:
            sns.set(rc={'text.usetex': True},
                    font='times')
        sns.set_theme(font_scale=font_scale, context=context, style=style)
        
        group_to_colour = {'Unvisited': unvisited_node_colour,
                           'Visited': visited_node_colour,
                           'Fathomed': fathomed_node_colour,
                           'Incumbent': incumbent_node_colour}
        
        f, ax = plt.subplots()
        
        pos = graphviz_layout(self.tree, prog='dot')

        node_groups = self._get_node_groups()
        for group_label, nodes in node_groups.items():
            nx.draw_networkx_nodes(self.tree,
                                   pos,
                                   nodelist=nodes,
                                   node_color=group_to_colour[group_label],
                                   edgecolors=node_edge_colour,
                                   label=group_label)
            
        if self.curr_node_id is not None:
            nx.draw_networkx_nodes(self.tree,
                                   pos,
                                   nodelist=[self.curr_node_id],
                                   node_color=unvisited_node_colour,
                                   edgecolors=next_node_colour,
                                   linewidths=3,
                                   label='Next')
            num_groups = len(list(node_groups.keys())) + 1
        else:
            num_groups = len(list(node_groups.keys()))
    
        nx.draw_networkx_edges(self.tree,
                               pos)
        
        nx.draw_networkx_labels(self.tree, pos, labels={node: node for node in self.tree.nodes})
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=num_groups)
        plt.show()
        # plt.savefig(f'imgs/img_{self.step_idx}.png', dpi=500, bbox_inches='tight')
        # plt.close()
