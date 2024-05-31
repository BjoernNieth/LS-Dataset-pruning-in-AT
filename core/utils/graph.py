from collections import defaultdict


class Graph(object):
    """ Undirected graph datastructure with unweighted edges"""

    def __init__(self):
        self._graph = defaultdict(set)

    def add_connection(self, node1, node2):
        """Add an undirected edge between two nodes"""
        self._graph[node1].add(node2)
        self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        # Loop over all sets of edges in the graph
        for _, edges in self._graph.items():
            # Try to remove the node from the set
            try:
                edges.remove(node)
            except KeyError:
                pass
        # Try the remove the node from the graph
        try:
            del self._graph[node]
        except KeyError:
            pass

    def remove_all_edges(self):
        """Function to remove the minimum amount of nodes to delete all edges"""
        removed_keys = []
        # Check if graph contains more than one node
        if len(self._graph.keys()) >= 1:
            # Search for node with the highest degree
            max_key = max(self._graph, key=lambda a: len(self._graph[a]))
            # Loop until node with highest degree has degree == 0
            while len(self._graph[max_key]) != 0:
                # Remove node with highest degree and edges containing it
                removed_keys.append(max_key)
                self.remove(max_key)
                # Search node with the highest degree in new graph
                max_key = max(self._graph, key=lambda a: len(self._graph[a]))

        # Return all the nodes removed
        return removed_keys


    def get_nodes(self):
        return self._graph.keys()

    def get_graph(self):
        return self._graph

    def get_subgraphs_nodes(self):
        """Get for all graph components a list with their list"""
        # List containing sets of all subgraphs
        sub_graphs = []
        # Loop over all nodes with their edges in the graph
        for p, edges in self._graph.items():
            # Check if node is already contained in a sub graph
            if not any([ p in i for i in sub_graphs]):
                # Make a new subgraph as a set
                sub_graph = {p}
                # Add all neighbours of p to a queue for expansion
                next_nodes = list(edges)
                sub_graph.update(next_nodes)
                # Loop until the queue next_nodes is empty
                while len(next_nodes) != 0:
                    # Get the first element of the queue
                    next_node = next_nodes.pop(0)
                    sub_graph.add(next_node)

                    # Expand the current node
                    expand = list(self._graph[next_node])
                    for node in expand:
                        # Add all nodes to the queue and subgraph which are not contained in the subgraph yet
                        if node not in sub_graph:
                            sub_graph.add(node)
                            next_nodes.append(node)

                # After fully expanding the subgraph add it to the list of subgraphs
                sub_graphs.append(sub_graph)

        return sub_graphs
