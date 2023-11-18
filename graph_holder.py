import random
import networkx as nx
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

class GraphHolder():
    def __init__(self, graph_no=100, node_no=1000, edge_prob=0.05, graph_type='small world'):
        self.graph_no = graph_no
        self.node_no = node_no
        self.graph_type = graph_type
        self.max_edge_no = int((self.node_no * (self.node_no - 1)) / 2)
        self.edge_prob = edge_prob
        self.edge_no = int(self.max_edge_no * self.edge_prob)  # Approximately 0.05 of max possible edge number
        if graph_type == 'small world':
            self.graphs = self.make_small_worlds()
        elif graph_type == 'scale free':
            self.graphs = self.make_scale_free()
        elif graph_type == 'random':
            self.graphs = self.make_random()
        else:
            raise Exception("invalid graph type!")

    def make_small_worlds(self):
        small_worlds = []
        self.seed_values = random.sample(range(1, 100000),
                                         self.graph_no)  # generating GRAPH_NUMBER unique random number to be used as seed
        k = round(((2 * self.edge_no) / self.node_no))
        for i in range(self.graph_no):
            print('Graph No: ', i)
            # we want to have EDGE_NUMBER edge, and in the base graph we have k degree
            # for each node. And we know summation of node degrees, is 2 * EDGE_NUMBER
            # so we have tohave k = (2 * EDGE_NUMBER) / NODE_NUMBER  for each node.
            rewiring_probability = random.uniform(0.2, 0.3)
            graph = nx.watts_strogatz_graph(n=self.node_no, k=k, p=rewiring_probability, seed=self.seed_values[i])
            small_worlds.append(graph)
        return small_worlds

    def make_scale_free(self):
        self.seed_values = random.sample(range(1, 100000),
                                         self.graph_no)  # generating GRAPH_NUMBER unique random number to be used as seed
        m = round(((self.edge_no) / self.node_no))
        scale_frees = []
        for i in range(self.graph_no):
            print('Graph no: ', i)
            graph = nx.barabasi_albert_graph(n=self.node_no, m=m, seed=self.seed_values[i], initial_graph=None)
            scale_frees.append(graph)
        return scale_frees

    def make_random(self):
        self.seed_values = random.sample(range(1, 100000),
                                         self.graph_no)  # generating GRAPH_NUMBER unique random number to be used as seed
        randoms = []
        for i in range(self.graph_no):
            print('graph no: ', i)
            graph = nx.erdos_renyi_graph(self.node_no, self.edge_prob, seed=self.seed_values[i])
            randoms.append(graph)
        return randoms

    def show_degree_distribution(self):
        for i in range(len(self.graphs)):
            graph = self.graphs[i]
            degree_freq_dic = self.one_graph_degree_distribution(graph)
            x_axis = degree_freq_dic.keys()
            y_axis = degree_freq_dic.values()
            y_axis = np.array(list(y_axis)) / graph.number_of_nodes()
            plt.title(f'Degree Distribution For Graph: {self.graph_type} Number: {i}')
            plt.xlabel("Degree")
            plt.ylabel("Frequesncy")
            plt.plot(x_axis, y_axis, label='degree probability')
            upper_y = np.array([0, max(y_axis)])
            avg = np.average([graph.degree(n) for n in graph.nodes()])
            upper_x = np.array([avg, avg])
            plt.plot(upper_x, upper_y, color='red', linestyle='-.', label='mean')
            plt.legend(loc='best') # setting best location for labels
            plt.show()

    def one_graph_degree_distribution(self, graph):
        degrees = [graph.degree(n) for n in graph.nodes()]
        degrees = list(sorted(degrees))
        degree_freq_dic = Counter(degrees)
        return degree_freq_dic