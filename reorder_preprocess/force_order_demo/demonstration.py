import time
import argparse
import networkx as nx
import random
import numpy as np
from typing import Any, Optional, Tuple, List, Set, Dict, Union
import matplotlib.pyplot as plt
from matplotlib import animation

import os
from force import self_defined_force_d, self_defined_force_c


def WriteEdgeList(
    filename: Union[str, Any],
    Edge_list: List[Union[List, Tuple]],
    first_line: str = None,
) -> None:
    """
    write list of edges to file. e.g.
    (optional) first_line
    0 1
    0 2
    1 2
    """
    num_E = len(Edge_list)
    print("GraphFileIO: write edge of the whole graph is", num_E)

    with open(filename, "w") as f:
        if first_line:
            f.write(first_line)
            if first_line[-1] != "\n":
                f.write("\n")
        for edge in Edge_list:
            f.write(str(edge[0]) + " " + str(edge[1]) + "\n")
    f.close()


def el2nx(edge_list: list, directed=False) -> nx.Graph:
    """
    convert edge_list to networkx graphs
    """
    graph = nx.Graph(directed=directed)
    graph.add_edges_from(edge_list)
    return graph


class DynamicAdjMatrix:
    def __init__(self, ax, size) -> None:
        self.ax = ax
        self.edge_lists = []
        self.size = size
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        plt.gca().invert_yaxis()

    def __call__(self, i):
        self.scat.set_offsets(self.edge_lists[i])
        self.ax.set_title("iteration: " + str(i))
        return (self.scat,)

    def __len__(self):
        return len(self.edge_lists)

    def init_graph(self):
        # self.ax.invert_yaxis()
        self.scat = self.ax.scatter([], [], s=1)
        return (self.scat,)

    def add_graph(self, graph: Union[nx.Graph, nx.DiGraph]):
        if not graph.is_directed():
            graph = graph.to_directed()
        edges = np.array(graph.edges).reshape(-1, 2)
        self.edge_lists.append(edges)


class degree_interval_force:
    def __init__(self, alpha=1.0, interval_dict: Dict = None) -> None:
        self.alpha = alpha
        self.interval_dict = interval_dict

    def __call__(self, index, degree) -> Any:
        distance = 0
        if index >= self.interval_dict[degree][1]:
            distance = self.interval_dict[degree][1] - index - 1
        elif index < self.interval_dict[degree][0]:
            distance = self.interval_dict[degree][0] - index
        force = self.alpha * distance
        return force


class cluster_force:
    def __init__(self, alpha: np.array, layer=1) -> None:
        self.alpha = alpha
        self.layer = layer

    def __call__(
        self,
        index,
        comm_id_list: List,
        comm_list_list: List = None,
        comm_center_list: List = None,
        node_mapping: Dict = None,
    ) -> Any:
        force = 0
        for i in range(self.layer):
            if comm_center_list[i][comm_id_list[i]] != -1:
                cluster_center = comm_center_list[i][comm_id_list[i]]

            else:
                new_cluster = [
                    node_mapping[node] for node in comm_list_list[i][comm_id_list[i]]
                ]
                cluster_center = np.sum(new_cluster) / len(new_cluster)
                comm_center_list[i][comm_id_list[i]] = cluster_center
            force += self.alpha[i] * (cluster_center - index)
        return force


# class self_defined_force_d:
#     def __init__(self, alpha = 1.0, degree_list: List = None) -> None:
#         self.alpha = alpha
#         self.degree_list = degree_list # sorted list of degrees
#     def __call__(self, index, degree) -> Any:
#         avg_degree = np.sum(self.degree_list)/len(self.degree_list)
#         force = 0 # define the force by yourself, e.g., self.alpha*(degree-self.avg_degree)
#         return force

# class self_defined_force_c:
#     def __init__(self, alpha:np.array, layer = 1) -> None:
#         self.alpha = alpha
#         self.layer = layer
#     def __call__(self, index, comm_id_list: List, comm_list_list: List = None, comm_center_list: List = None, node_mapping: Dict = None) -> Any:
#         force = 0
#         for i in range(self.layer):
#             if comm_center_list[i][comm_id_list[i]] != -1:
#                 cluster_center = comm_center_list[i][comm_id_list[i]]

#             else:
#                 new_cluster = [node_mapping[node] for node in comm_list_list[i][comm_id_list[i]]] # new_cluster is the community this node belongs to
#                 cluster_center = 0 # define the center by yourself, e.g., np.sum(new_cluster)/len(new_cluster)
#                 comm_center_list[i][comm_id_list[i]] = cluster_center
#             force += 0 # define the force by yourself, e.g., self.alpha[i]*(cluster_center-index)
#         return force


class degree_difference_force:
    def __init__(self, alpha=1.0, degree_list: List = None) -> None:
        self.alpha = alpha
        self.degree_list = degree_list

    def __call__(self, index, degree) -> Any:
        force = (
            2
            * self.alpha
            * (len(self.degree_list) - 1)
            * ((self.degree_list[index] - degree))
            / ((self.degree_list[-1] - self.degree_list[0]) ** 1.5)
        )
        return force


class degree_difference_force_linear:
    def __init__(self, alpha=1.0, degree_list: List = None) -> None:
        self.alpha = alpha
        self.degree_list = degree_list

    def __call__(self, index, degree) -> Any:
        force = (
            self.alpha
            * (len(self.degree_list) - 1)
            * (degree - self.degree_list[index])
            / (self.degree_list[-1] - self.degree_list[0])
        )
        return force


class naive_degree_force:
    def __init__(self, alpha=1.0, avg_degree: int = 10) -> None:
        self.alpha = alpha
        self.avg_degree = avg_degree

    def __call__(self, index, degree) -> Any:
        force = self.alpha * (degree - self.avg_degree)
        return force


def random_reorder(graph: nx.Graph, seed=0):
    random.seed(seed)
    random_seq = [(v, random.random()) for v in graph.nodes]
    random_seq.sort(key=lambda x: x[1], reverse=True)
    node_mapping = {seq[0]: i for i, seq in enumerate(random_seq)}
    graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    return node_mapping, graph


def iwd_reorder(graph: nx.Graph, seed=0):
    degree_seq = [(v, graph.degree(v)) for v in graph.nodes]
    degree_seq.sort(key=lambda x: x[1], reverse=False)
    node_mapping = {seq[0]: i for i, seq in enumerate(degree_seq)}
    graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    return node_mapping, graph


def reverse_reorder(graph: nx.Graph, seed=0):
    node_num = len(graph.nodes)
    node_mapping = {i: node_num - 1 - i for i in range(node_num)}
    graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    return node_mapping, graph


def ReadCommDict(file_name: str) -> Dict:
    comm_dict = dict()
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            node_id, comm_id = line.split()
            comm_dict[int(node_id)] = int(comm_id)
    return comm_dict


def community_preprocessing(comm_dict: Dict = None, avg_degree: float = 10):
    """
    comm_dict: key: node index value: the community index this node belongs to
    return
    comm_mapping: mapping used to convert unconsecutive community index into consecutive number
    comm_list: each element of this list is a community containing all node in this community
    """
    comm_set = set()
    comm_mapping = dict()
    comm_list = list()

    for node in comm_dict:
        if comm_dict[node] not in comm_set:
            comm_mapping[comm_dict[node]] = len(comm_set)
            comm_set.add(comm_dict[node])
            comm_list.append([])
        comm_list[comm_mapping[comm_dict[node]]].append(node)
    return comm_mapping, comm_list


def ReadEdgeFile_comm(
    file: str, edge_list: List = None, node_set: Set = None, comm_dict: Dict = None
) -> Optional[Tuple[List, Set, Dict]]:
    """
    Read edge list and sort in ascending order.
    If edge_list and node_set is given, increamental add is used with inplace replacement.
    If not, return new edge_list and node_set
    """
    with open(file, "r") as f:
        rawlines = f.readlines()
    f.close()

    if (edge_list is None) and (node_set is None):
        incremental = False
        edge_list = list()
        node_set = set()
        comm_dict = dict()
    elif (edge_list is not None) and (node_set is not None):
        incremental = True
    else:
        print(
            "GraphFileIO: Either both edge_list and node_set are not given or both are given"
        )
        raise NotImplementedError

    for line in rawlines:
        if not line.startswith("#"):
            splitted = line.strip("\n").split()

            from_node = int(splitted[0])
            to_node = int(splitted[1])
            from_comm = int(splitted[2])
            to_comm = int(splitted[3])

            node_set.add(from_node)
            node_set.add(to_node)

            comm_dict[from_node] = from_comm
            comm_dict[to_node] = to_comm

            edge_list.append([from_node, to_node])

    # sort the edges
    edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))

    print("GraphFileIO: edge_num of the whole graph is", len(edge_list))
    print("GraphFileIO: node_num of the graph is", len(node_set))

    if not incremental:
        return (edge_list, node_set, comm_dict)


def main(args, graph_name, order):
    edge_list = list()
    node_set = set()
    layer = args.layer
    comm_dict_list = [dict() for i in range(layer)]

    cluster_type = args.cluster_type

    if cluster_type == "center":
        ReadEdgeFile_comm(args.input_file, edge_list, node_set, comm_dict_list[0])
        graph = el2nx(edge_list, False)
    else:
        print("GraphFileIO: Wrong cluster type")
        raise NotImplementedError

    # relabel node from rabbit to random
    init = args.init
    if init == "random":
        node_mapping, graph = random_reorder(graph)
    else:
        node_mapping = {node: node for node in graph.nodes}
    # node_mapping = {node:node for node in graph.nodes}

    new_comm_dict_list = [dict() for i in range(layer)]
    for i in range(layer):
        for node in comm_dict_list[i]:
            new_comm_dict_list[i][node_mapping[node]] = comm_dict_list[i][node]

    avg_degree = np.mean([graph.degree(v) for v in graph.nodes])
    # return a list: each element is a community containing all node in this community
    comm_mapping_list = [dict() for i in range(layer)]
    comm_list_list = [list() for i in range(layer)]
    for i in range(layer):
        comm_mapping_list[i], comm_list_list[i] = community_preprocessing(
            new_comm_dict_list[i], avg_degree
        )

    # set alpha
    alpha_control = 0.5
    alpha_cluster = np.array([0.5])
    # prepare work
    degree_list = [graph.degree(v) for v in range(len(graph.nodes))]
    degree_list.sort(reverse=True)

    interval_dict = dict()
    cur_degree = degree_list[0]
    start_pos = 0
    for i in range(len(degree_list)):
        if degree_list[i] != cur_degree:
            end_pos = i
            interval_dict[cur_degree] = [start_pos, end_pos]
            start_pos = i
            cur_degree = degree_list[i]
    interval_dict[cur_degree] = [start_pos, len(degree_list)]

    control_type = args.control_type
    if control_type == "diff":
        f_control = degree_difference_force(
            alpha=alpha_control, degree_list=degree_list
        )
    elif control_type == "interval":
        f_control = degree_interval_force(
            alpha=alpha_control, interval_dict=interval_dict
        )
    elif control_type == "linear":
        f_control = degree_difference_force_linear(
            alpha=alpha_control, degree_list=degree_list
        )
    elif control_type == "naive":
        f_control = naive_degree_force(alpha=alpha_control, avg_degree=avg_degree)
    elif control_type == "self_defined":
        f_control = self_defined_force_d(alpha=alpha_control, degree_list=degree_list)

    if cluster_type == "center":
        f_cluster = cluster_force(alpha=alpha_cluster, layer=layer)
    elif cluster_type == "self_defined":
        f_cluster = self_defined_force_c(alpha=alpha_cluster, layer=layer)

    node_mapping = {node: node for node in graph.nodes}
    relabel_list = [
        [node, node, node, graph.degree(node)] for node in graph.nodes
    ]  # 0:init id 1:relabled id 2:float num 3:community of this node 4:degree
    for node_tuple in relabel_list:
        comm_id_list = [
            comm_mapping_list[i][new_comm_dict_list[i][node_tuple[0]]]
            for i in range(layer)
        ]
        node_tuple.extend(comm_id_list)
    relabel_list.sort(key=lambda x: x[1])

    # draw dynamic graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    dma = DynamicAdjMatrix(ax, size=len(graph))
    # copy graph to tmp_graph
    tmp_graph = nx.Graph()
    tmp_graph.add_nodes_from(graph.nodes)
    tmp_graph.add_edges_from(graph.edges)
    dma.add_graph(tmp_graph)
    # start relabel

    iterate_count = 0
    iterate_time = 10

    for node in graph:
        graph.nodes[node]["id"] = node

    # record time
    mode = args.mode
    start_time = time.time()
    if mode == "simutaneous":
        while iterate_count < iterate_time:
            comm_center_list = [
                [-1 for i in range(len(comm_list_list[j]))] for j in range(layer)
            ]
            for node_tuple in relabel_list:
                F_control = f_control(index=node_tuple[1], degree=node_tuple[3])
                F_cluster = f_cluster(
                    index=node_tuple[1],
                    comm_id_list=node_tuple[4 : 4 + layer],
                    comm_list_list=comm_list_list,
                    comm_center_list=comm_center_list,
                    node_mapping=node_mapping,
                )
                node_tuple[2] += F_cluster + F_control
            relabel_list.sort(key=lambda x: x[2])
            for i in range(len(relabel_list)):
                relabel_list[i][1] = i
                relabel_list[i][2] = i
                node_mapping[relabel_list[i][0]] = i
                graph.nodes[relabel_list[i][0]]["id"] = i

            tmp_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
            dma.add_graph(tmp_graph)
            iterate_count += 1

    end_time = time.time()
    print("time:", end_time - start_time)
    graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    anim = animation.FuncAnimation(
        fig,
        dma,
        frames=len(dma),
        blit=True,
        repeat_delay=1000,
        init_func=dma.init_graph,
    )
    # get current time
    now = time.localtime()
    current_time = time.strftime("%H_%M_%S", now)
    anim.save(
        "dynamic_graph_" + str(current_time) + ".gif", writer="imagemagick", fps=1
    )
    plt.show()
    plt.close()

    if args.output_format == "edge_list":
        edge_list = [(e[0], e[1]) for e in graph.edges]
        edge_list.sort()
        num_node = len(graph.nodes)
        num_edge = len(graph.edges)
        WriteEdgeList(
            args.output_file,
            edge_list,
            first_line="#" + str(num_node) + " " + str(num_edge),
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="force model")

    parser.add_argument(
        "--input_file", type=str, help="name of the input edge_list file"
    )
    parser.add_argument("--output_file", type=str, help="name of the output dict file")
    parser.add_argument(
        "--output_format",
        type=str,
        help="the output format of graph, chose from 'csr' and 'edge_list'",
    )
    parser.add_argument("--control_type", type=str)
    parser.add_argument("--cluster_type", type=str)
    parser.add_argument("--init", type=str)
    parser.add_argument("--layer", type=int)
    graph_name = "citeseer"

    parser.set_defaults(
        input_file="/home/weichiyue/my_edge_list/natural/" + graph_name + ".txt",
        output_file="/home/weichiyue/" + graph_name + ".txt",
        output_format="edge_list",
        init="random",
        control_type="self_defined",
        cluster_type="self_defined",
        layer=1,
    )

    args = parser.parse_args()
    # args = parser.parse_args([]) # for debug

    def set_args(args, graph, order, format):
        graph_name = graph
        args.output_format = format
        args.init = "random"
        args.control_type = "interval"
        args.cluster_type = "center"
        args.mode = "simutaneous"
        args.layer = 1

        if format == "edge_list":
            args.output_file = "./" + graph_name + "_output.txt"
        else:
            raise NotImplementedError

        args.input_file = graph_name + "_rabbit.txt"
        # make output file if not exist
        if not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file))

    graph_list = ["citeseer"]
    order = "test"
    format = "edge_list"

    for graph_name in graph_list:
        print(graph_name)
        set_args(args, graph_name, order, format)
        main(args, graph_name, order)
