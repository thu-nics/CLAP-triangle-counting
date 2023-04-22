import numpy as np

from typing import Any, List, Dict


class self_defined_force_d:
    """
    input: node index, node degree
    """

    def __init__(self, alpha=1.0, degree_list: List = None) -> None:
        self.alpha = alpha
        self.degree_list = degree_list  # sorted list of degrees

    def __call__(self, index, degree) -> Any:
        avg_degree = np.sum(self.degree_list) / len(self.degree_list)  # noqa
        # define the force by yourself, e.g., self.alpha*(degree-self.avg_degree)
        force = self.alpha * (degree - self.avg_degree)  # noqa
        return force


class self_defined_force_c:
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
                ]  # new_cluster is the community this node belongs to
                cluster_center = np.sum(new_cluster) / len(
                    new_cluster
                )  # define the center by yourself, e.g., np.sum(new_cluster)/len(new_cluster)
                comm_center_list[i][comm_id_list[i]] = cluster_center
            force += self.alpha[i] * (
                cluster_center - index
            )  # define the force by yourself, e.g., self.alpha[i]*(cluster_center-index)
        return force
