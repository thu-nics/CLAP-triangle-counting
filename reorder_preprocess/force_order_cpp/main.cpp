#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <string>
#include "graph.hpp"
#define ALPHA_DEGREE 0.95
#define ALPHA_CLUSTER 0.05

// use the force-directed algorithm to reorder the graph based on the node degree
float degree_force(int index, int min_pos, int max_pos){
    float force = 0;
    if (index < min_pos) {
        force = min_pos - index;
    } else if (index > max_pos) {
        force = max_pos - index;
    }
    return force;
}

// use the force-directed algorithm to reorder the graph based on the node degree
float cluster_force(int index, float center){
    float force = center - (float) index;
    return force;
}

// return true if a < b
bool cmp_relabel(std::pair<float,int>a, std::pair<float,int>b){
    return a.first < b.first;
}

int main(int argc, char** argv) {
    // get the path of the graph
    std::string path = argv[1];
    int iterate_time = atoi(argv[2]);

    // load the graph
    Graph g;
    g.load_revised_COO(path.c_str());

    // set the hyperparameters
    float alpha_degree = ALPHA_DEGREE;
    float alpha_cluster = ALPHA_CLUSTER;

    // initialize the list of degrees and communities
    std::vector<float> comm_center_list(g.num_comm);
    std::vector<std::pair<int, int>> degree_list(g.num_node); // pair of (degree, index)
    for (int i = 0; i < g.num_node; ++i) {
        degree_list[i] = std::make_pair(g.degree_table[i], i);
    }
    std::vector<std::pair<int,int>> degree_interval(g.num_node); // pair of (min, max) index of each degree

    // initialize the list of node current positions and initial indices
    std::vector<std::pair<float, int>> relabel_list(g.num_node);

    auto clock = std::chrono::high_resolution_clock();
    auto start = clock.now();

    // noqa: sort the degree list and return the index of each degree
    std::sort(degree_list.begin(), degree_list.end());
    int cur_degree = 0;
    degree_interval[0] = std::make_pair(0, 0);
    for (int i = 0; i < degree_list.size(); ++i)  {
        if (degree_list[i] != cur_degree) {
            degree_interval[cur_degree].second = i - 1;
            cur_degree = degree_list[i];
            degree_interval[cur_degree] = std::make_pair(i, i);
        }
    }
    degree_interval[cur_degree].second = degree_list.size() - 1;

    // initialize the relabel list with the initial indices
    for (int i = 0; i < g.num_node; ++i) {
        relabel_list[i].first = (float) i;
        relabel_list[i].second = i;
    }

    // iterate the force-directed algorithm
    for (int i = 0; i < iterate_time; ++i) {
        // update the community center list
        for (int j = 0; j < g.num_comm; ++j) {
            comm_center_list[j] = 0;
        }
        for (int j = 0; j < g.num_node; ++j) {
            comm_center_list[g.comm_table[relabel_list[j].second]] += j;
        }
        for (int j = 0; j < g.num_comm; ++j) {
            comm_center_list[j] /= g.comm_list[j].size();
        }

        // update the relabel list position with force
        for (int j = 0; j < g.num_node; ++j) {
            float degree = degree_force(j, degree_interval[g.degree_table[relabel_list[j].second]].first, degree_interval[g.degree_table[relabel_list[j].second]].second);
            float cluster = cluster_force(j, comm_center_list[g.comm_table[relabel_list[j].second]]);
            relabel_list[j].first += alpha_degree * degree + alpha_cluster * cluster;
        }

        // sort the relabel list and update the relabel map, vlog(v) time complexity
        std::sort(relabel_list.begin(), relabel_list.end(), cmp_relabel);
        for (int j = 0; j < g.num_node; ++j) {
            relabel_list[j].first = (float) j;
        }
    }

    // generate the relabel map
    std::vector<int> relabel_map(g.num_node);
    for (int i = 0; i < g.num_node; ++i) {
        relabel_map[relabel_list[i].second] = i;
    }

    auto end = clock.now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s" << std::endl;

    std::cout << relabel_map[0] << std::endl;
    std::cout << relabel_map[1] << std::endl;

    return 0;
}
