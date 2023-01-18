#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <string>
#include "graph.hpp"

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

bool cmp_degree(std::pair<int,int>a, std::pair<int,int>b){
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
    float alpha_degree;
    float alpha_cluster;
    if (argc == 5) {
        alpha_degree = atof(argv[3]);
        alpha_cluster = atof(argv[4]);
    }
    else if (!g.evenly_distributed) {
        alpha_degree = 0.9;
        alpha_cluster = 0.1;
    } else {
        alpha_degree = 0.01;
        alpha_cluster = 0.99;
    }

    auto clock = std::chrono::high_resolution_clock();
    auto start = clock.now();
    
    // initialize the list of degrees and communities
    std::vector<float> comm_center_list(g.num_comm);
    std::vector<std::pair<int, int>> degree_list(g.num_node); // pair of (degree, index)
    for (int i = 0; i < g.num_node; ++i) {
        degree_list[i] = std::make_pair(g.degree_table[i], i);
    }
    std::vector<std::pair<int,int>> degree_interval(g.num_node); // pair of (min, max) index of each degree

    // initialize the list of node current positions and initial indices
    std::vector<std::pair<float, int>> relabel_list(g.num_node);

    // initialize the degree interval list
    std::sort(degree_list.begin(), degree_list.end(), cmp_degree);
    int cur_degree = 0;
    int start_pos = 0;
    int end_pos = -1;

    for (int i = 0; i < degree_list.size(); ++i)  {
        if (degree_list[i].first != cur_degree) {
            cur_degree = degree_list[i].first;
            start_pos = end_pos + 1;
            end_pos = i;
            for (int j = start_pos; j <= end_pos; ++j) {
                degree_interval[degree_list[j].second].first = start_pos;
                degree_interval[degree_list[j].second].second = end_pos;
            }
        }
    }
    start_pos = end_pos + 1;
    end_pos = degree_list.size() - 1;
    for (int j = start_pos; j <= end_pos; ++j) {
        degree_interval[degree_list[j].second].first = start_pos;
        degree_interval[degree_list[j].second].second = end_pos;
    }

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
            float degree = degree_force(j, degree_interval[relabel_list[j].second].first, degree_interval[relabel_list[j].second].second);
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

    return 0;
}
