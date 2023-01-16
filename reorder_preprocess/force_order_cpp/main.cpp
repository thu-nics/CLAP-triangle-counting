#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <string>
#include "graph.hpp"
#define ALPHA_CONTROL 0.95
#define ALPHA_CLUSTER 0.05

float control_force(int index, int min_pos, int max_pos)
{
    float force = 0;
    if (index < min_pos) {
        force = ALPHA_CONTROL * (min_pos - index);
    } else if (index > max_pos) {
        force = ALPHA_CONTROL * (max_pos - index);
    }
    return force;
}

float cluster_force(int index, float center)
{
    float force = ALPHA_CLUSTER * (center - (float) index);
    return force;
}

bool cmp_relabel(std::pair<float,int>a, std::pair<float,int>b) 
{
    return a.first < b.first;
}
int main(int argc, char** argv) {
    std::string path = argv[1];
    int iterate_time = atoi(argv[2]);
    path = "/home/nfs_data/weichiyue/output/rabbit/" + path + "_rabbit.txt";
    Graph g;
    g.load_revised_COO(path.c_str());
    float alpha_control = 0.95;
    float alpha_cluster = 0.05;
    
    std::vector<std::pair<float, int>> relabel_list;
    std::vector<float> comm_center_list;
    std::vector<int> degree_list(g.degree_table.begin(), g.degree_table.end());
    std::unordered_map<int, std::pair<int,int>> degree_interval;

    auto clock = std::chrono::high_resolution_clock();
    auto start = clock.now();
    
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

    for (int i = 0; i < g.num_node; ++i) {
        relabel_list.push_back(std::make_pair((float) i, i));
    }
    for (int i = 0; i < g.num_comm; ++i) {
        comm_center_list.push_back(0);
    }


    for (int i = 0; i < iterate_time; ++i) {
        for (int j = 0; j < g.num_comm; ++j) {
            comm_center_list[j] = 0;
        }
        for (int j = 0; j < g.num_node; ++j) {
            comm_center_list[g.comm_table[relabel_list[j].second]] += j;
        }
        for (int j = 0; j < g.num_comm; ++j) {
            comm_center_list[j] /= g.comm_list[j].size();
        }
        for (int j = 0; j < g.num_node; ++j) {
            float control = control_force(j, degree_interval[g.degree_table[relabel_list[j].second]].first, degree_interval[g.degree_table[relabel_list[j].second]].second);
            float cluster = cluster_force(j, comm_center_list[g.comm_table[relabel_list[j].second]]);
            relabel_list[j].first += control + cluster;
        }
        std::sort(relabel_list.begin(), relabel_list.end(), cmp_relabel);
        for (int j = 0; j < g.num_node; ++j) {
            relabel_list[j].first = (float) j;
        }
    }
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