#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>
#include <set>
#define THRESHOLD 1.2

class Graph {
    public:
    int num_node;
    int num_edge;
    int num_comm;
    bool evenly_distributed;
    std::vector<int> CSR_node_pos;
    std::vector<int> CSR_neigh;
    std::vector<int> comm_table;
    std::vector<int> degree_table;
    std::unordered_map<int, std::set<int>> comm_list;

    Graph()
    {
        num_node = 0;
        num_edge = 0;
        num_comm = 0;
        CSR_node_pos.clear();
        CSR_neigh.clear();
        comm_table.clear();
        degree_table.clear();
        comm_list.clear();
    }

    ~Graph()
    {
        CSR_node_pos.clear();
        CSR_neigh.clear();
        comm_table.clear();
        degree_table.clear();
        comm_list.clear();
    }

    static bool cmp_pair(std::pair<int,int>a, std::pair<int,int>b) {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
    }

    bool load_revised_COO(const char* path)
    {
        if (freopen(path, "r", stdin) == NULL)
        {
            printf("File not found. %s\n", path);
            return false;
        }

        char buf;
        scanf("%c%d%d",&buf, &num_node, &num_edge);
        CSR_node_pos.resize(num_node + 1);
        CSR_neigh.resize(num_edge*2);
        comm_table.resize(num_node);
        degree_table.resize(num_node);
        std::unordered_map<int, int> comm_map;
        int x, y, comm_x, comm_y;
        int cur_comm = 0;
        std::vector<std::pair<int, int>> edge_buffer;
        while(scanf("%d%d%d%d",&x,&y,&comm_x,&comm_y) != EOF)
        {
            edge_buffer.push_back(std::make_pair(x, y));
            edge_buffer.push_back(std::make_pair(y, x));
            if (comm_map.find(comm_x) == comm_map.end())
                comm_map[comm_x] = cur_comm++;
            if (comm_map.find(comm_y) == comm_map.end())
                comm_map[comm_y] = cur_comm++;
            comm_table[x] = comm_map[comm_x];
            comm_list[comm_map[comm_x]].insert(x);
            comm_table[y] = comm_map[comm_y];
            comm_list[comm_map[comm_y]].insert(y);
        }
        std::sort(edge_buffer.begin(), edge_buffer.end(), cmp_pair);
        int cur_node = 0;
        CSR_node_pos[0] = 0;
        for (int i = 0; i < edge_buffer.size(); ++i)
        {
            if (edge_buffer[i].first != cur_node)
            {
                CSR_node_pos[++cur_node] = i;
                degree_table[cur_node - 1] = i - CSR_node_pos[cur_node - 1];
            }
            CSR_neigh[i] = edge_buffer[i].second;
        }
        CSR_node_pos[cur_node + 1] = edge_buffer.size();
        degree_table[cur_node] = edge_buffer.size() - CSR_node_pos[cur_node];
        num_comm = cur_comm;
        
        // calculate variance of degree
        double mean = 0;
        for (int i = 0; i < num_node; ++i)
            mean += degree_table[i];
        mean /= num_node;
        double var = 0;
        for (int i = 0; i < num_node; ++i)
            var += (degree_table[i] - mean) * (degree_table[i] - mean);
        if (var / mean >= THRESHOLD)
            evenly_distributed = false;
        else
            evenly_distributed = true;
        return true;
    }
};
