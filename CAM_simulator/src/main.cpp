#define PE_NUM 120
#define CAM_SIZE 512
#define RECORD_TRACE true
#define SORTED_CAM false

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <thread>
#include <random>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <list>

#include "vectorIO.hpp"
#include "mem.hpp"


unsigned long int max_num_trace = 5000000;


bool cmp_pair(const std::pair<int, int> &a, const std::pair<int, int> &b){
    return a.second < b.second;
}


int main(int argc, const char *argv[]){
    
    // prepare
    auto clock = std::chrono::steady_clock();
    
    // args
    if (argc!=4 && argc!=5){
        std::cout<<"argc: "<<argc<<std::endl;
        std::cout << "args: input_file, num_core, repeat_loop, (trace_file)" << std::endl;
    }
    if (argc<4) {
        return -1;
    }

    std::string filename = argv[1];
    int usedCore = std::stoi(argv[2]);
    int repeat = std::stoi(argv[3]);
    
    // record time of load data
    auto loadStart = clock.now();

    // fileIO
    std::ifstream infile(filename,std::ios::binary);
    if(!infile){
        std::cout << "Can't read file : "<< filename << std::endl;
        return -1;
    }
    int num_node;
    std::vector<int> CSR_pos;
    std::vector<int> CSR_neigh;
    {
        Graph g;
        g.load(infile);
        infile.close();
        num_node = g.num_node;
        CSR_pos = std::move(g.CSR_node_pos);
        CSR_neigh = std::move(g.CSR_neigh);
    }

    mem<int> main_mem[PE_NUM];
    long long end_addr, start_addr_pos, start_addr_neigh;
    long long mem_offset_pos, mem_offset_neigh;


    std::cout << "num_node: " << num_node << std::endl;
    std::cout << "total_space: " << CSR_neigh.size()+num_node<< std::endl;
    float prepare_time = (clock.now() - loadStart).count() / 1e9;
    std::cout << "prepare time: " << prepare_time << std::endl;

    //mining triangle
    std::vector<std::pair<int, int>> CAM_instance;
    auto computeStart= clock.now();

    int numCores = usedCore;
    float durtion;
    int cur_node = 0;
    // begin mining loop
    std::list<std::pair<int,int>> workload;
    std::list<std::pair<int,int>> workload_pos;


    end_addr = 0;
    start_addr_pos = end_addr;
    mem_offset_pos = (long long)(&CSR_pos[0]) - start_addr_pos;
    start_addr_neigh = end_addr + (long long)CSR_pos.size() * sizeof(int) + sizeof(int);    
    mem_offset_neigh = (long long)(&CSR_neigh[0]) - start_addr_neigh;
    end_addr = start_addr_neigh + (long long)CSR_neigh.size() * sizeof(int) + sizeof(int);
    
    for (int i_PE = 0; i_PE < PE_NUM; i_PE++){

        main_mem[i_PE].offset_addrs.push_back(start_addr_pos);
    }
    std::cout << "end_addr: " << "0x" << std::setfill('0') << std::setw(16) << std::hex << end_addr << std::endl;
    std::cout << "mem_offset_pos: " << "0x" << std::setfill('0') << std::setw(16) << std::hex << mem_offset_pos << std::endl;
    std::cout << "mem_offset_neigh: " << "0x" << std::setfill('0') << std::setw(16) << std::hex << mem_offset_neigh << std::endl;
    std::cout << "start_addr_pos: " << "0x" << std::setfill('0') << std::setw(16) << std::hex << start_addr_pos << std::endl;
    std::cout << "start_addr_neigh: " << "0x" << std::setfill('0') << std::setw(16) << std::hex << start_addr_neigh << std::endl;

    // std::cout << "&CSR_pos[0]: " << "0x" << std::setfill('0') << std::setw(16) << std::hex << (&CSR_pos[0]) << std::endl;
    // std::cout << "&CSR_pos[0]+1: " << "0x" << std::setfill('0') << std::setw(16) << std::hex << (&CSR_pos[0]+1) << std::endl;
    // compute workload
    
    int real_cam_size = CSR_neigh.size() / PE_NUM;
    if (real_cam_size > CAM_SIZE){
        real_cam_size = CAM_SIZE;
    }
    while(cur_node < num_node){
        int CAM_op = 0;
        int cur_size = 0;
        int begin_pos = cur_node;
        int end_pos = cur_node;
        // std::cout << std::dec <<cur_node << std::endl;
        for (int a = cur_node; a < num_node; a++, end_pos++)
        {
            const int* Na_begin = CSR_neigh.data() + CSR_pos[a];
            const int* Na_end = CSR_neigh.data() + CSR_pos[a + 1];
            if ((Na_end - Na_begin) > real_cam_size ) // skip the node has neigh more than CAM_SIZE
                continue;
            if (cur_size + Na_end - Na_begin > real_cam_size)
                break;
            cur_size += Na_end - Na_begin;
        }
        cur_node = end_pos;
        for (int a = begin_pos; a < end_pos; a++)
        {
            const int* Na_begin = CSR_neigh.data() + CSR_pos[a];
            const int* Na_end = CSR_neigh.data() + CSR_pos[a + 1];
            for (auto b_iter = Na_begin; b_iter != Na_end; b_iter++)
            {
                int b = *b_iter;
                const int* Nb_begin = CSR_neigh.data() + CSR_pos[b];
                const int* Nb_end = CSR_neigh.data() + CSR_pos[b + 1];
                CAM_op += Nb_end - Nb_begin;
             
            }
        }
        workload.push_back(std::make_pair(cur_size, CAM_op));
        workload_pos.push_back(std::make_pair(begin_pos, end_pos));

    }

    std::cout<< "compute done"<<std::endl;
    
    // assign workload
    int state[PE_NUM] = {0};
    int work_time[PE_NUM] = {0};
    int operation[PE_NUM] = {0};
    std::vector<std::vector<std::pair<int,int>>> work_pos_list(PE_NUM);
    while(!workload.empty())
    {
        for (int i_PE = 0; i_PE < PE_NUM; i_PE++)
        {
            if (state[i_PE] == 0 && !workload.empty())
            {
                state[i_PE] = workload.front().first*2 + workload.front().second;
                work_time[i_PE] += workload.front().first*2 + workload.front().second;
                operation[i_PE] += workload.front().second;
                work_pos_list[i_PE].push_back(workload_pos.front());
                workload.pop_front();
                workload_pos.pop_front();

            }
            if (state[i_PE] > 0)
            {
                state[i_PE]--;
            }
        }
    }

    std::cout<< "assign done"<<std::endl;
    int max_work_time = 0;
    int max_operation = 0;
    int max_PE = 0;
    long long total_CAM_op = 0;
    for (int i_PE = 0; i_PE < PE_NUM; i_PE++)
    {
        total_CAM_op += operation[i_PE];
        if (work_time[i_PE] > max_work_time)
        {
            max_work_time = work_time[i_PE];
            max_operation = operation[i_PE];
            max_PE = i_PE;
        }
    }
    long long num_pattern = 0;

    std::cout << "CAM_trace" << std::endl;
    for (int i_PE = 0; i_PE < PE_NUM; i_PE++)
    {
        for (int i = 0; i < work_pos_list[i_PE].size(); i++)
        {
            int begin_pos = work_pos_list[i_PE][i].first;
            int end_pos = work_pos_list[i_PE][i].second;
            const int* begin = CSR_neigh.data() + CSR_pos[begin_pos];
            const int* end = CSR_neigh.data() + CSR_pos[end_pos];

            main_mem[i_PE].autoswitch_track_detail(max_num_trace);
            main_mem[i_PE].add_trace(&CSR_pos[0]+begin_pos, &CSR_pos[0]+begin_pos+1, mem_offset_pos, 'l');
            main_mem[i_PE].add_trace(&CSR_pos[0]+end_pos, &CSR_pos[0]+end_pos+1, mem_offset_pos, 'l');
            main_mem[i_PE].add_trace(begin, end, mem_offset_neigh, 'l');

            CAM_instance.clear();            
            for (int a = begin_pos; a < end_pos; a++)
            {
                const int* Na_begin = CSR_neigh.data() + CSR_pos[a];
                const int* Na_end = CSR_neigh.data() + CSR_pos[a + 1];               
                for (auto b_iter = Na_begin; b_iter != Na_end; b_iter++)
                {
                    int b = *b_iter;
                    CAM_instance.push_back(std::make_pair(a, b));
                }                    
                
            }
#if SORTED_CAM
            std::sort(CAM_instance.begin(), CAM_instance.end(), cmp_pair);
#endif
            for (int i = 0; i < CAM_instance.size(); i++)
            {
                int a = CAM_instance[i].first;
                int b = CAM_instance[i].second;
                const int* Nb_begin = CSR_neigh.data() + CSR_pos[b];
                const int* Nb_end = CSR_neigh.data() + CSR_pos[b + 1];
                main_mem[i_PE].autoswitch_track_detail(max_num_trace);
                main_mem[i_PE].add_trace(&CSR_pos[0]+b, &CSR_pos[0]+b+2, mem_offset_pos, 'l');
                main_mem[i_PE].add_trace(Nb_begin, Nb_end, mem_offset_neigh, 'l');
                for (auto c_iter = Nb_begin; c_iter != Nb_end; c_iter++)
                {
                    int c = *c_iter;
                    if (std::find(CAM_instance.begin(), CAM_instance.end(), std::make_pair(a, c)) != CAM_instance.end())
                    {
                        num_pattern++;
                    }
                }
            }
        }
    }

    long long total_trace_num = 0;
    unsigned long long max_trace_num = main_mem[max_PE].count_trace('a');
    if (argc == 4) {
            printf("Do not write to file\n");
        }
        else {

            for (int i_PE = 0; i_PE < PE_NUM; i_PE++)
            {
                total_trace_num += main_mem[i_PE].count_trace('a');
                std::ofstream outfile;
                std::string const& filename = std::string(argv[4]) + std::to_string(i_PE) + std::string(".trace");
                char const* name = filename.c_str();
                outfile.open(name, std::ios::out);
                if (!outfile.is_open()) {
                    printf("Can't write file %s%d\n", argv[4], i_PE);
                }
                else {
                    printf("Write to output_file %s%d\n", argv[4], i_PE);
                    main_mem[i_PE].write_file(outfile, max_num_trace);                 
                }
            }
            

        }
    

    durtion = (clock.now() - computeStart).count() / 1e9;
    std::cout << "num_triangle: " << num_pattern << std::endl;
    std::cout << "time: " << durtion << std::endl;
    std::cout << "max_trace_num: " << std::dec << max_trace_num << std::endl;
    std::cout << "max_work_time: " << max_work_time << std::endl;
    std::cout << "max_operation: " << max_operation << std::endl;
    std::cout << "max_PE: " << std::dec << max_PE << std::endl;
    std::cout << "total_cam_op: " << std::dec << total_CAM_op << std::endl;
    std::cout << "total_trace_num: " << std::dec << total_trace_num << std::endl;
    return 0;
}
