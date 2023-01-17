//
//  mem.cpp
//  TriangleCounting
//
//
#include "mem.hpp"
#define CACHELINE_SIZE 4

template<class T>
int mem<T>::add_trace(const T* begin, const T* end, long long offset, char type){
    int stride = CACHELINE_SIZE / sizeof(T);

    int track_cnt = 0;
    if (track_trace_detail){
        trace<T> this_trace;
        for (const T* i=begin; i!=end; i++) {
        if (track_cnt == 0){
            this_trace.type = type;
            long long cur_addr = (long long)i;
            assert(cur_addr-offset>=0);
            this_trace.addr = (long long)(cur_addr-offset);
            mem_trace.push_back(this_trace);
        }
        track_cnt = (track_cnt+1)%stride;
        }
    } else {
        if (type == 'l'){
            num_read_trace += (end-begin+1)/stride;
        } else {
            num_write_trace += (end-begin+1)/stride;
        }
    }
    return 0;
};

template<class T>
int mem<T>::write_file(std::ostream &file, long unsigned int max_num_trace){
    const long unsigned int endpos = mem_trace.size();
    for (long unsigned int i=0; i<endpos; ++i) {
        file << mem_trace[i].type <<" "<< "0x" << std::setfill('0') << std::setw(8) << std::hex << mem_trace[i].addr << '\n';
    }
    std::cout<<"write "<<std::dec<<endpos<<" trace"<<std::endl;
    return 0;
}

template<class T>
unsigned long long mem<T>::count_trace(char mode){
    unsigned long long count=0;
    if (track_trace_detail){
        switch (mode) {
            case 'a':
                count = mem_trace.size();
                break;
            case 'l':
                for (auto it=mem_trace.begin(); it!=mem_trace.end(); ++it) {
                    count = count + (it->type=='l');
                }
                break;
            case 's':
                for (auto it=mem_trace.begin(); it!=mem_trace.end(); ++it) {
                    count = count + (it->type=='s');
                }
                break;
            default:
                std::cout<<"usage: mode = {a:all, l:load, s:save}"<<std::endl;
                count = -1;
                break;
        }
    } else {
        switch (mode) {
            case 'a':
                count = num_read_trace + num_write_trace;
                break;
            case 'l':
                count = num_read_trace;
                break;
            case 's':
                count = num_write_trace;
                break;
            default:
                std::cout<<"usage: mode = {a:all, l:load, s:save}"<<std::endl;
                count = -1;
                break;
        }
    }
    return count;
}
template<class T>
void mem<T>::print_offset_info(){
    // the smallest element as offset_init
    unsigned int offset_init = offset_addrs[0];
    std::cout<<"********** offset_info **********"<<std::endl;
    for (auto it = offset_addrs.begin(); it!=offset_addrs.end() ; ++it) {
        std::cout << "0x" << std::setfill('0') << std::setw(8) << std::hex << (*it - offset_init) << std::endl;
    }
    std::cout<<"max_phisical_addr: "<< "0x" << std::setfill('0') << std::setw(8) << std::hex<<max_phisical_addr<<std::endl;
    std::cout<<"*********************************"<<std::endl;
    std::cout<<std::dec;
}

template<class T>
bool mem<T>::autoswitch_track_detail(const unsigned long int max_runtime_trace){
    if (track_trace_detail){
        if (mem_trace.size() >= max_runtime_trace){
            num_read_trace = count_trace('l');
            num_write_trace = count_trace('a') - num_read_trace;
            std::cout<<"SWITCH to simple track at trace: "<<count_trace('a')<<std::endl;
            track_trace_detail = false;
        }
    }
    return track_trace_detail ;
}

// explicit instantiations
// int
template class mem<int>;
// template class trace<int>;
