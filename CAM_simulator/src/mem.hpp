//
//  mem.hpp
//  TriangleCounting
//
//  
//

#ifndef mem_hpp
#define mem_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>

template<typename T>
struct trace{
    char type; // l for load, s for save
    long long addr;
};

template<class T>
class mem {
private:
    bool track_trace_detail=true;
    std::vector<trace<T>> mem_trace;
    unsigned long long num_read_trace;
    unsigned long long num_write_trace;
public:
    std::vector<unsigned int> offset_addrs;
    void print_offset_info();
    int add_trace(const T* begin, const T* end, long long offset, char type='l');
    int write_file(std::ostream& file, long unsigned int max_num_trace = (long unsigned int)(-1));
    unsigned long long count_trace(char mode='a'); // a:all, l:load, s:save
    long long int max_phisical_addr = 0;
    bool autoswitch_track_detail(const unsigned long int max_runtime_trace);
};

// The elements of a vector are stored contiguously, meaning that if v is a vector where T is some type other than bool, then it obeys the identity &v[n] == &v[0] + n for all 0 <= n < v.size().

#endif /* mem_hpp */
//
//  mem.hpp
//  TriangleCounting
//
//  Created by 傅天予 on 2021/7/25.
//

#ifndef mem_hpp
#define mem_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>

template<typename T>
struct trace{
    char type; // l for load, s for save
    long int addr;
};

template<class T>
class mem {
private:
    bool track_trace_detail=true;
    std::vector<trace<T>> mem_trace;
    unsigned long int num_read_trace;
    unsigned long int num_write_trace;
public:
    std::vector<unsigned int> offset_addrs;
    void print_offset_info();
    int add_trace(const T* begin, const T* end, const T* offset, char type='l');
    int write_file(std::ostream& file, long unsigned int max_num_trace = (long unsigned int)(-1));
    long long count_trace(char mode='a'); // a:all, l:load, s:save
    long long int max_phisical_addr = 0;
    bool autoswitch_track_detail(const unsigned long int max_runtime_trace);
};

// The elements of a vector are stored contiguously, meaning that if v is a vector where T is some type other than bool, then it obeys the identity &v[n] == &v[0] + n for all 0 <= n < v.size().

#endif /* mem_hpp */
