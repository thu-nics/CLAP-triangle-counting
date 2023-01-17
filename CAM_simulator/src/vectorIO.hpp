#pragma once
#include <fstream>
#include <vector>
#define NUM_SEP 8
template<class T>
std::vector<T> loadVector(std::istream& file){
  int len;
  file.read((char*)&len,sizeof(int));
  std::vector<T> vec;
  vec.resize(len);

  file.read((char*)vec.data(),len * sizeof(T));
  return vec;
}

template<class T>
void writeVector(std::ostream& file, const std::vector<T>& v){
  int len = v.size();
  file.write((char*)&len, sizeof(int));
  file.write((const char*)v.data(),len * sizeof(T));
}

struct Graph{
    int num_node;
    std::vector<int> CSR_node_pos;
    std::vector<int> CSR_neigh;
    std::vector<int> CSR_mid;
    std::vector<int> CSR_sep[NUM_SEP];

    void load(std::istream& file){
        file.read((char*)&num_node,sizeof(int));
        CSR_node_pos = loadVector<int>(file);
        CSR_neigh = loadVector<int>(file);
        CSR_mid = loadVector<int>(file);
    }
    void write(std::ostream& file){
        file.write((char*)&num_node,sizeof(int));
        writeVector(file,CSR_node_pos);
        writeVector(file,CSR_neigh);
    }
};
