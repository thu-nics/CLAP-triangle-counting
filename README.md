# CLAP-triangle-counting

This is the official software implementation of the paper "CLAP: Locality Aware and Parallel Triangle Counting with Content Addressable Memory".

## Overview

![pic1](https://github.com/thu-nics/CLAP-triangle-counting/blob/main/figure/slides1.gif)

![pic2](https://github.com/thu-nics/CLAP-triangle-counting/blob/main/figure/slides2.gif)

![pic3](https://github.com/thu-nics/CLAP-triangle-counting/blob/main/figure/slides3.png)

## Code structure

`CAM_simulator` contains a simulator of our CAM-based triangle counting architecture implemented with C++, it performs the triangle counting task on a given target graph and generates the memory trace at the same time.  

`reorder_preprocess` contains the the force-based node reorder method that reorders the graph indices to enable better runtime performance. It is implemented with Python/C++.

`cache_sim` contains the cache simulator we used to emulate the behavior of cache, it takes the memory trace generated by CAM_simulator and output DRAM request.

`data` contains the example datesets to run the code.  

## Requirement

- Python >= 3.9
- CMake >= 3.10
- C++ standard 14

## Setup

Clone this repository with submodule:  
```bash
git clone --recursive https://github.com/dubcyfor3/CLAP-triangle-counting.git
git submodule init
git submodule update
```

`conda` is recommended for python environment  
```bash
conda create -n CLAP python=3.9 pip
conda activate CLAP
```

Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

Take the dataset citeseer as an example:  
First, generate rabbit order (a cluster-based order) and get the cluster information
```bash
cd reorder_preprocess/rabbit_order
sh rabbit_run.sh
```


Then, preprocess the target graph with force-based reorder and output a reorderd graph in CSR format  
```bash
cd reorder_preprocess/force_order
python force_order.py
```

Build the simulator of CAM-based triangle counting
```bash
cd ../..
cd CAM_simulator
mkdir build
cd build
cmake ..
make
```

Run the simulator
```bash
sh CAM_run.sh
```

Build the simulator of CAM-based triangle counting
```bash
cd ../..
cd CAM_simulator
mkdir build
cd build
cmake ..
make
```

Run the simulator
```bash
cd ..
mkdir ../output/trace/force_based_order/astro_cache
build/CacheSim 8 64 2 4 2 1 ../output/trace/force_based_order/astro/astro0.trace ../output/trace/force_based_order/astro_cache/astro0.trace
```


## Data format

If you plan to run the code on other graph, please convert the graph into the following specified formats and modify the name of graph in the corresponding position of `rabbit_order.py` `force_order.py` `CAM_run.sh`.

### COO format

COO format is used to express the adjacency matrix of a graph. We use COO format as the input format of rabbit order.  
The first line is a '#' followed by the number of node $|V|$ and the number of edge $|E|$.  
There are $|E|$ lines followed, each line consists of the index of source node and destination node of an edge.  
The index of nodes are consecutive integers start from zero.  

For example, for a graph consists of a four-clique, the COO format expression can be:
```
#4 6
0 1
0 2
0 3
1 2
1 3
2 3
```

### Revised COO format

We revised the COO format to record the cluster information of nodes. We use the revised COO format as the input format of force-based reorder.  
The first line is a '#' followed by the number of node $|V|$ and the number of edge $|E|$.  
There are $|E|$ lines followed, each line consists of four numbers.  
The first and second number of node is the node index of source node and destination node of an edge. The third and fourth number is the cluster index of the source node and destination node.  
The index of nodes are consecutive integers start from zero.  

For example, for a graph consists of two triangles, the revised COO format expression can be:
```
#6 6
0 1 0 0
1 2 0 0
2 0 0 0
3 4 1 1
4 5 1 1
3 5 2 2
```

### CSR format

CSR format is a compressed storage format of an adjacency matrix. The output format of force-based reorder is a reordered graph in binary CSR format. It is also the input format of the CAM simulator.  
CSR format consists of two array: a pointer array pointing to the begin and end storage position of the neighbors of nodes; a neighbor indices array that store the neighbor indices of each node (**only neighbors with smaller indices than the node itself are stored**).  
In the view of matrix, we only store the lower triangular part of the adjacency matrix. The aforementioned array can be understand as: a row pointer array with pointers pointing to the begin and end storage position of each row; a column indices array that store the column indices of nonzero elements in the adjacency matrix, the nonzero elements with same row indices are stored consecutively.

For example, for a graph consists of a four-clique, the CSR format expression can be:
```
pointer array: 0 0 1 3 6 %point to the correspond position in the following array
neighbor/column indices array: 0 0 1 0 1 2
```

## Citation

If you find this work helpful, please consider citing our paper:
```
@article{fuclap,
  title={CLAP: Locality Aware and Parallel Triangle Counting with Content Addressable Memory},
  author={Fu, Tianyu and Wei, Chiyue and Zhu, Zhenhua and Yang, Shang and Yu, Zhongming and Dai, Guohao and Yang, Huazhong and Wang, Yu}
}
```

**CLAP paper is [here](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/e30cd3d6-8152-4358-aca5-2d289c4ddcbf.pdf).**

## CI/CD

We use pre-commit to ensure the code quality.

Check your code format by running

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Visualization

We visualize the force-based reorder method.
This is the adjacency matrix of the graph in the process of force-based reorder. See how the matrix is reordered iteratively.

![force-based demo](https://github.com/thu-nics/CLAP-triangle-counting/blob/main/figure/force_based_reorder_demo.gif)

You can also modify the code in `reorder_preprocess/force_order_demo` to visualize customized reorder methods.
