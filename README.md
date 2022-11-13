# CLAP-triangle-counting
This is the implementation of CLAP, a locality aware and parallel triangle counting with content addressable memory.

## Code structure
`CAM_simulator` contains a simulator of our CAM-based triangle counting architecture implemented by c++, it perform the triangle counting of a given target graph and generate the memory trace at the same time  

`reorder_preprocess` contains the pre-processing technique implemented by python that reorder the graph to enable better performance of triangle counting  

`data` contains the example datesets to run the code  

## Usage

Take the dataset citeseer as an example
First, preprocess the target graph with force-based reorder and output a reorderd graph in CSR format  
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
sh run.sh
```

## Data format
If you plan to run the code on other graph, please convert the graph into the following specified formats
### COO format
COO format is used to express the adjacency matrix of a graph. We use COO format as the input format of force-based reorder.  
The first line is a '#' followed by the number of node |V| and the number of edge |E|.  
There are |E| lines followed, each line consists of the index of source node and destination node of an edge.  
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

### CSR format
CSR format is a compressed storage format of an adjacency matrix. The output format of force-based reorder is a reordered graph in binary CSR format. It is also the input format of the CAM simulator.
CSR format consists of two array: a pointer array pointing to the begin and end storage position of the neighbors of nodes; a neighbor indices array that store the neighbor indices of each node (**only neighbors with smaller indices than the node itself are stored**).
In the view of matrix, we only store the lower triangular part of the adjacency matrix. The aforementioned array can be understand as: a row pointer array with pointers pointing to the begin and end storage position of each row; a column indices array that store the column indices of nonzero elements in the adjacency matrix, the nonzero elements with same row indices are stored consecutively.

For example, for a graph consists of a four-clique, the CSR format expression can be:
```
pointer array: 0 0 1 3 6
neighbor/column indices array: 0 0 1 0 1 2
```