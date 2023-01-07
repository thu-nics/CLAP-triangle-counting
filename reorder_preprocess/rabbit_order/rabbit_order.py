#!/usr/bin/env python3
import time
import torch
import rabbit

class graph_input(object):
    def __init__(self, path=None):
        self.load_flag = False
        self.reorder_flag = False
        self.path = path
        self.edge_index = None
        
        self.dgl_flag = False
        self.pyg_flag = False

        self.dgl_graph = False
        self.pyg_graph = False


    def load(self, load_from_txt=True):
        '''
        load the graph from the disk --> CPU memory.
        '''
        if self.path == None:
            raise ValueError("Graph path must be assigned first")
        
        start = time.perf_counter()
        '''
        edge in the txt format:
        s0 d0
        s1 d1
        s2 d2
        '''
        fp = open(self.path, "r")
        src_li = []
        dst_li = []
        
        info = fp.readline()
        for line in fp:
            tmp = line.rstrip('\n').split()
            src, dst = int(tmp[0]), int(tmp[1])
            src_li.append(src)
            dst_li.append(dst)
        src_idx = torch.IntTensor(src_li)
        dst_idx = torch.IntTensor(dst_li)

        self.edge_index = torch.stack([src_idx, dst_idx], dim=0)



        dur = time.perf_counter() - start
        print("Loading graph from txt source (ms): {:.3f}".format(dur*1e3))

        self.load_flag = True
        return info

    def reorder(self):
        '''
        reorder the graph if specified.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before reordering.")
        
        print("Original edge_index\n", self.edge_index)
        new_edge_index = rabbit.reorder(self.edge_index)
        print("Reordered edge_index\n", new_edge_index)

        # for i in range(len(new_edge_index[1])):
        #     src, dst = new_edge_index[0][i], new_edge_index[1][i]
            # print('{}--{}'.format(src, dst))
        # print(new_edge_index.size())

        self.reorder_flag = True
        return new_edge_index
        

    def create_dgl_graph(self):
        '''
        create a DGL graph from edge index.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting PyG graph.")
        
        self.dgl_flag = True
    
    def create_pyg_graph(self):
        '''
        create a PyG graph from edge index.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting DGL graph.")
        
        self.pyg_flag = True


    def get_dgl_graph(self):
        '''
        return the dgl graph.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting DGL graph.")
        if not self.dgl_flag:
            raise ValueError("DGL Graph MUST be created Before getting DGL graph.")

        return self.dgl_graph

    def get_pyg_graph(self):
        '''
        return the pyg graph.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting PyG graph.")

        if not self.pyg_flag:
            raise ValueError("PyG Graph MUST be created Before getting PyG graph.")
        
        return self.pyg_graph


if __name__ == "__main__":
    # path = sys.argv[1]
    graph_list = ["astro"]
    for graph in graph_list:
        path = "../../data/edge_list/random/"+graph+".txt"
        pathout = "../../data/rabbit_with_cluster/"+graph+".txt"
    
        input_mat = path
        output_mat = pathout

        graph = graph_input(input_mat)
        info = graph.load(load_from_txt=True)
        new_edge_idx = graph.reorder()
        
        out_idx = torch.stack([new_edge_idx[0],new_edge_idx[1],new_edge_idx[2], new_edge_idx[3], new_edge_idx[4], new_edge_idx[5]],dim = 0)


        fout = open(output_mat,'w')
        fout.write(info) 
        for i in range(len(out_idx[0])):
            fout.write(str(int(out_idx[0][i]))+' '+str(int(out_idx[1][i]))+' '+str(int(out_idx[2][i]))+' '+str(int(out_idx[3][i]))+'\n')
        fout.close()


  

