# coding=utf-8
# Author: Jung
# Time: 2022/3/6 20:57

from dgl.data import DGLDataset
import dgl
import torch
import torch.nn.functional as F
import dgl.data
import numpy as np
from collections import defaultdict
file_path = "datasets\\"

class VirtualDGL(DGLDataset):

    def __init__(self, name = "", num_classes = 0):
        self.path = name
        self.components = num_classes
        super(VirtualDGL, self).__init__(name = name)

    def process(self):
        topology = file_path + self.path +"\\" + "network.dat" # 拓扑信息
        label = file_path + self.path + "\\" + "community.dat" # 标签
        adj_str = []
        adj_end = []
        labels = []  # 用于存放每个节点对应类别的列表
        compo = set()
        with open(label, "r", encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                assert len(info) == 2
                lab = int(info[1]) - 1
                labels.append(lab)
                compo.add(lab)
        labels = np.asarray(labels, dtype=np.int64)
        self.components = len(compo)
        print("社区数： {}".format(self.components))
        num_nodes = len(labels)
        with open(topology, "r", encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                assert len(info) == 2
                u = int(info[0]) - 1
                v = int(info[1]) - 1
                adj_str.append(u)
                adj_end.append(v)
        estr = adj_str + adj_end
        eend = adj_end + adj_str
        labels = torch.tensor(labels)
        self.graph = dgl.graph((estr, eend), num_nodes=num_nodes)
        self.graph = dgl.add_self_loop(self.graph)  # 加自环
        self.graph.ndata['label'] = labels
    def __getitem__(self, i):
        return self.graph

    @property
    def num_classes(self):
        return self.components

    def __len__(self):
        return 1