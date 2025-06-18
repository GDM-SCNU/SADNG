from dgl.data import DGLDataset
import dgl
import torch
import dgl.data
import numpy as np


class WikiGraphDataset(DGLDataset):
    def __init__(self, path=""):
        self.path = path
        super(WikiGraphDataset, self).__init__(name='wiki')

    def process(self):
        topology = self.path + "/Wiki_edgelist.txt"
        labels_path = self.path + "/wiki_labels.txt"

        node_map = {}  # 将节点进行重新编码
        labels = []  # 用于存放每个节点对应类别的列表
        label_map = {}  # 将label映射为数字

        with open(labels_path, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                labels.append(int(info[-1]))
        labels = np.asarray(labels, dtype=np.int64)
        adj_str = []
        adj_end = []
        with open(topology, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                u = int(info[0])
                v = int(info[1])
                adj_str.append(u)
                adj_end.append(v)
        estr = adj_str + adj_end
        eend = adj_end + adj_str
        num_nodes = len(labels)
        labels = torch.tensor(labels)
        self.graph = dgl.graph((estr, eend), num_nodes=num_nodes)
        self.graph = dgl.add_self_loop(self.graph)  # 加自环
        self.graph.ndata['label'] = labels


    def __getitem__(self, i):
        return self.graph

    @property
    def num_classes(self):
        return 17

    def __len__(self):
        return 1

# dataset = TexasGraphDataset()
