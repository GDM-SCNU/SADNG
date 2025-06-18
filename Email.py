from dgl.data import DGLDataset
import dgl
import torch
import dgl.data
import numpy as np


class EmailGraphDataset(DGLDataset):
    def __init__(self, path=""):
        self.path = path
        super(EmailGraphDataset, self).__init__(name='email')

    def process(self):
        topology = self.path + "/email.cites"
        labels_path = self.path + "/email.labels"

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
        return 42

    def __len__(self):
        return 1

from scipy.sparse import csc_matrix
import scipy.sparse as sp
import pickle as pkl
datasets = EmailGraphDataset("./datasets/Emails")
graph = datasets[0]
adj = graph.adjacency_matrix().to_dense().numpy()
adj = csc_matrix(adj)
feat = csc_matrix(sp.eye(adj.shape[0]))
label = graph.ndata['label'].numpy()

data = {
    'name' : "Email",
    'adj' : adj,
    'feat': feat,
    'label': label
}
with open("Email"+'.pkl', 'wb') as f:
    pkl.dump(data, f)
print(1)