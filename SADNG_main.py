# coding=utf-8
# Author: Jung
# Time: 2022/1/9 15:30
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from dgl.nn import GraphConv

from scipy import sparse
import numpy as np
import networkx as nx
from sklearn.decomposition import NMF
from tqdm import tqdm
from SADNG.parser import *
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from sklearn.metrics.pairwise import cosine_similarity
from SADNG.TransDGL import *
class Encoder(object):
    def __init__(self, graph, args):
        super(Encoder, self).__init__()
        self.graph = graph
        self.args = args
        self.p = len(self.args.layers)
        self.A = graph.adjacency_matrix().to_dense().float()
        self.setup_D()
        self.nmi = []
        self.ac = []
        self.f1 = []
        self.labels = graph.ndata['label']
    def setup_D(self):
        self.L = torch.diag(self.A.sum(axis=1)) - self.A
        self.D = self.L + self.A
    def setup_z(self, i):
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i-1]

    def sklearn_pretain(self, i):
        nmf_model = NMF(n_components= self.args.layers[i],
                        init = "random",
                        random_state= self.args.seed,
                        max_iter=self.args.pre_iterations)
        U = nmf_model.fit_transform(self.Z)
        V = nmf_model.components_
        return torch.from_numpy(U).float(), torch.from_numpy(V).float()

    def train(self):
        self.V_s = []
        self.U_s = []
        for i in tqdm(range(self.p), desc = "Layers trained: ", leave= True):
            self.setup_z(i)
            U, V = self.sklearn_pretain(i)
            self.V_s.append(V)
            self.U_s.append(U)
    def clear_feat(self):
        self.V_s.clear()
        self.U_s.clear()


    # DANMF iterator
    def setup_Q(self):
        """
        Setting up Q matrices.
        """
        self.Q_s = [None for _ in range(self.p+1)]
        # 最后一层
        self.Q_s[self.p] = torch.eye(self.args.layers[self.p-1]).float()
        # 逆序
        for i in range(self.p-1, -1, -1):
            self.Q_s[i] = self.U_s[i].matmul(self.Q_s[i+1])

    def update_U(self, i):
        """
        Updating left hand factors.
        :param i: Layer index.
        """
        if i == 0:
            R = self.U_s[0].matmul(self.Q_s[1].matmul(self.VpVpT).matmul(self.Q_s[1].t()))
            R = R+ self.A_sq.matmul(self.U_s[0].matmul(self.Q_s[1].matmul(self.Q_s[1].t()))) + 1e-3
            Ru = 2 * self.A.matmul(self.V_s[self.p-1].t().matmul(self.Q_s[1].t()))
            self.U_s[0] = (self.U_s[0] * Ru) / R
        else:
            R = self.P.t().matmul(self.P).matmul(self.U_s[i]).matmul(self.Q_s[i+1]).matmul(self.VpVpT).matmul(self.Q_s[i+1].t())
            R = R + self.A_sq.matmul(self.P).t().matmul(self.P).matmul(self.U_s[i]).matmul(self.Q_s[i+1]).matmul(self.Q_s[i+1].t()) + 1e-3
            Ru = 2 * self.A.matmul(self.P).t().matmul(self.V_s[self.p-1].t()).matmul(self.Q_s[i+1].t())
            self.U_s[i] = (self.U_s[i]*Ru)/ R

    def update_P(self, i):
        """
        Setting up P matrices.
        :param i: Layer index.
        """
        if i == 0:
            self.P = self.U_s[0]
        else:
            self.P = self.P.matmul(self.U_s[i])

    def update_V(self, i):
        """
        Updating right hand factors.
        :param i: Layer index.
        """
        if i < self.p-1:
            Vu = 2*self.A.matmul(self.P).t()
            Vd = self.P.t().matmul(self.P).matmul(self.V_s[i])+self.V_s[i] + 1e-3
            self.V_s[i] = self.V_s[i] * Vu/ Vd
        else:
            Vu = 2*self.A.matmul(self.P).t() + (self.args.lamb * self.A.matmul(self.V_s[i].t())).t()
            Vd = self.P.t().matmul(self.P).matmul(self.V_s[i])
            Vd = Vd + self.V_s[i] + 1e-3 + (self.args.lamb * self.D.matmul(self.V_s[i].t())).t()
            self.V_s[i] = self.V_s[i] * Vu/ Vd
    def training(self):
        """
        Training process after pre-training.
        """
        self.A_sq = self.A.matmul(self.A.t())
        for iteration in tqdm(range(self.args.iterations), desc="Training pass: ", leave=True):
            self.setup_Q()
            self.VpVpT = self.V_s[self.p - 1].matmul(self.V_s[self.p - 1].t())
            for i in range(self.p):
                self.update_U(i)
                self.update_P(i)
                self.update_V(i)
        return self.V_s[-1]

class Transform(nn.Module):

    def __init__(self, graph, args):
        super(Transform, self).__init__()
        self.graph = graph
        self.args = args
        self.layers = self.args.layers
        self.trans1 = Linear(self.layers[0], self.layers[0])
        self.trans2 = Linear(self.layers[1], self.layers[1])
        self.trans3 = Linear(self.layers[2], self.layers[2])


    def forward(self, v1 , v2 , v3):
        H1 = F.relu(self.trans1(v1.t()))
        H2 = F.relu(self.trans2(v2.t()))
        H3 = F.relu(self.trans3(v3.t()))

        return H1 , H2 , H3

class GraphNMF(nn.Module):

    def __init__(self, graph, args):
        super(GraphNMF,self).__init__()
        self.graph = graph
        self.args = args
        self.layers = self.args.layers
        self.A = graph.adjacency_matrix().to_dense().float()
        # self.S = torch.diag(torch.ones(size = (self.graph.ndata['feat'].shape)))
        self.S = torch.eye(self.A.shape[0])

        self.transform = Transform(graph, args)
        self.encoder = Encoder(graph, args)
        self.B = self.compute_B_matrix()

        self.conv1 = GraphConv(self.S.shape[1], self.layers[0]) # 128
        self.conv2 = GraphConv(self.layers[0], self.layers[1])  # 64
        self.conv3 = GraphConv(self.layers[1], self.layers[2])  # 32

    def compute_B_matrix(self):
        nx_graph = dgl.to_networkx(self.graph)
        degree = nx.degree(nx_graph)
        deg = torch.FloatTensor([d for id, d in degree]).reshape(-1, 1)
        sum_deg = deg.sum()
        B = self.A - (deg.matmul(deg.t()) / sum_deg)
        return B
    def forward(self):

        self.encoder.train()
        # H1, H2, H3 = self.transform(self.encoder.V_s[0], self.encoder.V_s[1], self.encoder.V_s[2])
        H1, H2, H3 = self.encoder.V_s[0].t(), self.encoder.V_s[1].t(), self.encoder.V_s[2].t()

        h = self.conv1(self.graph, self.S)
        h = F.relu(h) # 128
        sigma = 0.3 # 3
        h = self.conv2(self.graph, (1 - sigma)*h + sigma * H1)
        h = F.relu(h) # 64
        sigma = 0.9 # 9
        h = self.conv3(self.graph, (1 - sigma)*h + sigma * H2)


        return h # 32

def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='micro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)

def degree_graph(M):
    return M.sum(1), M.sum(1).sum(0)

def PMI(A, P):
    node_degree, total_degree = degree_graph(P)
    node_degree = np.expand_dims(node_degree, 0)
    degree_matrix = np.dot(node_degree.T, node_degree)

    res = (A * total_degree + 1e-10) / (degree_matrix + 1e-10)
    res_log  = np.log(res) - np.log(1)
    return torch.from_numpy(np.maximum(res_log, 0))

def train(graph, args):
    model = GraphNMF(graph, args)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-2)
    test_nmi_list = []
    test_ac_list = []
    test_f1_list = []
    test_ari_list = []
    labels = graph.ndata['label']


    for i in range(1, 500):

        gcn_pred = model()
        log_gcn = F.log_softmax(gcn_pred, dim=1)
        gcn_pred = torch.sigmoid(gcn_pred)

        nmf_pred = model.encoder.training().t().softmax(dim=1)

        loss = -1e-4 * torch.trace(gcn_pred.t().matmul(model.B).matmul(gcn_pred))
        #loss += 1e-4 * torch.trace(nmf_pred.t().matmul(model.encoder.A).matmul(nmf_pred))
        A_1 = torch.sigmoid(gcn_pred.matmul(gcn_pred.t()))
        A_0 = 1 - A_1
        A = model.A * torch.log(A_1) + (1 - model.A) * torch.log(A_0)
        loss += 1 / 2708 * -(A.sum().sum())

        # X = cosine_similarity(gcn_pred.detach()) # 1 - torch.exp(-gcn_pred.matmul(gcn_pred.t()).detach())
        # X = X - np.eye(X.shape[0])
        # indices = X.max(axis=1)
        # indices = np.expand_dims(indices,1)
        # X = np.where(X[np.arange(len(X)),] == indices[np.arange(len(indices))], 1, 0)

        # loss += torch.norm(model.encoder.A - model.encoder.U_s[0].matmul(model.encoder.U_s[1]).matmul(model.encoder.U_s[2]).matmul(model.encoder.V_s[-2]))


        loss += F.kl_div(log_gcn, nmf_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # 修改图结构

        # 移除边
        X = cosine_similarity(gcn_pred.detach()) # 计算相似度
        # X = (gcn_pred @ gcn_pred.t()).detach().numpy()
        X = abs(X - np.eye(X.shape[0])) # 自己与自己的相似度设为0
        purty = X * model.encoder.A.numpy() # 得到存在边的purty矩阵
        threshold =  (1. / (model.encoder.A.numpy().sum(0).sum(0))) * (purty.sum(0).sum(0))
        purty = np.where(purty < threshold, 0, 1) # remove
        model.encoder.A += torch.from_numpy(purty).float()
       #  model.encoder.A = torch.from_numpy(np.where(model.encoder.A <0, 0, model.encoder.A)).float()

        # 加边
        purty = X * model.encoder.A.numpy() # 有边时的相似度
        aff = np.zeros(shape=(gcn_pred.shape[0], model.layers[2]))
        aff[np.arange(aff.shape[0]), gcn_pred.detach().argmax(dim=1)] = 1
        affiliation = aff.dot(aff.T) # 隶属社区（节点位于同一个社区，则为1）
        purty = affiliation * purty
        threshold = (1. / (affiliation.sum(0).sum(0))) * (purty.sum(0).sum(0))
        purty = np.where(purty >= threshold,1 ,0)
        model.encoder.A += torch.from_numpy(purty).float()

        model.encoder.setup_D()


        if i % 1 == 0:
            model.eval()
            # pred = gcn_pred.argmax(1)
            pred = nmf_pred.argmax(dim=1)
            model.encoder.clear_feat()
            nmi = compute_nmi(pred.numpy(), labels.numpy())
            ac = compute_ac(pred.numpy(), labels.numpy())
            f1 = computer_f1(pred.numpy(), labels.numpy())
            ari = computer_ari(labels.numpy(), pred.numpy())
            test_nmi_list.append(nmi)
            test_ac_list.append(ac)
            test_f1_list.append(f1)
            test_ari_list.append(ari)
            print(
                'epoch={}, loss={:.3f},  nmi: {:.3f}, f1_score={:.3f},  ac = {:.3f}, ari= {:.3f}, MAX_NMI={:.3f}, MAX_F1={:.3f}, MAX_AC = {:.3f}, MAX ARI = {:.3f}'.format(
                    i,
                    loss,
                    nmi,
                    f1,
                    ac,
                    ari,
                    max(test_nmi_list),
                    max(test_f1_list),
                    max(test_ac_list),
                    max(test_ari_list)
                ))

if __name__ == "__main__":
    args = parameter_parser()

    ####### load data #####33#
    # from SADNG.virtual_dgl import *
    # dataset = VirtualDGL("virtual1000")
    # dataset = TrasnDGL("cornell", args.layers[-1])
    # dataset = dgl.data.CoraGraphDataset()
    # dataset = WikiCSGraphDataset(r"datasets/Wikics")

    # from SADNG.Email import *
    # dataset = EmailGraphDataset("datasets/Emails")
    from SADNG.wiki import *
    dataset = WikiGraphDataset("datasets/wiki")
    train(dataset.graph, args)

    # other datasets use this way
    # graph = dataset[0]
    # train(dataset.graph, args)


