import numpy as np
from SADNG.wiki import *
from SADNG.Email import *
from sklearn.decomposition import NMF
import dgl
import torch
from SADNG.TransDGL import *
from sklearn import metrics
import time
import random
from tqdm import tqdm
from SADNG.TransDGL import *
from sklearn.metrics.pairwise import cosine_similarity
def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='micro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)

class MDNMF(object):
    def __init__(self, graph, layers):
        super(MDNMF,self).__init__()
        self.graph = graph
        self.p = len(layers)
        self.layers = layers
        self.A = graph.adjacency_matrix().to_dense().float()
        self.label = graph.ndata['label']
        self.alpha = 10
        self.beta = 10
        self.lamba = 10
        self.L = torch.diag(self.A.sum(dim=1)) - self.A
        degree = self.A.sum(dim=1)
        degree = degree[:,np.newaxis]
        degree = degree @ degree.t()
        e = self.A.sum().sum()
        self.B = self.A - degree / e
    def setup_Q(self):

        # 逆序
        self.Q_s = [None for _ in range(self.p+1)]
        self.Q_s[self.p] = torch.eye(self.layers[self.p-1]).float()
        for i in range(self.p-1, -1, -1): # [p-1, 0]
            self.Q_s[i] = self.U_s[i] @ self.Q_s[i+1]

    def pre_init(self):
        self.U_s = []
        np.random.seed(42)
        U1 = np.random.random((self.A.shape[0], self.layers[0]))
        U2 = np.random.random((self.layers[0], self.layers[1]))
        U3 = np.random.random((self.layers[1], self.layers[2]))
        self.U_s.append(torch.from_numpy(U1).float())
        self.U_s.append(torch.from_numpy(U2).float())
        self.U_s.append(torch.from_numpy(U3).float())

        V = np.random.random((self.layers[2], self.A.shape[0]))
        self.V = torch.from_numpy(V).float()

        C = np.random.random((self.layers[2], self.layers[2]))
        self.C = torch.from_numpy(C).float()

        M = np.random.random((self.A.shape[0], self.layers[2]))
        self.M = torch.from_numpy(M).float()

    def update_P(self, i):
        # 正序
        if i == 0 :
            self.P = self.U_s[0]
        else:
            self.P = self.P @ self.U_s[i]

    def update_U(self, i):
        if i == 0 :
            molecular = self.A @ self.V.t() @ self.Q_s[i+1].t()
            denominator = self.U_s[i] @ self.Q_s[i+1] @ self.V @ self.V.t() @ self.Q_s[i+1].t() + 1e-3
            self.U_s[i] = self.U_s[i] * (molecular / denominator)
        else:
            molecular = self.P.t() @ self.A @ self.V.t() @ self.Q_s[i+1].t()
            denominator = self.P.t() @ self.P @ self.U_s[i] @ self.Q_s[i+1] @ self.V @ self.V.t() @ self.Q_s[i+1].t() + 1e-3
            self.U_s[i] = self.U_s[i] * (molecular / denominator)
    def update_V(self):

        molecular = 2 *self.P.t() @ self.A - 2 * self.P.t() @ self.P @ self.V - self.lamba * self.V @ (self.L + self.L.t())
        denominator = 2 * self.alpha * (self.C.t() @ self.M.t() - self.C.t() @ self.C @ self.V) + 1e-3
        self.V = self.V * (molecular / denominator)

    def update_C(self):
        molecular = self.C.t() @ self.V @ self.V.t()
        denominator = self.M.t() @ self.V.t() + 1e-3
        self.C = self.C * (molecular / denominator)

    def update_M(self):
        molecular = 2 * self.alpha * (self.M - self.V.t() @ self.C.t())
        denominator = self.beta * (self.B + self.B.t()) @ self.M + 1e-3
        self.M = self.M * (molecular / denominator)

    def training(self):
        self.pre_init()
        nmi_list = []
        ac_list = []
        f1_list = []
        ari_list = []

        for iteration in tqdm(range(1000), desc="Training MDNMF: ", leave=True):
            self.setup_Q()
            for i in range(self.p):
                self.update_U(i)
                self.update_P(i)
            self.update_V()
            self.update_C()
            self.update_M()
            pred = self.V.t().softmax(dim=1).argmax(dim=1)
            nmi = compute_nmi(pred.numpy(), self.label.numpy())
            ac = compute_ac(pred.numpy(), self.label.numpy())
            f1 = computer_f1(pred.numpy(), self.label.numpy())
            ari = computer_ari(self.label.numpy(), pred.numpy())
            nmi_list.append(nmi)
            ac_list.append(ac)
            f1_list.append(f1)
            ari_list.append(ari)
        print(
            'nmi: {:.3f}, f1_score={:.3f},  ac = {:.3f}, ari= {:.3f}.'.format(
                np.array(nmi_list).max(),
                np.array(f1_list).max(),
                np.array(ac_list).max(),
                np.array(ari_list).max(),
            ))
if __name__ == "__main__":
    k = 5
    # dataset = EmailGraphDataset("datasets/Emails")
    dataset = TrasnDGL("cora", k)
    # dataset = WikiGraphDataset("datasets/wiki")
    model = MDNMF(dataset.graph, [128, 64, 17])
    model.training()
