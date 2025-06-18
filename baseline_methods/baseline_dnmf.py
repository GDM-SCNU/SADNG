"""DANMF class."""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import NMF
from sklearn import metrics
from SADNG.parser import *
from SADNG.TransDGL import *
import time
import torch
class DNMF(object):

    def __init__(self, graph, args):

        self.graph = graph
        self.A = graph.adjacency_matrix().to_dense().numpy()
        self.L = np.diag(self.A.sum(axis=1)) - self.A
        self.D = self.L+self.A
        self.args = args
        self.p = len(self.args.layers)
        self.labels = graph.ndata['label']

    def setup_z(self, i):
        """
        Setup target matrix for pre-training process.
        """
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i-1]

    def param_init(self, i):
        """
        Pretraining a single layer of the model with sklearn.
        :param i: Layer index.
        """
        np.random.seed(42)
        U = np.random.random((self.Z.shape[0], self.args.layers[i]))
        V = np.random.random((self.args.layers[i], self.Z.shape[1]))
        return U, V

    def pre_training(self):
        """
        Pre-training each NMF layer.
        """
        print("\nLayer pre-training started. \n")
        self.U_s = []
        self.V_s = []
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_z(i)
            U, V = self.param_init(i)
            self.U_s.append(U)
            self.V_s.append(V)

    def setup_Q(self):
        """
        Setting up Q matrices.
        """
        self.Q_s = [None for _ in range(self.p+1)]
        # 最后一层
        self.Q_s[self.p] = np.eye(self.args.layers[self.p-1])
        # 逆序
        for i in range(self.p-1, -1, -1):
            self.Q_s[i] = np.dot(self.U_s[i], self.Q_s[i+1])

    def update_U(self, i):
        """
        Updating left hand factors.
        :param i: Layer index.
        """
        if i == 0:
            R = self.U_s[0].dot(self.Q_s[1].dot(self.VpVpT).dot(self.Q_s[1].T))
            R = R+self.A_sq.dot(self.U_s[0].dot(self.Q_s[1].dot(self.Q_s[1].T)))
            Ru = 2*self.A.dot(self.V_s[self.p-1].T.dot(self.Q_s[1].T))
            self.U_s[0] = (self.U_s[0]*Ru)/np.maximum(R, 10**-10)
        else:
            R = self.P.T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.VpVpT).dot(self.Q_s[i+1].T)
            R = R+self.A_sq.dot(self.P).T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.Q_s[i+1].T)
            Ru = 2*self.A.dot(self.P).T.dot(self.V_s[self.p-1].T).dot(self.Q_s[i+1].T)
            self.U_s[i] = (self.U_s[i]*Ru)/np.maximum(R, 10**-10)

    def update_P(self, i):
        """
        Setting up P matrices.
        :param i: Layer index.
        """
        if i == 0:
            self.P = self.U_s[0]
        else:
            self.P = self.P.dot(self.U_s[i])

    def update_V(self, i):
        """
        Updating right hand factors.
        :param i: Layer index.
        """
        if i < self.p-1:
            Vu = 2*self.A.dot(self.P).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])+self.V_s[i]
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)
        else:
            Vu = 2*self.A.dot(self.P).T+(self.args.lamb*self.A.dot(self.V_s[i].T)).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])
            Vd = Vd + self.V_s[i]+(self.args.lamb*self.D.dot(self.V_s[i].T)).T
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)


    def compute_nmi(self, pred, labels):
        return metrics.normalized_mutual_info_score(labels, pred)

    def compute_ac(self, pred, labels):
        return metrics.accuracy_score(labels, pred)

    def computer_ari(self, true_labels, pred_labels):
        return metrics.adjusted_rand_score(true_labels, pred_labels)

    def computer_f1(self, pred, labels):
        return metrics.f1_score(labels, pred, average='micro')

    def training(self):
        """
        Training process after pre-training.
        """
        print("\n\nTraining started. \n")
        self.loss = []
        self.A_sq = self.A.dot(self.A.T)
        nmi_list = []
        ac_list = []
        f1_list = []
        ari_list = []
        for iteration in tqdm(range(self.args.iterations), desc="Training pass: ", leave=True):
            self.setup_Q()
            self.VpVpT = self.V_s[self.p-1].dot(self.V_s[self.p-1].T)
            for i in range(self.p):
                self.update_U(i)
                self.update_P(i)
                self.update_V(i)
            pred = torch.from_numpy(self.V_s[-1].T).softmax(dim=1).argmax(dim=1).numpy()
            nmi = self.compute_nmi(pred, self.labels.numpy())
            ac = self.compute_ac(pred, self.labels.numpy())
            f1 = self.computer_f1(pred, self.labels.numpy())
            ari = self.computer_ari(self.labels.numpy(), pred)
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
    args = parameter_parser()
    #
    # dataset = TrasnDGL("texas", args.layers[-1])
    # from SADNG.Email import *
    from SADNG.wiki import *
    dataset = WikiGraphDataset("datasets/wiki")
    # from SADNG.Email import EmailGraphDataset
    # dataset = EmailGraphDataset("datasets/Emails")
    dnmf = DNMF(dataset.graph, args)
    dnmf.pre_training()
    dnmf.training()
