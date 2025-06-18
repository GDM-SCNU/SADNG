import numpy as np
from SADNG.wiki import *
from SADNG.Email import *
from sklearn.decomposition import NMF
import dgl
import torch
from SADNG.TransDGL import *
from sklearn import metrics
import time
def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='micro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)
def run(A):
    start = time.clock()
    nmf_model = NMF(n_components= 7,
                            init = "random",
                            random_state= 42,
                            max_iter=1000)
    U = nmf_model.fit_transform(A)
    V = nmf_model.components_
    end = time.clock()
    print(end-start)
    return V.T

if __name__ == "__main__":

    # dataset = WikiGraphDataset("datasets/wiki")
    # dataset = EmailGraphDataset("datasets/Emails")
    dataset = TrasnDGL("cora", 7)
    A = dataset.graph.adjacency_matrix().to_dense()
    labels = dataset.graph.ndata['label']
    V = run(A)
    pred = torch.from_numpy(V).softmax(dim=1).argmax(dim=1)
    nmi = compute_nmi(pred.numpy(), labels.numpy())
    ac = compute_ac(pred.numpy(), labels.numpy())
    f1 = computer_f1(pred.numpy(), labels.numpy())
    ari = computer_ari(labels.numpy(), pred.numpy())
    print(
        'nmi: {:.3f}, f1_score={:.3f},  ac = {:.3f}, ari= {:.3f}.'.format(
            nmi,
            f1,
            ac,
            ari,
        ))
