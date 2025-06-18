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
def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='micro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)

def train(graph, components):
    A = graph.adjacency_matrix().to_dense()
    n = A.shape[0]
    np.random.seed(42)
    torch.manual_seed(826)
    random.seed(826)
    W = np.random.random(size=(n, components))
    Z = np.random.random(size=(components, n))
    W = torch.from_numpy(W).float()
    Z = torch.from_numpy(Z).float()
    labels = graph.ndata['label']
    start = time.clock()
    nmi_list = []
    ac_list = []
    f1_list = []
    ari_list = []
    for i in tqdm(range(1000), desc = "Sun's model ", leave= True):
        denominator =  W @ Z @ Z.t() + A @ A.t() @ W + 1e-3
        molecular = A @ Z.t()
        W = W * (molecular / denominator)

        denominator = W.t() @ W @ Z + Z + 1e-3
        molecular = W.t() @ A
        Z = Z * (molecular / denominator)
        pred = W.softmax(dim=1).argmax(dim=1)
        nmi = compute_nmi(pred.numpy(), labels.numpy())
        ac = compute_ac(pred.numpy(), labels.numpy())
        f1 = computer_f1(pred.numpy(), labels.numpy())
        ari = computer_ari(labels.numpy(), pred.numpy())
        nmi_list.append(nmi)
        ac_list.append(ac)
        f1_list.append(f1)
        ari_list.append(ari)

    end = time.clock()

        # nmi: 0.000, f1_score=0.302,  ac = 0.302, ari= 0.000.
    print(
            'nmi: {:.3f}, f1_score={:.3f},  ac = {:.3f}, ari= {:.3f}.'.format(
                np.array(nmi_list).max(),
                np.array(f1_list).max(),
                np.array(ac_list).max(),
                np.array(ari_list).max(),
        ))

    print(end - start)

if __name__ == "__main__":
    k = 5
    dataset = EmailGraphDataset("datasets/Emails")
    # dataset = TrasnDGL("texas", k)
    train(dataset.graph, dataset.num_classes)