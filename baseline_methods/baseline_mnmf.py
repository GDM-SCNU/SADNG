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
from sklearn.metrics.pairwise import cosine_similarity
def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='micro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)

def train(graph, components):
    eta = 5.0
    alpha = 0.05
    beta = 0.05
    lambd = 0.2
    A = graph.adjacency_matrix().to_dense()
    degree = A.sum(dim=1)
    degree = degree[:, np.newaxis]
    degree = degree @ degree.t()
    e = A.sum().sum()
    B = degree / e
    S2 = torch.from_numpy(cosine_similarity(A)).float()
    S = A + eta * S2
    dimensions = 16
    n = A.shape[0]
    np.random.seed(42)
    torch.manual_seed(826)
    random.seed(826)
    M = np.random.random(size=(n, dimensions))
    M = torch.from_numpy(M).float()

    U = np.random.random(size=(n, dimensions))
    U = torch.from_numpy(U).float()

    C = np.random.random(size=(components, dimensions))
    C = torch.from_numpy(C).float()

    H = np.random.random(size=(n, components))
    H = torch.from_numpy(H).float()


    labels = graph.ndata['label']
    start = time.clock()
    nmi_list = []
    ac_list = []
    f1_list = []
    ari_list = []
    for i in tqdm(range(1000), desc = "MNMF", leave= True):

        molecular = S @ U
        denominator = M @ U.t() @ U + 1e-3
        M = M * (molecular / denominator)

        molecular = S.t() @ M + alpha * H @ C
        denominator = U @ (M.t() @ M + alpha * C.t() @ C) + 1e-3
        U = U * (molecular / denominator)

        molecular = H.t() @ U
        denominator = C @ U.t() @ U + 1e-3
        C = C * (molecular / denominator)

        deta = (2 * beta * B @ H) * (2 * beta * B @ H) + (16 * lambd * H @ H.t() @ H ) * (2 * beta * A @ H + 2 * alpha * U @ C.t() + (4 * lambd - 2 * alpha)*H)
        sq_deta = torch.sqrt(deta)
        molecular = -2 * beta * B @ H + sq_deta
        denominator = 8 * lambd * H @ H.t() @ H + 1e-3
        H = H * molecular / denominator



        pred = H.softmax(dim=1).argmax(dim=1)
        nmi = compute_nmi(pred.numpy(), labels.numpy())
        ac = compute_ac(pred.numpy(), labels.numpy())
        f1 = computer_f1(pred.numpy(), labels.numpy())
        ari = computer_ari(labels.numpy(), pred.numpy())
        nmi_list.append(nmi)
        ac_list.append(ac)
        f1_list.append(f1)
        ari_list.append(ari)

    end = time.clock()

    print(
            'nmi: {:.3f}, f1_score={:.3f},  ac = {:.3f}, ari= {:.3f}.'.format(
                np.array(nmi_list).max(),
                np.array(f1_list).max(),
                np.array(ac_list).max(),
                np.array(ari_list).max(),
        ))

    print(end - start)

if __name__ == "__main__":
    k = 17
    # dataset = EmailGraphDataset("datasets/Emails")
    # dataset = TrasnDGL("texas", k)
    dataset = WikiGraphDataset("datasets/wiki")
    train(dataset.graph, dataset.num_classes)