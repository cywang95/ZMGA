import torch
from datasets import Data_Sampler, TrainDataset
import torch.nn.functional as F
from sklearn.cluster import KMeans
from Nmetrics import evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import pairwise_distances as pair
import numpy as np
import scipy.sparse as sp



def extract_features(model, all_loader, device, A, args):
    """Extract features from the model."""
    fea_emb = []
    with torch.no_grad():
        xs, _ = next(iter(all_loader))
        xs = [torch.squeeze(x).to(device) for x in xs]
        zs, xrs, means, disps, pis, A_pred = model(xs, A)
        fea_emb.append(zs.cpu())
    return fea_emb

def validate(X,Y , model, device, A, y_true, args):

    all_dataset = TrainDataset(X, Y)
    batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.N, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler_all)
    commonZ = extract_features(model, all_loader, device, A, args)
    commonZ = commonZ[0].numpy()

    kmeans = KMeans(n_clusters=args.K)
    kmeans.fit(commonZ)
    pred2 = kmeans.labels_
    acc, nmi, purity, fscore, precision, recall, ari = evaluate(y_true, pred2)
    return ari, nmi, acc, purity
def kl_loss(z, cluster_layer):
    alpha=1
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - cluster_layer) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()

    log_q = torch.log(q)
    loss = torch.nn.functional.kl_div(log_q, p, reduction='batchmean')
    return loss

def X2A(features,method):

    N = features.shape[0]

    if method == 'heat':
        dist = -0.5 * pair(features, metric='manhattan') ** 2
        dist = np.exp(dist)

    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)

    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)


    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    topk=10
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    # 构建邻接矩阵
    A = np.zeros((N, N))
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                continue
            A[i, vv] = 1
            A[vv, i] = 1  # 确保矩阵是对称的

    adj = sp.coo_matrix(A)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)



def NormalizeFeaTorch(features):  # features为nxd维的特征矩阵
    rowsum = torch.tensor((features ** 2).sum(1))
    r_inv = torch.pow(rowsum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    normalized_feas = torch.mm(r_mat_inv, features)
    return normalized_feas


def get_Similarity(fea_mat1, fea_mat2):
    Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)
    return Sim_mat

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def clustering(feature, cluster_num):
    # predict_labels,  cluster_centers = kmeans(X=feature, num_clusters=cluster_num, distance='euclidean', device=torch.device('cuda'))
    predict_labels,  cluster_centers = kmeans(X=feature, num_clusters=cluster_num, distance='euclidean', device=device)
    return predict_labels.numpy(), cluster_centers
    # acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    # return 100 * acc, 100 * nmi, 100 * ari, 100 * f1, predict_labels.numpy(), initial



def euclidean_dist(x, y, root=False):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

