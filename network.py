import torch
import torch.nn as nn
from layers import MeanAct,DispAct
from utils import dot_product_decode
from torch.nn.parameter import Parameter


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),  # 改
            nn.ReLU(),
            nn.Linear(2000, feature_dim),  # 改
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2*feature_dim, 2000),  # 改
            nn.ReLU(),
            nn.Linear(2000, 500),  # 改
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# A general GCN layer.
class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = torch.tanh(output)
        return output

class Network(nn.Module):
    def __init__(self, views, input_size, feature_dim, n_clusters):
        super(Network, self).__init__()
        self.views = views
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, 2 * feature_dim))
        self.trans_enc = nn.TransformerEncoderLayer(d_model=2 * feature_dim, nhead=1, dim_feedforward=256)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)

        # Using list comprehensions to initialize the layers
        self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim) for v in range(views)])
        self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim) for v in range(views)])
        self.GNNs = nn.ModuleList([GNNLayer(input_size[v], feature_dim) for v in range(views)])
        self.means = nn.ModuleList([nn.Sequential(nn.Linear(input_size[v], input_size[v]), MeanAct()) for v in range(views)])
        self.disps = nn.ModuleList([nn.Sequential(nn.Linear(input_size[v],input_size[v]), DispAct()) for v in range(views)])
        self.pis = nn.ModuleList([nn.Sequential(nn.Linear(input_size[v], input_size[v]), nn.Sigmoid()) for v in range(views)])

    def forward(self, xs, adjs):
        zs = [self.GNNs[v](xs[v], adjs[v]) for v in range(self.views)]
        combined_z = torch.cat(zs, dim=1)
        h = combined_z
        A_pred = dot_product_decode(h)
        xrs = [decoder(h) for decoder in self.decoders]

        mean1 = self.means[0](xrs[0])
        mean2 = self.means[1](xrs[1])
        means = [mean1,mean2]

        disp1 = self.disps[0](xrs[0])
        disp2 = self.disps[1](xrs[1])
        disps = [disp1, disp2]

        pi1 = self.pis[0](xrs[0])
        pi2 = self.pis[1](xrs[1])
        pis = [pi1, pi2]

        return h, xrs, means, disps, pis, A_pred
