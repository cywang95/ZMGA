from tqdm import tqdm
import torch.nn as nn
import argparse
import load_data as loader
from network import Network
from layers import ZINBLoss
from utils import *


def setup_parser(data_para):
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--dataset', default=data_para)
    parser.add_argument('--lr_pre', default=0.001, type=float)  # Learning rate for pretraining
    parser.add_argument('--pretrain_epochs', default=10, type=int)  # Pretrain epochs #983
    parser.add_argument("--is_validate", default=True)
    parser.add_argument("--feature_dim", default=32)
    parser.add_argument("--weight_decay", default=0.0)
    parser.add_argument("--V", default=data_para['V'])
    parser.add_argument("--K", default=data_para['K'])
    parser.add_argument("--N", default=data_para['N'])
    parser.add_argument("--view_dims", default=data_para['n_input'])
    parser.add_argument("--alpha", default=1,type=float)
    parser.add_argument("--beta", default=1,type=float)
    return parser.parse_args()

def train_and_evaluate(model, train_loader, all_loader, X, Y, A, args, device, model_path, is_validate=False,alpha=1,beta=1 ):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pre, weight_decay=args.weight_decay)
    mse_loss_fn, zinb_loss_fn = nn.MSELoss(), ZINBLoss()
    best_score, epoch_save = 0, 0

    for epoch in tqdm(range(args.pretrain_epochs), desc='Pretraining'):
        xs, _ = next(iter(train_loader))
        xs = [x.squeeze().to(device) for x in xs]
        A = [a.to_dense().to(device) for a in A]

        # Forward pass
        zs, xrs, means, disps, pis, A_pred = model(xs, A)
        mse_loss = sum([mse_loss_fn(xs[v], xrs[v]) for v in range(args.V)])
        zinb_loss = sum([zinb_loss_fn(xs[v], means[v], disps[v], pis[v]) for v in range(args.V)])
        re_graphloss = sum([mse_loss_fn(A_pred.view(-1), a.view(-1)) for a in A])

        # Backward pass
        loss = zinb_loss +  alpha*mse_loss + beta*re_graphloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_validate and (epoch + 1) % 1 == 0:
            ari, nmi, _, pur = validate(X, Y, model, device=device, A=A, y_true=Y[0].copy(), args=args)
            if ari > best_score:
                best_score, epoch_save,nmis,purs = ari, epoch, nmi, pur
                torch.save(model.state_dict(), model_path)
    return  best_score, epoch_save, nmis,purs

def main():
    my_data_dic = loader.ALL_data
    for i_d, data_para in my_data_dic.items():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_para = my_data_dic[i_d]
        args = setup_parser(data_para)
        X, Y, A = loader.load_data(args.dataset)
        model = Network(args.V, args.view_dims, args.feature_dim, args.K).to(device)

        train_dataset = TrainDataset(X, Y)
        batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.N, drop_last=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

        all_dataset = TrainDataset(X, Y)
        batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.N, drop_last=False)
        all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler_all)

        model_path = f'/mnt/d/Code/ZMGA/test.pth'
        best_score, epoch_save,nmis,purs = train_and_evaluate(model, train_loader, all_loader, X, Y, A, args, device, model_path, is_validate=args.is_validate,alpha=args.alpha,beta=args.beta)
        print("final ARI:" ,best_score, "final NMI:" , nmis ,"final PUR:", purs)



if __name__ == "__main__":
    main()
