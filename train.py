import os
import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import SEGC
import torch.nn.functional as F
import addedge

parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--sigma', type=float, default=0.01, help='Sigma of gaussian distribution')
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--lam',type = float,default= 0.1,help = 'lambda')
args = parser.parse_args()


for args.dataset in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
    print("Using {} dataset".format(args.dataset))
    file = open("result_baseline.csv", "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 3
        args.lr = 3e-3
        args.dims = [500]
    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 2
        args.lr = 5e-5
        args.dims = [500]
    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 5
        args.lr = 1e-5
        args.dims = [500]
    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 5
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'uat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'corafull':
        args.cluster_num = 70
        args.gnnlayers = 2
        args.lr = 1e-3
        args.dims = [500]

    # load data
    X, y, A = load_graph_data(args.dataset, show_details=False)
    features = X
    true_labels = y
    adj = sp.csr_matrix(A)
    adj_increase = addedge.compute_ppr(adj)
    adj_i2 = addedge.compute_heat(adj)
    #adj_increase
    #adj_decrease
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_increase = adj_increase - sp.dia_matrix((adj_increase.diagonal()[np.newaxis, :], [0]), shape=adj_increase.shape)
    adj_i2 = adj_i2 - sp.dia_matrix((adj_i2.diagonal()[np.newaxis, :], [0]), shape=adj_i2.shape)
    adj.eliminate_zeros()
    print(type(adj_increase))
    adj_increase.eliminate_zeros()
    print(type(adj_i2))
    adj_i2.eliminate_zeros()
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    adj_norm_s_increase = preprocess_graph(adj_increase, args.gnnlayers, norm='sym', renorm=True)
    adj_norm_s_i2 = preprocess_graph(adj_i2,args.gnnlayers,norm='sym',renorm= True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    sm_fea_s_increase = sp.csr_matrix(features).toarray()
    sm_fea_s_i2 = sp.csr_matrix(features).toarray()
    path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    path_increase = "dataset/{}/{}_feat_sm_{}_increase.npy".format(args.dataset, args.dataset, args.gnnlayers)
    path_i2 = "dataset/{}/{}_feat_sm_{}_i2.npy".format(args.dataset, args.dataset, args.gnnlayers)
    if os.path.exists(path):
        sm_fea_s = sp.csr_matrix(np.load(path, allow_pickle=True)).toarray()
    else:
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        np.save(path, sm_fea_s, allow_pickle=True)
    if os.path.exists(path_increase):
        sm_fea_s_increase = sp.csr_matrix(np.load(path_increase,allow_pickle=True)).toarray()
    else:
        for a in adj_norm_s:
            sm_fea_s_increase = a.dot(sm_fea_s_increase)
        np.save(path_increase, sm_fea_s_increase, allow_pickle=True)
    if os.path.exists(path_i2):
        sm_fea_s_i2 = sp.csr_matrix(np.load(path_i2, allow_pickle=True)).toarray()
    else:
        for a in adj_norm_s:
            sm_fea_s_i2 = a.dot(sm_fea_s_i2)
        np.save(path_i2,sm_fea_s_i2,allow_pickle=True)

    sm_fea_s = torch.FloatTensor(sm_fea_s)
    sm_fea_s_increase = torch.FloatTensor(sm_fea_s_increase)
    sm_fea_s_i2 = torch.FloatTensor(sm_fea_s_i2)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()
    adj_1st_increase = adj_increase.toarray()
    adj_1st_i2 = adj_i2.toarray()

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for seed in range(10):
        setup_seed(seed)
        best_acc, best_nmi, best_ari, best_f1, prediect_labels = clustering(sm_fea_s, true_labels, args.cluster_num)
        model = SEGC([features.shape[1]] + args.dims)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        weights_optimizer = torch.optim.Adam([model.weights], lr=args.lr)
        model = model.to(args.device)
        inx = sm_fea_s.to(args.device)
        #target的初始化也要修改
        target = torch.FloatTensor(adj_1st).to(args.device)
        #修改这里：
        inx_i = sm_fea_s_increase.to(args.device)
        target_increase = torch.FloatTensor(adj_1st_increase).to(args.device)
        inx_d = sm_fea_s_i2.to(args.device)
        target_i2 = torch.FloatTensor(adj_1st_i2).to(args.device)
        #==================
        print('Start Training...')
        for epoch in tqdm(range(args.epochs)):
            model.train()
            z1, z2 ,zi , zd , _ = model(inx, inx_i,inx_d,is_train=True, sigma=args.sigma)

            S1 = z1 @ z2.T
            loss1 = F.mse_loss(S1, target)

            # loss1.backward(retain_graph=True)
            # optimizer.step()


            S2 = z1 @ zi.T
            loss2 = F.mse_loss(S2, target_increase)
            # loss2.backward(retain_graph=True)
            # optimizer.step()


            S3 = z1 @ zd.T
            loss3 = F.mse_loss(S3, target)
            # loss3.backward()
            # optimizer.step()
            loss = loss1 + loss2 * args.lam #0.1,0.05,0.01,0.005,0.001
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                z1, z2 , zi , zd , raw_weights= model(inx,inx_i,inx_d, is_train=False, sigma=args.sigma)
                # weights = F.softmax(raw_weights,dim = 0)
                hidden_emb = (z1 + z2 + args.lam * zi)/(2+args.lam)
                # hidden_emb = (z1+z2+zi+zd)/4
                # hidden_emb = z1 * raw_weights[0] + z2..
                acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1

                # performance_loss = calculate_performance_loss(hidden_emb.detach(), true_labels, args.cluster_num)
                # weights_optimizer.zero_grad()
                # performance_loss.backward(retain_graph=True)
                # weights_optimizer.step()
        tqdm.write('acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
        file = open("result_baseline.csv", "a+")
        print(best_acc, best_nmi, best_ari, best_f1, file=file)
        file.close()
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        f1_list.append(best_f1)

    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)
    file = open("result_baseline.csv", "a+")
    print(args.gnnlayers, args.lr, args.dims, args.sigma, file=file)
    print(round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
    print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
    print(round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
    print(round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
    file.close()