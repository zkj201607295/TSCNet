import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits
#from sampler import NeighborSampler
from torch_geometric.loader import NeighborSampler
from models import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
from gat_neighsampler import GAT_NeighSampler
from torch_geometric.utils import to_undirected
from sklearn.metrics import f1_score
from ogb.nodeproppred import Evaluator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


def RunExp(args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate, train_loader, device):
        model.train()

        #pbar = tqdm(total=data.train_mask.size(0), ncols=80)
        #pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_correct = 0
        time_total = 0
        for batch_size, n_id, adjs in train_loader:
            t_st = time.time()
            adjs = [adj for adj in adjs]
            #adjs = [adjs.to(device)]
            optimizer.zero_grad()
            #out = model(data)[data.train_mask]
            out = model(data.x[n_id], adjs, device)
            #print(out)
            #print(data.y[n_id[:batch_size]])
            nll = F.nll_loss(out, data.y[n_id[:batch_size]])
            loss = nll
            reg_loss=None
            #loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()
            time_batch = time.time() - t_st  # each epoch train times
            time_total = time_total + time_batch
            del out
        return time_total
        #pbar.close()

    '''
    def test(model, data, layer_loader, device):
        model.eval()
        logits, accs, losses, preds = model.inference(data.x, layer_loader, device), [], [], []

        for _, mask in data('val_mask', 'val_mask', 'val_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses
    '''

    def test(model, data, val_loader, test_loader, device):
        model.eval()
        accs, losses, preds = [], [], []

        logits_train, features = model.inference(data.x, train_loader, device)
        pred = logits_train.argmax(dim=-1)
        #pred = logits_train
        #acc = pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        acc = f1_score(data.y[train_mask].cpu(), pred.cpu(), average='micro')
        #acc = evaluator_wrapper(pred, data.y[train_mask])

        loss = F.nll_loss(logits_train, data.y[train_mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())

        del logits_train

        logits_val, _ = model.inference(data.x, val_loader, device)
        pred = logits_val.argmax(dim=-1)
        #pred = logits_val
        #acc = pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        acc = f1_score(data.y[val_mask].cpu(), pred.cpu(), average='micro')
        #acc = evaluator_wrapper(pred, data.y[val_mask])

        loss = F.nll_loss(logits_val, data.y[val_mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())

        del logits_val

        logits_test, _ = model.inference(data.x, test_loader, device)
        pred = logits_test.argmax(dim=-1)
        #pred = logits_test
        #acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        acc = f1_score(data.y[test_mask].cpu(), pred.cpu(), average='micro')
        #acc = evaluator_wrapper(pred, data.y[test_mask])

        loss = F.nll_loss(logits_test, data.y[test_mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())

        del logits_test

        return accs, preds, losses, features

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

    gat_neighsampler_parameters = {'lr': args.lr,
                                   #'lr': 0.003,
                                   'num_layers': 2,
                                   'hidden_channels': args.hidden,
                                   'dropout': args.dropout,
                                   'dprate': args.dprate,
                                   'batchnorm': True,
                                   #'l2': 5e-7,
                                   'l2': args.weight_decay,
                                   'layer_heads': [3, 1]}
    para_dict = gat_neighsampler_parameters
    model_para = gat_neighsampler_parameters.copy()
    model_para.pop('lr')
    model_para.pop('l2')
    tmp_net = Net(in_channels=data.x.size(-1), out_channels=dataset.num_classes, **model_para)


    total = sum([param.nelement() for param in tmp_net.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))

    #evaluator = Evaluator(name='ogbn-arxiv')
    #tmp_net = Net(dataset, args)

    #randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb,args.seed)


    if (args.dataset == 'ogbn-arxiv'):
        split_idx = dataset.get_idx_split()
        train_mask, val_mask, test_mask = split_idx["train"], split_idx["valid"], split_idx["test"]
    else :
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    #evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]

    '''
    test_features = data.x[data.train_mask].detach().cpu()
    y = data.y[data.train_mask].detach().cpu()
    X_pca = umap.UMAP(n_neighbors=5, min_dist=0.3).fit_transform(test_features)

    colors = ['#F94144', '#277DA1', '#90BE6D', '#F3722C', '#7400B8', '#70D6FF', '#FF70A6', '#0D3B66', '#4D908E', '#577590', '#43AA8B', '#C38E70', '#F9844A', '#FAF0CA', '#E9FF70' ]

    # 创建一个散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[colors[label] for label in y], s=5)

    # 添加标签和标题
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Visualization with Labels')

    plt.savefig('./Figure/'+ args.dataset + '_original', dpi=3000)

    # 显示图形
    #plt.show()
    '''


    train_loader = NeighborSampler(data.edge_index, node_idx=train_mask, sizes=[10, 5], batch_size=35000, shuffle=False, num_workers=1)
    val_loader = NeighborSampler(data.edge_index, node_idx=val_mask, sizes=[10, 5], batch_size=512, shuffle=False, num_workers=1)
    test_loader = NeighborSampler(data.edge_index, node_idx=test_mask, sizes=[10, 5], batch_size=512, shuffle=False, num_workers=1)

    model, data = tmp_net.to(device), data.to(device)
    model.reset_parameters()

    #optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    theta = args.alpha

    time_run=[]
    for epoch in range(args.epochs):

        time_epoch = train(model, optimizer, data, args.dprate, train_loader, device)
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss], features = test(model, data, val_loader, test_loader, device)

        #print(train_acc)
        #print(val_acc)
        #print(tmp_test_acc)

        if val_loss <= best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net =='BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:', epoch)
                    '''
                    test_features = features.detach().cpu()
                    y = data.y[data.train_mask].detach().cpu()
                    X_pca = umap.UMAP(n_neighbors=5, min_dist=0.3).fit_transform(test_features)

                    colors = ['#F94144', '#277DA1', '#90BE6D', '#F3722C', '#7400B8', '#70D6FF', '#FF70A6', '#0D3B66','#4D908E', '#577590', '#43AA8B', '#C38E70', '#F9844A', '#FAF0CA', '#E9FF70']

                    # 创建一个散点图
                    plt.figure(figsize=(8, 6))
                    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[colors[label] for label in y], s=5)

                    # 添加标签和标题
                    plt.xlabel('Principal Component 1')
                    plt.ylabel('Principal Component 2')
                    plt.title('2D PCA Visualization with Labels')

                    plt.savefig('./Figure/'+ args.dataset + '_TSCNet', dpi=3000)

                    # 显示图形
                    #plt.show()
                    '''
                    break
        print(epoch, ": train_loss", train_loss, " val_loss", val_loss, " tmp_train_acc", train_acc, " tmp_test_acc", tmp_test_acc, " test_acc", test_acc)
    return test_acc, best_val_acc, theta, time_run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=500, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=32, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR', help='initialization for GPRGNN.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')

    parser.add_argument('--dataset', type=str, choices=['Cora','Citeseer','Pubmed','Computers','Photo', 'Actor','Texas','Cornell', 'CS', 'Physics', 'washington', 'chameleon', 'Reddit', 'Flickr', 'ogbn-arxiv'],
                        default='PubMed')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=1, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'MLP','GraInc','AFGCN','SIGN','GraIncV2','GraIncV3','FAGCN','SGC','GraphSAGE','ConvG', 'TSCGCN'], default='TSCGCN')
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
    parser.add_argument('--Inc1', type=int, default=1, help='1*1 Receive Field.')
    parser.add_argument('--Inc2', type=int, default=2, help='2*2 Receive Field.')
    parser.add_argument('--Inc3', type=int, default=3, help='3*3 Receive Field.')

    args = parser.parse_args()

    #10 fixed seeds for splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766666,4289935166,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    '''
    gnn_name = args.net
        if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name =='MLP':
        Net = MLP
    elif gnn_name == 'GraInc':
        Net = GraInc
    elif gnn_name == "SGC" :
        Net = SGC_Net
    elif gnn_name == 'AFGCN':
        Net = AFGCN
    elif gnn_name == 'SIGN':
        Net = SIGN
    elif gnn_name == 'GraIncV2':#best
        Net = GraIncV2
    elif gnn_name == "FAGCN":
        Net = FA_NET
    elif gnn_name == "GraphSAGE":
        Net = SAGE_Net
    elif gnn_name == "ConvG":
        Net = ConvG
    '''
    Net = GAT_NeighSampler

    dataset = DataLoader(args.dataset)
    data = dataset[0]

    #data.edge_index = to_undirected(edge_index=data.edge_index, num_nodes=data.num_nodes)
    #print(data)

    percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(args.val_rate*len(data.y)))

    results = []
    time_results=[]
    for RP in tqdm(range(args.runs)):
        args.seed=SEEDS[RP]
        test_acc, best_val_acc, theta_0,time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc, theta_0])
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
        if args.net == 'BernNet':
            print('Theta:', [float('{:.4f}'.format(i)) for i in theta_0])

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    #print(uncertainty*100)
    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
