from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from gat_conv import GATConv
from tqdm import tqdm
from torch.nn import Linear, Parameter
from torch_geometric.nn.inits import glorot, zeros

class GAT_NeighSampler(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 ,dprate
                 , layer_heads = []
                 , batchnorm=True):
        super(GAT_NeighSampler, self).__init__()

        #self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers

        self.lins = Linear(in_channels, layer_heads[0] * hidden_channels)
        self.convs = GATConv(hidden_channels, hidden_channels, heads=layer_heads[0], concat=True)
        self.line = Linear(layer_heads[0] * hidden_channels, out_channels)

        #self.bias = Parameter(torch.Tensor(out_channels))

        self.temp = Parameter(torch.Tensor(3))

        '''
        if len(layer_heads)>1:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[0]))
            for i in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels*layer_heads[i-1], hidden_channels, heads=layer_heads[i], concat=True))
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[i-1]))

            self.convs.append(GATConv(hidden_channels*layer_heads[num_layers-2]
                              , out_channels
                              , heads=layer_heads[num_layers-1]
                              , concat=False))
        else:
            self.convs.append(GATConv(in_channels, out_channels, heads=layer_heads[0], concat=False))     
        '''

        self.dropout = dropout
        self.dprate = dprate
        
    def reset_parameters(self):
        glorot(self.lins.weight)
        glorot(self.line.weight)
        #self.lins.reset_parameters()
        #self.line.reset_parameters()
        self.convs.reset_parameters()
        self.temp.data.fill_(0.3)
        #zeros(self.bias)
        '''
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()   
        '''
        
    def forward(self, x, adjs, device):

        x_all = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins(x)
        #x += self.bias
        x = F.relu(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        x_all.append(x[:adjs[-1].size[1]])
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            edge_index = edge_index.to(device)
            xx = self.convs((x, x_target), edge_index)
            x_all.append(xx[:adjs[-1].size[1]])

        #temp = F.softmax(self.temp)

        x = self.temp[0] * x_all[0] + self.temp[1] * x_all[1] + self.temp[2] * x_all[2]

        x = self.line(x)
        #x += self.bias

        '''
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        '''

        '''
        edge_index, _, size = adjs[0]
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x_target = x[:size[1]]
        x = self.convs[-1]((x, x_target), edge_index)
        '''

        return x.log_softmax(dim=-1)
    
    '''
    subgraph_loader: size = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=**, shuffle=False,
                                  num_workers=12)
    You can also sample the complete k-hop neighborhood, but this is rather expensive (especially for Reddit). 
    We apply here trick here to compute the node embeddings efficiently: 
       Instead of sampling multiple layers for a mini-batch, we instead compute the node embeddings layer-wise. 
       Doing this exactly k times mimics a k-layer GNN.  
    '''
    
    def inference_all(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
    def inference(self, x_all, layer_loader, device):
        #pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        #pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        #for i in range(self.num_layers):
        '''
            xs = []

            for batch_size, n_id, adj in layer_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm:
                        x = self.bns[i](x)
                xs.append(x)

                #pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)
        '''

        xs = []
        out_all = []

        for batch_size, n_id, adjs in layer_loader:
            adjs = [adj for adj in adjs]
            #edge_index, _, size = adjs.to(device)
            x = x_all[n_id]

            x_al = []

            x = self.lins(x)
            x = F.relu(x)
            #x = F.dropout(x, p=0.5, training=self.training)
            x_al.append(x[:adjs[-1].size[1]])
            #print(adjs[-1].size[1])
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]
                #print(x_target.shape)
                edge_index = edge_index.to(device)
                xx = self.convs((x, x_target), edge_index)
                #print(xx.shape)
                #print(xx[:adjs[-1].size[1]].shape)
                x_al.append(xx[:adjs[-1].size[1]])

            x = self.temp[0] * x_al[0] + self.temp[1] * x_al[1] + self.temp[2] * x_al[2]

            #print(x.shape)

            out = self.line(x)
            #x += self.bias

            xs.append(x)
            out_all.append(out)

        out_all = torch.cat(out_all, dim=0)
        xs = torch.cat(xs, dim=0)

        #print(x_all.shape)
        #pbar.close()

        return out_all.log_softmax(dim=-1), xs



class GATv2_NeighSampler(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , layer_heads = []
                 , batchnorm=True):
        super(GATv2_NeighSampler, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        
        if len(layer_heads)>1:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[0]))
            for i in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels*layer_heads[i-1], hidden_channels, heads=layer_heads[i], concat=True))
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[i-1]))
            self.convs.append(GATv2Conv(hidden_channels*layer_heads[num_layers-2]
                              , out_channels
                              , heads=layer_heads[num_layers-1]
                              , concat=False))
        else:
            self.convs.append(GATv2Conv(in_channels, out_channels, heads=layer_heads[0], concat=False))        

        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()        
        
        
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                
        return x.log_softmax(dim=-1)
    
    '''
    subgraph_loader: size = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=**, shuffle=False,
                                  num_workers=12)
    You can also sample the complete k-hop neighborhood, but this is rather expensive (especially for Reddit). 
    We apply here trick here to compute the node embeddings efficiently: 
       Instead of sampling multiple layers for a mini-batch, we instead compute the node embeddings layer-wise. 
       Doing this exactly k times mimics a k-layer GNN.  
    '''
    
    def inference_all(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
    def inference(self, x_all, layer_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm: 
                        x = self.bns[i](x)
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all.log_softmax(dim=-1)
