import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = self.W(input)
        # [batch_size, N, out_features]
        batch_size, N,  _ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layer):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer = layer
        # if self.layer == 1:
        #     self.attentions = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # else:
        #     self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        #     self.attentions_l2 = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        #     # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        # for i, attention in enumerate(self.attentions + self.attentions_l2):
        #     self.add_module('attention_{}'.format(i), attention)

        self.attention_layers = []
        for _ in range(self.layer):
            assert nhid * nheads == nfeat
            self.attention_layers.append([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        for i, attention in enumerate(chain(*self.attention_layers)):
            self.add_module('attention_{}'.format(i), attention)
            
    # def forward(self, x, adj):
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     if self.layer == 1:
    #         x = torch.stack([att(x, adj) for att in self.attentions], dim=2)
    #         x = x.sum(2)
    #         x = F.dropout(x, self.dropout, training=self.training)
    #         return F.log_softmax(x, dim=2)
    #     else:
    #         x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
    #         x = F.dropout(x, self.dropout, training=self.training)
    #         x = F.elu(self.out_att(x, adj))
    #         return F.log_softmax(x, dim=2)

    # def forward(self, x, adj):
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
    #     x = F.dropout(x, self.dropout, training=self.training)

    #     x = torch.cat([att(x, adj) for att in self.attentions_l2], dim=2)
    #     # x = F.dropout(x, self.dropout, training=self.training)
    #     return x

    def forward(self, x, adj):
        for attention_layer in self.attention_layers:
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in attention_layer], dim=2)
        return x



class HeteGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(HeteGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a1 = [nn.Linear(out_features, 1, bias=False) for i in range(4)]
        self.a2 = [nn.Linear(out_features, 1, bias=False) for i in range(4)]
        for i in range(4):
            self.add_module('a1_{}'.format(i), self.a1[i])
            self.add_module('a2_{}'.format(i), self.a2[i])
            nn.init.xavier_uniform_(self.a1[i].weight, gain=1.414)
            nn.init.xavier_uniform_(self.a2[i].weight, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, seq_len, seg_len, gaz_len):
        h = self.W(input)
        # [batch_size, N, out_features]
        batch_size, N,  _ = h.size()

        attentions = []
        # 0 self-loop, 1 dep, 2 word-char, 3 char-lexicon
        for i in range(4):
            middle_result1 = self.a1[i](h).expand(-1, -1, N)
            middle_result2 = self.a2[i](h).expand(-1, -1, N).transpose(1, 2)
            e = self.leakyrelu(middle_result1 + middle_result2)
            attentions.append(e)
        
        mask = torch.zeros(4, N, N, dtype=torch.float32).to(attentions[0])
        mask[0] = torch.eye(N)
        mask[1][seq_len:seq_len + seg_len, seq_len:seq_len + seg_len] = 1 - torch.eye(seg_len)
        mask[2][:seq_len, seq_len:seq_len + seg_len] = 1.0
        mask[2][seq_len:seq_len + seg_len, :seq_len] = 1.0
        mask[3][seq_len + seg_len:, :seq_len] = 1.0
        mask[3][:seq_len, seq_len + seg_len:] = 1.0
        # torch.set_printoptions(profile="full")
        # for i in range(4):
        #     print(mask[i])
        # torch.set_printoptions(profile="default")
        attention = attentions[0] * mask[0].unsqueeze(0) + attentions[1] * mask[1].unsqueeze(0) + \
                    attentions[2] * mask[2].unsqueeze(0) + attentions[3] * mask[3].unsqueeze(0)

        attention = attention.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class HeteGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layer):
        super(HeteGAT, self).__init__()
        self.dropout = dropout
        self.layer = layer
        if self.layer == 1:
            self.attentions = [HeteGraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        else:
            self.attentions = [HeteGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            self.attentions_l2 = [HeteGraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        for i, attention in enumerate(self.attentions + self.attentions_l2):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj, seq_len, seg_len, gaz_len):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, seq_len, seg_len, gaz_len) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.cat([att(x, adj, seq_len, seg_len, gaz_len) for att in self.attentions_l2], dim=2)
        # x = F.dropout(x, self.dropout, training=self.training)
        return x


# if __name__ == "__main__":
#     att = HeteGraphAttentionLayer(300, 60, 0, 1)
#     for name, param in att.named_parameters():
#         print(name, param.size())
#     seq_len = 20
#     x = torch.randn(50, seq_len, 300)
#     att(x, torch.randint(0, 2, (seq_len, seq_len)), 10, 5, 5)

class GCN(nn.Module):
    def __init__(self, in_dim, mem_dim, input_dropout, gcn_dropout, num_layers):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.in_drop = nn.Dropout(input_dropout)
        self.gcn_drop = nn.Dropout(gcn_dropout)
        self.layers = num_layers

        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, unified_feature, adj):
        gcn_inputs = unified_feature
        #gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        #print("MASK",mask.shape)
        # zero out adj for ablation
       
        # adj = torch.zeros_like(adj)
       
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        # print("gcn_inputs",gcn_inputs.shape)
        return gcn_inputs
