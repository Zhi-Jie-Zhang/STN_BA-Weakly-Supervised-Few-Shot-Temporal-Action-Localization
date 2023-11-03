import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
import math
from util.function import cat_score, bi_inter
from .linearsq import LinearSQ, LinearSQ_Sample

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=81, feat_dim=128, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, feature, labels):
        """
        Args:
            feature: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = feature.size(0)
        distmat = torch.pow(feature, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # import pdb
        # pdb.set_trace()
        distmat.addmm_(1, -2, feature, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()

        if labels.numel() > labels.size(0):
            mask = labels > 0
        else:
            labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = labels.eq(classes.expand(batch_size, self.num_classes).float())

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value *= labels[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

class FCEncoder(nn.Module):
    def __init__(self, n_feature=2048, out_dim=128):
        super(FCEncoder, self).__init__()
        self.n_featureby2 = int(n_feature / 2)
        self.out_dim = out_dim

        self.query_attention = self_attention(self.n_featureby2)
        self.support_attention = self_attention_support(self.n_featureby2)
        self.fc_f = nn.Linear(self.n_featureby2, self.n_featureby2)
        self.fc1_f = nn.Linear(self.n_featureby2, out_dim)
        self.fc_r = nn.Linear(self.n_featureby2, self.n_featureby2)
        self.fc1_r = nn.Linear(self.n_featureby2, out_dim)
        self.con_1 = nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.con_2 = nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU(True)
        self.dropout_f = nn.Dropout(0.5)
        self.dropout_r = nn.Dropout(0.5)
        self.apply(weights_init)

    def forward(self, inputs, inputs_len, is_query, is_training=True):
        # inputs - batch x seq_len x featSize
        base_x_f = inputs[:, :, self.n_featureby2:]
        base_x_r = inputs[:, :, :self.n_featureby2]

        if is_query:
            base_x_f = self.query_attention(base_x_f, inputs_len)
            base_x_r = self.query_attention(base_x_r, inputs_len)
            x_f, x_r = self.load_feat(base_x_f, base_x_r, is_training)

        else:
            base_x_f = self.support_attention(base_x_f, inputs_len)
            base_x_r = self.support_attention(base_x_r, inputs_len)
            x_f, x_r = self.load_feat(base_x_f, base_x_r, is_training)

        enc = torch.cat((x_f, x_r), -1)
        return enc


    def load_feat(self, base_x_f, base_x_r, is_training):
        x_f = self.relu(self.fc_f(base_x_f))
        x_r = self.relu(self.fc_r(base_x_r))
        if is_training:
            x_f = self.dropout_f(x_f)
            x_r = self.dropout_r(x_r)
        x_f = self.relu(self.fc1_f(x_f))
        x_r = self.relu(self.fc1_r(x_r))
        return x_f, x_r


class Classifier(nn.Module):
    def __init__(self, n_feature=256, num_class=81):
        super(Classifier, self).__init__()
        self.n_featureby2 = int(n_feature/2)
        self.fc_f = nn.Linear(self.n_featureby2, num_class)
        self.fc_r = nn.Linear(self.n_featureby2, num_class)
        self.apply(weights_init)
        self.mul_r = nn.Parameter(data=torch.Tensor(num_class).float().fill_(1))
        self.mul_f = nn.Parameter(data=torch.Tensor(num_class).float().fill_(1))

    def forward(self, inputs):

        base_x_f = inputs[:, :, :self.n_featureby2]
        base_x_r = inputs[:, :, self.n_featureby2:]

        cls_x_f = self.fc_f(base_x_f)
        cls_x_r = self.fc_r(base_x_r)

        tcam = cls_x_r * self.mul_r + cls_x_f * self.mul_f
        return cls_x_f, cls_x_r, tcam


class AttentionGenerator(nn.Module):
    def __init__(self, in_dim=4, bn_flag=True):
        super(AttentionGenerator, self).__init__()

        self.bn_flag = bn_flag

        self.fc = LinearSQ(in_dim, 1)
        if self.bn_flag:
            self.bn = nn.BatchNorm1d(in_dim)

        self.apply(weights_init)

    def forward(self, cmp):
        '''
        :param cmp: [bs, num_class*sample_per_class, length_query, length_sample, d]
        :return: tsm: [bs, num_class*sample_per_class, length_query, length_sample]
        '''
        if self.bn_flag:
            b, s, l, d = cmp.size()
            cmp = cmp.view(-1, l, d)

            mask = self.fc(self.bn(cmp.transpose(1, 2)).transpose(1, 2))
            mask = mask.squeeze(2).view(b,s,l)
        else:
            mask = self.fc(cmp).squeeze(-1)

        return mask


class self_attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.key = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.value = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, q_len):
        query_s = ''
        for i in range(len(query)):
            sam_feat_sub = query[i, :q_len[i]].unsqueeze(0)
            sam_feat_remain = query[i, q_len[i]:].unsqueeze(0)
            sam_feat_sub = self.self_att(sam_feat_sub)
            query_i = torch.cat([sam_feat_sub, sam_feat_remain], dim=1)
            query_s = cat_score(query_i, query_s, 0)
        return query_s


    def self_att(self, query):
        query = query.permute(0, 2, 1)
        batch, cha, dep = query.size()
        q = self.query(query).permute(0, 2, 1)
        k = self.key(query)
        v = self.value(query)

        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(torch.div(attn_matrix, math.sqrt(dep)))  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*query.shape)
        feat = self.gamma * out + query
        feat = feat.permute(0, 2, 1)
        return feat


class self_attention_support(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.key = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.value = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, q_len):
        query_s = ''
        for i in range(len(query)):
            sam_feat_sub = query[i, :q_len[i]].unsqueeze(0)
            sam_feat_remain = query[i, q_len[i]:].unsqueeze(0)
            sam_feat_sub = self.self_att(sam_feat_sub)
            query_i = torch.cat([sam_feat_sub, sam_feat_remain], dim=1)
            query_s = cat_score(query_i, query_s, 0)
        return query_s


    def self_att(self, query):
        query = query.permute(0, 2, 1)
        batch, cha, dep = query.size()
        q = self.query(query).permute(0, 2, 1)
        k = self.key(query)
        v = self.value(query)

        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(torch.div(attn_matrix, math.sqrt(dep)))  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*query.shape)
        feat = self.gamma * out + query
        feat = feat.permute(0, 2, 1)
        return feat




