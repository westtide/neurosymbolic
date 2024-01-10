import torch
from torch import nn, tensor

import numpy as np
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


def pca(X, k):
    """
    主成分分析
    Args:
        X: 反例
        k: 主成分数量
    Returns: 返回降维后的数据
    """
    n_samples, n_features = X.shape     # 样本数量, 特征数量
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])      # 计算每个特征的平均值
    # 减去平均值, 实现归一化
    norm_X = X - mean
    # 计算散布矩阵，这是 PCA 的一个关键步骤, transpose是转置矩阵, scatter_matrix是每一对变量之间的协方差（或者说是相关性）的矩阵表示
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # 计算散布矩阵的特征值和特征向量,特征值越大，表示这个方向的数据变化越大，也就是这个方向的信息量越大。特征向量则表示了这个方向在原始数据空间中的方向
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    # 将特征值和特征向量配对
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # 按照特征值的大小降序排序这些配对
    eig_pairs.sort(reverse=True, key= lambda x:x[0])
    # 选择前 k 个特征向量
    while len(eig_pairs) < k: # 如果特征向量的数量小于 k，则添加额外的零向量
        eig_pairs.append((0, np.transpose(np.matrix([0]*len(eig_pairs[0][1])))))
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # 将归一化后的数据投影到这 k 个特征向量上，得到降维后的数据
    data = np.dot(norm_X, np.transpose(feature))
    return data


class CEEmbedding(nn.Module):
    def __init__(self, vars):
        super().__init__()
        self.vars = vars    # 调用父类的初始化函数
        self.RNNs = {}      # 初始化 RNNs 字典, 用于保存 LSTM 网络
        for keyer in ['p', 'n', 'i_1', 'i_2']: # 3 类反例
            self.RNNs['CE_' + keyer ] = nn.LSTM(config.SIZE_PCA_NUM, config.SIZE_EXP_NODE_FEATURE, 2)
        # 注意力向量 attvec: 128, 用于计算注意力权重, 先初始化, 需要进行梯度下降
        self.attvec = Parameter(torch.randn((1,config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
        # 创建一个 softmax 函数，用于在第一个维度上对输入进行归一化
        self.softmaxer = nn.Softmax(dim=1)

    def matxlize(self, lister):
        """
        将输入的列表转换为一个矩阵
        Args:
            lister: 输入的列表
        Returns:矩阵

        """
        seq = []
        for example in lister:
            seq_1 = []
            for varer in self.vars:
                if varer in example:
                    seq_1.append(int(str(example[varer])))  # 将反例中的每个变量转换为整数
                else:
                    seq_1.append(0)
            seq.append(seq_1)
        if len(seq) == 0:
            seq.append([0])
        return np.matrix(seq)

    def matxlize_inductive(self, lister):
        """
        处理反例中的归纳变量: pre 和 post
        Args:
            lister:
        Returns:
        """
        seq_pre = []
        seq_post = []
        for example in lister:
            e_pre, e_post = example
            seq_1 = []
            seq_2 = []
            for varer in self.vars:
                if varer in e_pre:
                    seq_1.append(int(str(e_pre[varer])))
                else:
                    seq_1.append(0)

                if varer in e_post:
                    seq_2.append(int(str(e_post[varer])))
                else:
                    seq_2.append(0)
            seq_pre.append(seq_1)
            seq_post.append(seq_2)

        if len(seq_pre) == 0:
            seq_pre.append([0])
        if len(seq_post) == 0:
            seq_post.append([0])
        return np.matrix(seq_pre), np.matrix(seq_post)

    def forward(self, CE):
        matx_p = self.matxlize(CE['p'])     # positive CE 违反可达性的反例
        matx_n = self.matxlize(CE['n'])     # negative CE 违反安全性的反例
        matx_i1, matx_i2 = self.matxlize_inductive(CE['i']) # 违反归纳性的反例

        # PCA 分析
        pca_p = tensor(pca(matx_p, config.SIZE_PCA_NUM), dtype=torch.float32).reshape([-1,1,config.SIZE_PCA_NUM])
        pca_n = tensor(pca(matx_n, config.SIZE_PCA_NUM), dtype=torch.float32).reshape([-1,1,config.SIZE_PCA_NUM])
        pca_i1, pca_i2 = tensor(pca(matx_i1, config.SIZE_PCA_NUM), dtype=torch.float32).reshape([-1,1,config.SIZE_PCA_NUM]), \
                         tensor(pca(matx_i2, config.SIZE_PCA_NUM), dtype=torch.float32).reshape([-1,1,config.SIZE_PCA_NUM])
        if torch.cuda.is_available():
            pca_p = pca_p.cuda()
            pca_n = pca_n.cuda()
            pca_i1, pca_i2 = pca_i1.cuda(), pca_i2.cuda()

        # 使用 LSTM 网络对 p_emb 张量进行处理 (p_emb 是一个三维张量)
        p_emb,_ = self.RNNs['CE_p'](pca_p)
        p_emb = p_emb[-1]
        n_emb,_ = self.RNNs['CE_n'](pca_n)
        n_emb = n_emb[-1]
        i1_emb,_ = self.RNNs['CE_i_1'](pca_i1)
        i1_emb = i1_emb[-1]
        i2_emb,_ = self.RNNs['CE_i_2'](pca_i2)
        i2_emb = i2_emb[-1]

        # 计算 p_emb、n_emb、i1_emb 和 i2_emb 张量与注意力向量的余弦相似度，然后将这些相似度拼接在一起，并调整其形状
        weis = torch.cat([torch.cosine_similarity(p_emb, self.attvec),
                       torch.cosine_similarity(n_emb, self.attvec),
                       torch.cosine_similarity(i1_emb, self.attvec),
                       torch.cosine_similarity(i2_emb, self.attvec)], 0).reshape([1,4])
        swis = self.softmaxer(weis)
        three_emb = torch.cat((p_emb, n_emb, i1_emb, i2_emb), 0).reshape([4, config.SIZE_EXP_NODE_FEATURE])
        ce_emb = torch.mm(swis, three_emb)

        return ce_emb


    def GetParameters(self):
        res = {}
        PreFix = "CounterExample_P_"
        res[PreFix + "attvec"] = self.attvec
        for ky in self.RNNs.keys():
            res.update(getParFromModule(self.RNNs[ky], prefix=PreFix + str(ky)))
        return res

    def cudalize(self):
        self.attvec = Parameter(self.attvec.cuda())
        for ky in self.RNNs.keys():
            self.RNNs[ky] = self.RNNs[ky].cuda()


#unit test

if __name__ == "__main__":
    vars = ['x', 'y', 'z']
    CE = {
        'p': [{'x': 1, 'y': 2, 'z': 3},
              {'x': 4, 'y': 7, 'z': 2},
              {'x': 3, 'y': 4, 'z': 8},
              {'x': 11, 'y': 4, 'z': 8}],
        'n': [{'x': 1, 'y': 2, 'z': 3},
              {'x': 1, 'y': 2, 'z': 3},
              {'x': 1, 'y': 2, 'z': 3}],
        'i': [[{'x': 1, 'y': 2, 'z': 3},
               {'x': 1, 'y': 2, 'z': 3}],
              [{'x': 1, 'y': 2, 'z': 3},
               {'x': 1, 'y': 2, 'z': 3}]]
    }
    C = CEEmbedding(vars)
    print(C(CE))