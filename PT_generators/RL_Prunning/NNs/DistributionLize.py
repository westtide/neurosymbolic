import torch
from torch import nn, tensor

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings


class DistributionLize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, action_vector, available_acts):
        # 前向传播过程
        # construt the available action vectors
        # 将每个可用动作的符号嵌入与动作向量进行矩阵乘法并在第一个维度上拼接得到的
        # SymbolEmbeddings[str(x)]是将动作x转换为其对应的符号嵌入
        # action_vector.transpose(0, 1)是将动作向量转置
        # torch.mm(SymbolEmbeddings[str(x)], action_vector.transpose(0, 1))是将符号嵌入与动作向量进行矩阵乘法
        # for x in available_acts] 这是一个列表推导式，它对 available_acts 中的每个元素 x 执行上述的矩阵乘法操作，并将结果收集到一个列表中
        # torch.cat 将上述列表中的所有结果在第一个维度（dim=1）上进行拼接
        rawness = torch.cat([torch.mm(SymbolEmbeddings[str(x)], action_vector.transpose(0, 1)) for x in available_acts],1)
        likenesses = torch.softmax(rawness,1)
        return (likenesses, rawness)

    def GetParameters(self):
        res = {}

        return res



