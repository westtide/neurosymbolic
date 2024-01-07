import json

import torch
from torch import nn, tensor
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.ExternalProcesses.CFG_parser import GetAllCGraphfilePath
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule
from code2inv.common.ssa_graph_builder import ProgramGraph
from code2inv.graph_encoder.embedding import EmbedMeanField
from code2inv.prog_generator.file_solver import GraphSample
from loginit import logger


class CFG_Embedding(nn.Module):
    """
    继承自 PyTorch 的 nn.Module 类。这个类用于为控制流图（Control Flow Graph，CFG）创建嵌入向量
    """

    def __init__(self, cfg):
        super().__init__()

        # Need to prepare node type dict from the beginning.
        node_type_dict = {}
        allgpaths = GetAllCGraphfilePath()
        for gpath in allgpaths:
            graph_file = open(gpath, 'r')
            graph = ProgramGraph(json.load(graph_file))  # 对于每个 CFG 文件，将其打开并作为 ProgramGraph 对象加载
            for node in graph.node_list:  # 对于每个 CFG 文件中的每个节点，将其类型添加到 node_type_dict 中
                if not node.node_type in node_type_dict:
                    v = len(node_type_dict)
                    node_type_dict[node.node_type] = v
        # CFG 的编码器: 节点特征大小 = 128, 长度和最大级别 10
        self.encoder = EmbedMeanField(config.SIZE_EXP_NODE_FEATURE, len(node_type_dict), max_lv=10)
        # 注意力向量 attvec: 128
        self.attvec = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
        # softmaxer: 1 标准化注意力权重
        self.softmaxer = nn.Softmax(dim=1)
        # GraphSample: 用于从 CFG 中采样
        self.g_list = GraphSample(cfg, [], node_type_dict)

    def forward(self, emb_smt, emb_CE, stateVec):
        # 采样后的 CFG 使用编码器编码, 得到 CFG 的嵌入 emd
        self.cfg_emb = self.encoder(self.g_list)
        # 计算加权的 CFG 嵌入向量和原始的 CFG 嵌入向量的矩阵乘积，得到最终的 CFG 嵌入向量
        weighted1 = torch.mm(self.cfg_emb, stateVec.transpose(0, 1)).transpose(0, 1)
        cfg_emb = torch.mm(weighted1, self.cfg_emb)
        # 计算 CFG 嵌入向量、SMT 公式的嵌入向量和反例的嵌入向量与注意力向量的余弦相似度，得到它们的权重
        weis = torch.cat([torch.cosine_similarity(cfg_emb, self.attvec),
                          torch.cosine_similarity(emb_smt, self.attvec),
                          torch.cosine_similarity(emb_CE, self.attvec)], 0).reshape([1, 3])
        # 对权重进行 softmax 归一化，得到归一化的权重
        swis = self.softmaxer(weis)
        # 三合一: 将 CFG 嵌入向量、SMT 公式的嵌入向量和反例的嵌入向量拼接在一起，得到一个大的嵌入向量
        three_emb = torch.cat((cfg_emb, emb_smt, emb_CE), 0).reshape([3, config.SIZE_EXP_NODE_FEATURE])
        # 计算归一化的权重和大的嵌入向量的矩阵乘积，得到最终的特征向量
        overall_feature = torch.mm(swis, three_emb)
        logger.info(f'CFG_Embedding 参数: emb_smt = {emb_smt}, emb_CE = {emb_CE}')
        logger.info(f'CFG_Embedding 结果: self.cfg_emb = {self.cfg_emb}, weis = {weis}, swis = {swis}, three_emb = {three_emb}')
        return overall_feature

    def GetParameters(self):
        res = {}
        PreFix = "CFG_Embedding_P_"
        res[PreFix + "attvec"] = self.attvec
        res.update(getParFromModule(self.encoder, prefix=PreFix + "encoder"))
        return res

    def cudalize(self):
        self.attvec = Parameter(self.attvec.cuda())
        self.encoder = self.encoder.cuda()
