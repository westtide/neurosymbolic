import pickle

import torch
from torch import tensor
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.CFG_Embedding import CFG_Embedding
from PT_generators.RL_Prunning.NNs.CounterExampleEmbedding import CEEmbedding
from PT_generators.RL_Prunning.NNs.DistributionLize import DistributionLize
from PT_generators.RL_Prunning.NNs.IntLize import IntLize
from PT_generators.RL_Prunning.NNs.PolicyNetwork import PolicyNetwork
from PT_generators.RL_Prunning.NNs.RewardPredictor import RewardPredictor
from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings
from PT_generators.RL_Prunning.NNs.TreeLSTM import TreeLSTM
from PT_generators.RL_Prunning.TemplateCenter.TemplateCenter import RULE
from loginit import logger


def constructT():
    treeLSTM = TreeLSTM()
    return treeLSTM

def constructG(cfg):
    return CFG_Embedding(cfg)


def constructE(vars):
    return CEEmbedding(vars)

def constructP():
    return RewardPredictor()

def constructpi(ptg):
    """
    通过 self.pi = constructpi(self) 初始化 PolicyNetwork
    实际上是在执行 PolicyNetwork 的 forward 方法，将当前的状态向量和整合特征传入网络，生成动作向量
    """
    return PolicyNetwork(ptg, GetProgramFearture)

def construct_distributionlize():
    return DistributionLize()

def construct_intValuelzie():
    return IntLize()


def init_symbolEmbeddings():
    """
    对于每个非终端符号和它的每个动作，以及每个可能的深度，它都分配一个随机生成的参数张量作为它的嵌入。
    """
    Rule_keys = RULE.keys()

    # 为每个非终结符号生成参数张量
    for non_terminal in Rule_keys:
        SymbolEmbeddings[non_terminal] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
        actions = RULE[non_terminal]
        # 为非终结符号的每个动作也生成参数张量
        for act in actions:
            SymbolEmbeddings[str(act)] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)

    # 线性的程序
    for problems in config.LinearPrograms:
        # 在每个可能的深度上分配一个随机生成的参数张量
        for depth in range(config.MAX_DEPTH):
            SymbolEmbeddings[problems + "_" + str(depth)] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)

    # 非线性的程序
    for problems in config.NonLinearPrograms:
        for depth in range(config.MAX_DEPTH):
            SymbolEmbeddings[problems + "_" + str(depth)] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)

def GetProgramFearture(path2CFile, depth):
    """
    根据给定的文件路径和深度，返回对应的程序特征
    """
    problemID = path2CFile.split('/')[-1].split('.')[0]
    if 'NL' in problemID:
        problemStr = "Problem_NL" + problemID.split('NL')[-1]
    else:
        problemStr = "Problem_L" + problemID
    try:
        # logger.info(f'problemStr = {problemStr}')
        # logger.info(f'SymbolEmbeddings = {SymbolEmbeddings[problemStr + "_" + str(depth)]}')
        return SymbolEmbeddings[problemStr + "_" + str(depth)]
    except:
        return SymbolEmbeddings['?']

def GPUlizeSymbols():
    """
     参数的CUDA迁移和封装: 将 SymbolEmbeddings 中的所有元素转移到 GPU
    """
    for keyname in SymbolEmbeddings.keys():
        # Parameter封装，这些向量被标记为模型的参数，这意味着在模型训练过程中，它们将被优化器优化
        SymbolEmbeddings[keyname] = Parameter(SymbolEmbeddings[keyname].cuda())

def initialize_paramethers(path):
    """根据给定的路径初始化参数"""
    if "NL" in path:
        ppPath = r"code2inv/templeter/NL_initial.psdlf"
    else:
        ppPath = r"code2inv/templeter/L_initial.psdlf"
    with open(ppPath, 'rb') as f:
        dict = pickle.load(f)
        return dict


def GetActionIndex(last_left_handle,last_action):
    """
    返回给定的最后一个左句柄和最后一个动作在规则中的索引。如果在 CUDA 可用的情况下，返回的张量将在 CUDA 上
    Args:
        last_left_handle:
        last_action:

    Returns:

    """
    for i, action in enumerate(RULE[str(last_left_handle)]):
        if str(action) == str(last_action):
            if torch.cuda.is_available():
                return tensor([i]).cuda()
            else:
                return tensor([i])

    assert False # should not be here





