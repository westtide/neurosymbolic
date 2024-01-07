from z3 import If
class Config:
    SELECT_AN_ACTION = 0            # 选择一个动作
    SET_AN_VALUE = 1                # 设置一个值

    SIZE_EXP_NODE_FEATURE = 128     # 节点特征的大小?
    SIZE_PCA_NUM = 50               # PCA 的维度?

    MAX_DEPTH = 150 # 最大深度?
    BEST = False # 是否是最好的?

    CONTINUE_TRAINING = True # 是否继续训练?

    LinearPrograms = ["Problem_L" + str(i) for i in range(1,134)] # 线性程序
    NonLinearPrograms = ["Problem_NL" + str(i) for i in range(1,31)] # 非线性程序

    LearningRate = 1e-2 # 学习率

config = Config()

def Z3_abs(x):
    return If(x >= 0,x,-x)