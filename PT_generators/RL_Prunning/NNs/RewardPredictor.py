import torch
from torch import nn, tensor

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(config.SIZE_EXP_NODE_FEATURE * 2, config.SIZE_EXP_NODE_FEATURE)
        self.layer2 = nn.Linear(config.SIZE_EXP_NODE_FEATURE, config.SIZE_EXP_NODE_FEATURE // 2)
        self.layer3 = nn.Linear(config.SIZE_EXP_NODE_FEATURE // 2, 1)

    def forward(self, stateVec, overall_feature):
        # 两个tensor 在第二个维度上进行拼接, 得到一个新的 tensor
        tensorflow = torch.cat([stateVec, overall_feature], 1).clone().detach()
        if torch.cuda.is_available():
            tensorflow = tensorflow.cuda()
        if torch.cuda.is_available():
            tensorflow = tensorflow.cuda()
        # 第一层
        l1out = self.layer1(tensorflow)
        # 常数张量，分别表示 -10 和 10
        m10 = tensor([[-10]])
        p10 = tensor([[10]])
        if torch.cuda.is_available():
            m10 = m10.cuda()
            p10 = p10.cuda()
        # 先将网络输出与 -10 拼接，然后使用 torch.max 确保输出不低于 -10
        # 将这个结果与 10 拼接，使用 torch.min 确保输出不高于 10
        # 限制输出在 -10 到 10 的范围内
        return torch.min(torch.cat([torch.max(torch.cat([self.layer3(self.layer2(l1out)), m10], 1)).reshape(1,1), p10], 1)).reshape(1,1)

    def GetParameters(self):
        res = {}
        PreFix = "RewardPredictor_P_"
        res.update(getParFromModule(self.layer1, prefix=PreFix + "layer1"))
        res.update(getParFromModule(self.layer2, prefix=PreFix + "layer2"))
        res.update(getParFromModule(self.layer3, prefix=PreFix + "layer3"))

        return res

    def cudalize(self):
        self.layer1 = self.layer1.cuda()
        self.layer2 = self.layer2.cuda()
        self.layer3 = self.layer3.cuda()


# little test

if __name__ == "__main__":
    stateVec = torch.randn([1, config.SIZE_EXP_NODE_FEATURE])
    overall_feature = torch.randn([1, config.SIZE_EXP_NODE_FEATURE])
    rp = RewardPredictor()
    print(rp(stateVec, overall_feature))
