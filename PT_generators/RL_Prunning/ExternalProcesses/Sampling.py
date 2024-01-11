import numpy as np

from PT_generators.RL_Prunning.Conifg import config


def makeitsumtoone(dist):
    lister = [float(x) for x in list(dist[0])]
    sumer = sum(lister)
    lister = [x/sumer for x in lister]
    return lister

def sampling(action_dirtibution, available_acts, best=config.BEST):
    if best:
        # 最大概率选择, 遍历, 找概率最大的动作
        id = -1
        maxvalue = 0
        i = 0
        for dis in action_dirtibution[0]:
            if float(dis) >= maxvalue:
                id = i
                maxvalue = float(dis)
            i+=1
        return available_acts[id]
    try:
        # 概率加权随机选择
        # makeitsumtoone 是将概率分布归一化
        # np.random.choice 是从 available_acts 中以概率 action_dirtibution 选择一个动作
        # 在强化学习中，根据任务的不同阶段和目标，选择适当的策略是非常重要的。
        # 在早期阶段，更多的探索有助于更好地了解环境；而在后期，更倾向于利用已知的最佳策略。
        return np.random.choice(available_acts, p=makeitsumtoone(action_dirtibution))
    except Exception as e:
        print("shit", e)
        raise e


