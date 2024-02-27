import torch
import z3
from torch import tensor, optim
from torch.optim import Adam

from PT_generators.RL_Prunning.Conifg import *
from PT_generators.RL_Prunning.ExternalProcesses.CFG_parser import parseCFG
from PT_generators.RL_Prunning.ExternalProcesses.SMT_parser import parseSMT
from PT_generators.RL_Prunning.ExternalProcesses.Sampling import sampling
from PT_generators.RL_Prunning.NNs.NeuralNetwork import *
from PT_generators.RL_Prunning.TemplateCenter.TemplateCenter import InitPT, getLeftHandle, init_varSelection, \
    AvailableActionSelection, update_PT_rule_selction, update_PT_value, ShouldStrict, StrictnessDirtribution, const_ID, \
    simplestAction, init_constSelection, LossnessDirtribution, init_PT_Rules
from PT_generators.StaticAnalysis.AstAnalysis import get_loop_var
from PT_generators.StaticAnalysis.PrePostAnalysis import infer_inv_from_pre_post
from Utilities.Cparser import get_varnames_from_source_code, get_consts_from_source_code
import torch.nn.functional as F

from loginit import logger





class PT_generator:
    def __init__(self, path2CFile, path2CFG, path2SMT):
        """
        PT包含: LR学习率,C,CFG,SMT,vars,consts,T,G,E,P,pi,distributionlize,intValuelzie,depth,stateVec
        Args:
            path2CFile: C 文件的路径
            path2CFG: CFG 文件的路径
            path2SMT: SMT-LIB 文件的路径
        """
        logger.info(f'start: PT_generator')
        logger.info(f'PT_generator: path2CFile = {path2CFile}, path2CFG = {path2CFG}, path2SMT = {path2SMT} ')

        self.LR = config.LearningRate
        # Step1. Parse the inputs
        self.cfg = parseCFG(path2CFG)
        self.smt = parseSMT(path2SMT)
        self.path2CFile = path2CFile

        # 遍历 ast 获得所有的变量
        self.vars = get_varnames_from_source_code(self.path2CFile)
        logger.info(f'PT_generator: vars = {self.vars} ')

        # 遍历 ast 获得所有的常数
        self.consts = get_consts_from_source_code(self.path2CFile)
        logger.info(f'PT_generator: consts = {self.consts} ')

        # 初始化变量的选择规则
        init_varSelection(self.vars)

        # 初始化常量的选择规则
        init_constSelection(self.consts)

        # 初始化符号的 Embedding
        init_symbolEmbeddings()

        # Step2. Construct the NNs and Load the parameters
        self.T = constructT()                                   # TreeLSTM
        self.G = constructG(self.cfg)                           # CFG_Embedding
        self.E = constructE(self.vars)                          # CEEmbedding
        self.P = constructP()                                   # RewardPredictor
        self.pi = constructpi(self)                             # PolicyNetwork
        self.distributionlize = construct_distributionlize()    # DistributionLize
        # self.intValuelzie = construct_intValuelzie()

        # Step3. Init the learner and parameters 初始化学习器和参数
        self.init_learner_par()

        # if config.CONTINUE_TRAINING:
        #     self.load_parameters(config.ppath)
        self.init_parameters()

        # if we can use gpu
        if torch.cuda.is_available():
            self.gpulize()

        ########################################## 新增 ##########################################
        self.element_counter = {
            'var': {
                'sum': 0,
            },
            'constant': {
                'sum': 0,
            },
            'coefficients': {
                'sum': 0,
            },
            'operators': {
                'sum': 0,
            },
            'conjunctions': {
                'sum': 0,
            },
            'disjunctions': {
                'sum': 0,
            },
            'clauses': {
                'sum': 0,
            }
        }

        init_PT_Rules(self.element_counter)
        loop_variables = []
        # loop_variables.append(get_loop_var(path2CFile, path2CFG))

        infer_inv = []
        # infer_inv.append(infer_inv_from_pre_post(path2CFile, path2CFG))

        ########################################## 新增 ##########################################



    def generate_next(self, CE):
        """
        根据当前的例子生成一个新的部分模板
        :param CE: [正例,反例,归纳例]
        """
        self.depth = 0
        PT = InitPT()
        self.stateVec = self.T(PT)
        # the lists will be used when punish or prised.
        predicted_reward_list = []
        action_selected_list = []
        outputed_list = []
        action_or_value = []
        left_handles = []
        emb_CE = self.E(CE)
        self.emb_smt = self.T.forward_three(self.smt)
        left_handle = getLeftHandle(PT)
        while left_handle is not None:
            # 根据给定的 C 文件、CFG 文件和 SMT-LIB 文件，生成一个部分模板 PT
            left_handles.append(left_handle)
            # 根据当前的 left_handle 选择可用的动作或值
            # act_or_val = config.SELECT_AN_ACTION, available_acts = RULE[str(left_handle.decl())]
            act_or_val, available_acts = AvailableActionSelection(left_handle)
            # 整合来自不同源的特征
            overall_feature = self.G(self.emb_smt, emb_CE, self.stateVec)
            # 根据当前的状态向量和整合后的特征，预测奖励值
            predicted_reward = self.P(self.stateVec, overall_feature)
            predicted_reward_list.append(predicted_reward)
            # 使用策略网络（Policy Network）根据当前的状态向量和整合特征生成一个动作向量
            action_vector = self.pi(self.stateVec, overall_feature)
            # 判断是选择动作还是值
            if act_or_val == config.SELECT_AN_ACTION:
                # SELECT_AN_ACTION = 0
                # 根据动作向量和可用动作生成一个动作分布
                action_dirtibution, action_raw = self.distributionlize(action_vector, available_acts)
                # 从动作分布中采样一个动作
                action_selected = sampling(action_dirtibution, available_acts)
                # 如果深度达到了最大值 (config.MAX_DEPTH)，则选择一个简单的动作来避免过深递归
                if self.depth >= config.MAX_DEPTH:
                    # 安全检查: 防止递归过深, 这里 MAX_DEPTH = 150
                    # 递归太深了, 选择一个简单的动作来避免过深递归
                    action_selected = simplestAction(left_handle)
                action_selected_list.append(action_selected)
                outputed_list.append(action_raw)
                # 根据所选动作更新部分模板
                PT = update_PT_rule_selction(PT, left_handle, action_selected)
                # logger.info(f'depth = {self.depth}, action_selected_list = {action_selected_list}')
            else:
                assert False
                # should not be here now
                # value = self.intValuelzie(action_vector, left_handle)
                # value_of_int = int(value)
                # action_selected_list.append(value_of_int)
                # outputed_list.append(value)
                #
                # PT = update_PT_value(PT, left_handle, value_of_int)

            action_or_value.append(act_or_val)
            # 更新 left_handle 为部分模板中的下一个待处理元素
            left_handle = getLeftHandle(PT)
            # 更新状态向量为新的部分模板的状态
            self.stateVec = self.T(PT)
            self.depth += 1

        self.last_predicted_reward_list = predicted_reward_list
        self.last_action_selected_list = action_selected_list
        self.last_outputed_list = outputed_list
        self.last_action_or_value = action_or_value
        self.last_left_handles = left_handles
        return PT

    def punish(self, SorL, Deg, Whom):
        gama = 0    # 折扣因子
        reward = 0  # 负奖励
        if Deg == "VERY":
            reward = -10
            gama = 0.1
        elif Deg == "MEDIUM":
            reward = -5
            gama = 0.05
        elif Deg == "LITTLE":
            reward = -1
            gama = 0.01
        assert gama != 0
        assert reward != 0

        strict_loss = tensor([[0]], dtype=torch.float32)
        if torch.cuda.is_available():
            strict_loss = strict_loss.cuda()
        counter = 0
        for i in range(len(self.last_action_or_value)):
            if ShouldStrict(self.last_left_handles[i], Whom):
                if self.last_action_or_value[i] == config.SELECT_AN_ACTION:
                    if SorL == 'STRICT':
                        SD = StrictnessDirtribution(self.last_left_handles[i], Whom)
                    else:
                        assert SorL == 'LOOSE'
                        SD = LossnessDirtribution(self.last_left_handles[i], Whom)
                    # 基于所选择的损失分布 SD 和历史行动的输出值计算严格损失
                    # 损失分布 SD 和智能体选择的动作的原始输出向量 self.last_outputed_list[i]
                    # 先转换为一维张量, 再使用适用于多分类问题的对数softmax,将其转换为概率分布
                    # 两矩阵相乘, 计算损失分布和行动输出概率分布的负交叉熵损失, 用于衡量两个概率分布之间的差异
                    # 乘以折扣因子 gama, 折扣因子用于调整损失的影响，取负值则是因为优化过程通常是在寻找损失函数的最小值，
                    # 而在强化学习中，我们希望奖励（即负损失）尽可能大
                    Loss_strictness = -torch.mm(SD, torch.log_softmax(self.last_outputed_list[i].reshape(1, -1), 1).transpose(0,
                                                                                                                   1)) * gama
                else:
                    assert False  # should not be here
                    # Loss_strictness = F.mse_loss(self.last_outputed_list[i],
                    #                              torch.tensor([1], dtype=torch.float32)) * gama / 4

                strict_loss += Loss_strictness.reshape([1, 1])
                counter += 1
        if counter != 0:
            strict_loss /= counter
        a_loss = self.ALoss(reward)
        self.LearnStep((a_loss + strict_loss))

    def ALoss(self, final_reward):
        # 初始设置
        discounter = 0.95   # 折扣因子, 优先考虑即时奖励而非远期奖励
        reward_list = []
        for i in range(len(self.last_predicted_reward_list)):
            # reward_list 用于存储折扣后的奖励, 计算折扣后的奖励添加到其中, 并翻转
            reward_list.append(final_reward * discounter ** i)
        reward_list = reward_list[::-1]
        p_loss = 0
        for i in range(len(self.last_predicted_reward_list)):   # 遍历
            r_i = reward_list[i]                                # 暂存

            if i == 0:                                          # 初始的预测奖励值, 形状是 (1, 1)
                pr_i_1 = tensor([[0]], dtype=torch.float32)
            else:
                # pr_i_1 = self.last_predicted_reward_list[i - 1] 前一个预测奖励值就设为第i-1个样本的奖励
                pr_i_1 = tensor([reward_list[i - 1]], dtype=torch.float32)

            # 交叉熵损失函数
            if self.last_action_or_value[i] == config.SELECT_AN_ACTION:
                losser = F.cross_entropy(self.last_outputed_list[i].reshape(1, -1),
                                         GetActionIndex(self.last_left_handles[i], self.last_action_selected_list[i]))
            else:
                assert False

            # 检查是否可以使用CUDA: 计算p_loss
            if torch.cuda.is_available():
                # 如果可以使用 CUDA 在GPU上计算损失，并累加到总损失上
                p_loss += (tensor(r_i, dtype=torch.float32) - pr_i_1).cuda() * losser.reshape([1, 1])
                # logger.info(f'ALoss: CUDA is available, p_loss = {p_loss} ')
            else:
                p_loss += (tensor(r_i, dtype=torch.float32) - pr_i_1) * losser.reshape([1, 1])
                # logger.info(f'ALoss: CUDA is not available, p_loss = {p_loss} ')

        p_loss = p_loss / len(reward_list)
        # logger.info(f'ALoss: p_loss = {p_loss} ')

        # 检查是否可以使用CUDA: 均方误差损失函数
        if torch.cuda.is_available():
            mse_loss = F.mse_loss(tensor([reward_list], dtype=torch.float32).cuda(),
                                  torch.cat(self.last_predicted_reward_list, 1)).reshape([1, 1])
        else:
            mse_loss = F.mse_loss(tensor([reward_list], dtype=torch.float32),
                                  torch.cat(self.last_predicted_reward_list, 1)).reshape([1, 1])
        logger.info(f'ALoss: mse_loss = {mse_loss} ')
        # print("mse_loss", mse_loss)
        # print("p_loss", p_loss)
        return (p_loss + mse_loss)

    def prise(self, Deg):
        """
        根据输入的 Deg 参数来设定奖励值，然后调用 ALoss 方法计算损失，最后调用 LearnStep 方法进行学习步骤
        """
        if Deg == "VERY":
            reward = 10
        elif Deg == "LITTLE":
            reward = 1
        else:
            reward = 0
        a_loss = self.ALoss(reward) # 调用 ALoss 方法计算损失
        self.LearnStep(a_loss)      # 调用 LearnStep 方法进行学习步骤

    def LearnStep(self, loss):
        # if torch.cuda.is_available():
        #     loss = loss.cuda()
        self.adam.zero_grad()       # 调用优化器的 zero_grad 方法将梯度归零
        # print(loss)
        loss.backward()             # 调用 loss 的 backward 方法进行反向传播
        # if torch.cuda.is_available():
        #     loss = loss.cpu()
        self.adam.step()            # 调用优化器的 step 方法对参数进行更新

    def init_learner_par(self):

        paras = {}
        paras.update(SymbolEmbeddings)
        paras.update(self.T.GetParameters())
        paras.update(self.G.GetParameters())
        paras.update(self.E.GetParameters())
        paras.update(self.P.GetParameters())
        paras.update(self.pi.GetParameters())
        # paras.update(self.distributionlize.GetParameters())
        # paras.update(self.intValuelzie.GetParameters())
        for parname in paras:
            paras[parname].requires_grad = True

        self.adam = Adam(paras.values(), lr=self.LR)
        self.paras = paras

    def init_parameters(self):
        paradict = initialize_paramethers(self.path2CFile)
        for parname in self.paras:
            if parname in paradict:  # initialize
                assert self.paras[parname].shape == paradict[parname].shape
                self.paras[parname].data = paradict[parname].data

    def gpulize(self):
        self.T.cudalize()
        self.G.cudalize()
        self.E.cudalize()
        self.P.cudalize()
        self.pi.cudalize()
        GPUlizeSymbols()
