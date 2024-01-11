import torch
from torch import tensor
from z3 import *

from PT_generators.RL_Prunning.Conifg import config
from loginit import logger

# RULE = {
#     'non_nc': [And(Bool('non_nd')), And(Bool('non_nd'), Bool('non_nd')), And(Bool('non_nd'), Bool('non_nd'), Bool('non_nd'))],
#     'non_nd': [Or(Bool('non_p')), Or(Bool('non_p'), Bool('non_p')), Or(Bool('non_p'), Bool('non_p'), Bool('non_p'))],
#     'non_p': [Int('non_t') < Int('non_s'),
#               Int('non_t') <= Int('non_s'),
#               Int('non_t') == Int('non_s'),
#               Int('non_t') > Int('non_s'),
#               Int('non_t') >= Int('non_s')],
#     'non_t': [Int('non_v'), Int('non_s'), Int('non_op2')],
#     'non_op2': [Int('non_t') + Int('non_t'), Int('non_t') - Int('non_t'), Int('non_t') * Int('non_t'),
#                 Int('non_t') / Int('non_t'), Int('non_t') % Int('non_t')],
#     #'non_op1': [-Int('non_t')],  # 'Rule_op1_abs'],
#     'non_s': [Int('undecided')], #Int('non_decided')
#     # 'non_decided': ['VALUE'],
#     'non_v': []  # dynamically initialize this one
# }
RULE = {
    # conjunction: 1元/2元/3元的
    'non_nc': [And(Bool('non_nd')), And(Bool('non_nd'), Bool('non_nd')),
               And(Bool('non_nd'), Bool('non_nd'), Bool('non_nd'))],
    # disjunction: 1元/2元/3元的
    'non_nd': [Or(Bool('non_p')), Or(Bool('non_p'), Bool('non_p')), Or(Bool('non_p'), Bool('non_p'), Bool('non_p'))],
    # predicate 谓词 p := t < s | t <= s | t == s
    'non_p': [Int('non_t') < Int('non_s'),
              Int('non_t') <= Int('non_s'),
              Int('non_t') == Int('non_s')],
    # t := term | term+term | term+term+term | term+term+term+term
    'non_t': [Int('non_term'),
              Int('non_term') + Int('non_term'),
              Int('non_term') + Int('non_term') + Int('non_term'),
              Int('non_term') + Int('non_term') + Int('non_term') + Int('non_term')],
    # term := v | s*v | s*v*v | s*v*v*v | s*v*v*v*v
    'non_term': [Int('non_v'),
                 Int('non_s') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v') * Int('non_v') * Int('non_v')],
    # 'non_op1': [-Int('non_t')],  # 'Rule_op1_abs'],
    # s := undecided
    'non_s': [Int('undecided')],  # Int('non_decided')
    # 'non_decided': ['VALUE'],
    # v := [], 运行时初始化
    'non_v': []  # dynamically initialize this one
}
const_ID = 0


def InitPT():
    """ 生成全局变量 const_ID, 返回一个 z3 布尔表达式 Bool('non_nc') """
    global const_ID
    const_ID = 0
    return Bool('non_nc')



def getLeftHandle(PT):
    """
    getLeftHandle函数的作用是在AST中找到最左边的"Handle"。
    这里的"Handle"是指可以被替换或者归约的部分，它的标志是节点名称以'non_'开头。
    函数通过递归的方式遍历AST，当找到一个节点名称以'non_'开头时，就返回这个节点，否则就继续遍历其子节点。
    在自底向上的语法分析过程中，"Handle"是指在待归约串中，可以按某个产生式直接归约的部分。也就是说，"Handle"是待归约串中与某个产生式右部匹配的那一部分。
    if PT = And(Bool('non_nd'), Or(Bool('non_p'), Bool('non_p')))
    return Bool('non_nd')
    """
    if 'non_' in str(PT.decl()):    # decl() 方法用于获取该表达式的声明
        return PT                   # 如果一个表达式的声明以 'non_' 开头，那么这个表达式就是一个非终结符，可以根据生成规则进一步展开。
    else:
        for child in PT.children():
            l = getLeftHandle(child)
            if l is not None:
                return l
    return None


def AvailableActionSelection(left_handle):
    # if len(RULE[str(left_handle.decl())]) == 1 and str(RULE[str(left_handle.decl())][0]) == 'VALUE':
    #     return SET_AN_VALUE, None
    # else:
    return config.SELECT_AN_ACTION, RULE[str(left_handle.decl())]


def init_varSelection(vars):
    """
    init_varSelection函数的作用是初始化变量选择规则。
    Args:
        vars: 字符串 v 的列表，每个字符串代表一个变量
    Returns: Z3 整数表达式的列表
    """
    logger.info(f'start: init_varSelection, vars = {vars} ')
    RULE['non_v'] = [Int(v) for v in vars]
    """
    它遍历 vars 列表中的每一个元素 v，并对每个元素调用 Int(v) 函数。
    Int(v) 函数会将字符串 v 转换为一个 Z3 整数表达式。
    所以，[Int(v) for v in vars] 会生成一个新的列表，列表中的每个元素都是一个 Z3 整数表达式，代表一个变量
    """
    SIMPLEST_RULE['non_v'] = [Int(v) for v in vars]


def init_constSelection(consts):
    """
    init_constSelection函数的作用是初始化常量。
    Args:
        consts: IntVal(s) 函数会将整数 s 转换为一个 Z3 整数表达式
    Returns:  Z3 整数表达式的列表

    """
    RULE['non_s'].extend([IntVal(s) for s in consts])


def substitute_the_leftmost_one(node, left_handle, replacer):
    """
    在抽象语法树（AST）中找到最左边的非终结符节点，并用 replacer 替换它
    Args:
        node: 当前正在处理的 AST 节点
        left_handle: 最左边的非终结符节点
        replacer: 用来替换 left_handle 的表达式
    Returns: 成功 return True, replacer, 否则 return False, node
    """
    if 'non_' in str(node.decl()):
        # node 是最左边的非终结符节点
        assert str(node.decl()) == str(left_handle.decl())  # Since this must be the left most one.
        return True, replacer
    else:
        # node 不是最左边的非终结符节点, 找到它的子节点
        childs = node.children()
        if len(childs) >= 1:
            newchilds = []
            replaced = False
            for i, child in enumerate(childs):
                # 获取 node 的所有子节点，并对每个子节点递归调用 substitute_the_leftmost_one 函数。
                replaced, child_after = substitute_the_leftmost_one(child, left_handle, replacer)
                newchilds.append(child_after)
                if replaced:
                    i += 1  # The current i has been used
                    break
            if i < len(childs):
                newchilds.extend(childs[i:])
            if replaced:
                try:
                    return True, getattr(z3, str(node.decl()))(newchilds)
                except:
                    return True, node.decl()(newchilds)
            else:
                # 如果 node 的所有子节点都处理完毕，但都没有找到非终结符节点，那么就返回原来的 nod
                return False, node
        else:  # 没有子节点, just return
            return False, node


def update_PT_rule_selction(PT, left_handle, action_selected):
    assert str(action_selected) != 'VALUE'      # 值不应该被替换
    if str(action_selected) == 'undecided':     # 未定义的变量需要先创建全局的 const_ID 计数变量之后再替换新的表达式
        global const_ID
        action_selected = Int('const_' + str(const_ID))
        const_ID += 1
    return substitute_the_leftmost_one(PT, left_handle, action_selected)[1] # 用新的表达式替换最左非终结符节点


def update_PT_value(PT, left_handle, value_of_int):
    if str(left_handle) == 'non_nc':            # conjunction 使用 disjunction 的逻辑与 进行替换(disjunction: 1元/2元/3元的)
        return And([Bool('non_nd')] * value_of_int)
    elif str(left_handle) == 'non_nd':          # disjunction 使用 predicate 的逻辑或 进行替换(p := t < s | t <= s | t == s)
        return substitute_the_leftmost_one(PT, left_handle, Or([Bool('non_p')] * value_of_int))[1]
    else:                                       # 其他的都是用整数表达式进行替换, IntVal(s) 函数会将整数 s 转换为一个 Z3 整数表达式
        return substitute_the_leftmost_one(PT, left_handle, IntVal(value_of_int))[1]


# def ShouldStrict(lefthandle, Whom):
#     if Whom == "V":
#         if str(lefthandle) in ['non_nc', 'non_nd', 'non_t', 'non_op2']:
#             return True
#         else:
#             return False
#     else:
#         assert Whom == "S"
#         if str(lefthandle) in ['non_nc', 'non_nd', 'non_t', 'non_op2', 'non_s']:
#             return True
#         else:
#             return False

def ShouldStrict(lefthandle, Whom):
    if Whom == "V":
        if str(lefthandle) in ['non_nc', 'non_nd', 'non_t', 'non_term']:
            # conjunction, disjunction, +, *
            return True
        else:
            return False
    else:
        assert Whom == "S"

        if str(lefthandle) in ['non_nc', 'non_nd', 'non_t', 'non_term', 'non_s']:
            # conjunction, disjunction, +, *, undecided
            return True
        else:
            return False


# def StrictnessDirtribution(lefthandle, Whom):
#     if str(lefthandle) == 'non_op2':
#         return tensor([[0.4, 0.4, 0.1, 0.05, 0.05]], dtype=torch.float32)
#
#     if Whom == "V":
#         assert str(lefthandle) == 'non_t'
#         return tensor([[0.1, 0.85, 0, 0.05]], dtype=torch.float32)
#     else:
#         assert Whom == "S"
#         assert str(lefthandle) in ['non_t', 'non_s']
#         if str(lefthandle) == 'non_t':
#             return tensor([[0.85, 0.1, 0, 0.05]], dtype=torch.float32)
#         else:
#             return tensor([[1, 0]], dtype=torch.float32)

def StrictnessDirtribution(lefthandle, Whom):
    """
    根据给定的左句柄和动作类型，调整概率分布。
    Args:
        lefthandle: 最左边的非终结符节点
        Whom: S 或 V

    Returns: S 下,会调整乘法的概率分布, 更多是一元的乘法,V 模式下, 会调整乘法和加法的概率分布, 更多是一元/二元

    """
    distri_dict = {
        'non_nc': [0.95, 0.05, 0.0],
        'non_nd': [0.95, 0.05, 0.0],
        'non_t': [0.95,
                  0.049,
                  0.001,
                  0.0],
        'non_term': [0.5055,
                     0.4944,
                     0.0001,
                     0.0,
                     0.0],
        'non_s': [0]
    }
    if Whom == "S":
        # conjunction, disjunction, +, *, undecided
        distri_dict['non_term'] = [0.99, 0.01, 0.0, 0.0, 0.0]

    if (len(RULE['non_s']) - 1) > 0:
        # 如果 RULE['non_s'] 有多个元素，则为 non_s 类型动作分配均匀概率；如果只有一个元素，则概率为 1。
        distri_dict['non_s'].extend([1 / (len(RULE['non_s']) - 1)] * (len(RULE['non_s']) - 1))
    else:
        distri_dict['non_s'] = [1]
    for kk in distri_dict:
        try:
            assert len(distri_dict[kk]) == len(RULE[kk])
        except Exception as e:
            print(e)
            raise e

    res = tensor([distri_dict[str(lefthandle)]], dtype=torch.float32)
    if torch.cuda.is_available():
        res = res.cuda()
    return res


def LossnessDirtribution(lefthandle, Whom): #only S will ask it.
    """
    更加宽松, conjunction 集中在 3 元合取, disjunction 集中在 3 元析取, + 集中在 4 元加法, * 集中在 2 元乘法
    Args:
        lefthandle:
        Whom:

    Returns:

    """
    distri_dict = {
        'non_nc': [0.0, 0.25, 0.75],
        'non_nd': [0.0, 0.25, 0.75],
        'non_t': [0.05,
                  0.15,
                  0.2,
                  0.6],
        'non_term': [0.0,
                     0.8,
                     0.19,
                     0.009,
                     0.001],
        'non_s': [1]
    }
    if (len(RULE['non_s']) - 1) > 0:
        distri_dict['non_s'].extend([0] * (len(RULE['non_s']) - 1))

    for kk in distri_dict:
        try:
            assert len(distri_dict[kk]) == len(RULE[kk])
        except Exception as e:
            print(e)
            raise e

    res = tensor([distri_dict[str(lefthandle)]], dtype=torch.float32)
    if torch.cuda.is_available():
        res = res.cuda()
    return res


# SIMPLEST_RULE = {
#     'non_nc': [And(Bool('non_nd'))],
#     'non_nd': [Or(Bool('non_p'))],
#     'non_p': [Int('non_t') < Int('non_s')],
#     'non_t': [Int('non_s')],
#     'non_op2': [Int('non_t') + Int('non_t')],
#     #'non_op1': [-Int('non_t')],  # 'Rule_op1_abs'],
#     'non_s': [Int('undecided')],
#     #'non_decided': ['VALUE'],
#     'non_v': []  # dynamically initialize this one
# }
SIMPLEST_RULE = {
    'non_nc': [And(Bool('non_nd'))],
    'non_nd': [Or(Bool('non_p'))],
    'non_p': [Int('non_t') < Int('non_s')],
    'non_t': [Int('non_term')],
    'non_term': [Int('non_v')],
    'non_s': [Int('undecided')],
    'non_v': []  # dynamically initialize this one
}


def simplestAction(left_handle):
    return SIMPLEST_RULE[str(left_handle)][0]


# liitel test
if __name__ == "__main__":
    exp = And(Int('x') + Int('y') < 3, Bool('non_p'), Int('non_t') < Int('non_t'))
    print(exp)
    exp = substitute_the_leftmost_one(exp, getLeftHandle(exp), Int('non_t') >= Int('non_t'))[1]
    exp = substitute_the_leftmost_one(exp, getLeftHandle(exp), Int('z') % Int('q'))
    print(exp[1])
