import random
import signal

from z3 import *

from loginit import logger

set_param('parallel.enable', True)
from SMT_Solver.Config import config
from Utilities.SMT_parser import getConstsFromZ3Exp
from Utilities.TimeController import time_limit_calling


def Substitute(PT, assignment):
    """
    替换PT中的变量
    """
    assignedVars = assignment.keys()
    logger.info(f'PT = {PT}, assignedVars = {assignedVars}, assignment = {assignment}')
    canI = PT
    for avar in assignedVars:
        # try:
        theValue = IntVal(str(assignment[avar]))
        avar = Int(str(avar))
        canI = z3.z3.substitute(canI, (avar, theValue))
        # logger.info('canI = ' + str(canI))
        # except Exception as e:
        #     print(e)

    return canI  # remember


"""
2024-01-11 21:01:03,749 - start: log - INFO - PT = And(Or(y + y + y <= const_0,
       y + y + y + y <= const_1,
       y + y + y <= const_2),
    Or(y + y + y <= const_3,
       y + y + y <= const_4,
       y + y + y <= const_5)), assignedVars = dict_keys(['x', 'y']), assignment = {'x': 0, 'y': 0}
2024-01-11 21:01:03,751 - start: log - INFO - canI = And(Or(y + y + y <= const_0,
       y + y + y + y <= const_1,
       y + y + y <= const_2),
    Or(y + y + y <= const_3,
       y + y + y <= const_4,
       y + y + y <= const_5))
2024-01-11 21:01:03,753 - start: log - INFO - canI = And(Or(0 + 0 + 0 <= const_0,
       0 + 0 + 0 + 0 <= const_1,
       0 + 0 + 0 <= const_2),
    Or(0 + 0 + 0 <= const_3,
       0 + 0 + 0 <= const_4,
       0 + 0 + 0 <= const_5))
"""


def solve(PT, CE):
    """
    量化自由非线性整数算术的求解函数: 替换 PT 中的变量, 再尝试求解
    Args:
        PT: 模板
        CE: 反例
    Returns: 如果找到解，函数将返回一个包含解的字典。如果没有找到解，函数将返回None。如果求解器超时，函数将抛出一个超时错误。
    """
    sol = z3.SolverFor("QF_NIA")
    sol.set(auto_config=False)
    sol.set("timeout", config.PT_SOLVING_TIME)

    # Substitute all program vars with CE table.
    # PT_SOLVING_MAX_CE = 1000, 最多1000个
    Query = And(True, True)
    P_sampled = CE['p'] if len(CE['p']) <= config.PT_SOLVING_MAX_CE else random.sample(CE['p'],
                                                                                       config.PT_SOLVING_MAX_CE)
    N_sampled = CE['n'] if len(CE['n']) <= config.PT_SOLVING_MAX_CE else random.sample(CE['n'],
                                                                                       config.PT_SOLVING_MAX_CE)
    I_sampled = CE['i'] if len(CE['i']) <= config.PT_SOLVING_MAX_CE else random.sample(CE['i'],
                                                                                       config.PT_SOLVING_MAX_CE)

    logger.info(f'P_sampled = {P_sampled}, N_sampled = {N_sampled}, I_sampled = {I_sampled}')
    i = 0
    for counterexample in P_sampled:
        # 用 CE 代替 PT 里面的值
        i += 1
        # logger.info(f'i = {i}, counterexample = {counterexample}')
        pterm = Substitute(PT, counterexample)
        Query = And(Query, pterm)
        # logger.info(f'i = {i}, p_pterm = {pterm}, Query = {Query}')
    for counterexample in N_sampled:
        pterm = Substitute(PT, counterexample)
        Query = And(Query, Not(pterm))
        # logger.info(f'i = {i}, n_pterm = {pterm}, Query = {Query}')
    for counterexample in I_sampled:
        pre = Substitute(PT, counterexample[0])
        post = Substitute(PT, counterexample[1])
        Query = And(Query, Implies(pre, post))
        # logger.info(f'i = {i}, pre = {pre}, post = {post}, Query = {Query}')

    # Try to find a solution.
    try:
        Query = simplify(Query)
    except Exception as e:
        print(e)
    sol.reset()
    # set to QFNIA
    sol.add(Query)

    r = time_limit_calling(sol.check, (Query), config.PT_SOLVING_TIME)

    if r == z3.sat:  # coool
        m = sol.model()
        assignment = {}
        for s in m:
            if 'const_' not in str(s):
                continue
            assignment[str(s)] = str(m[s])
            # 如果元素的名称中包含'const_'，则将该元素及其值添加到赋值字典中

        consts = getConstsFromZ3Exp(PT)
        for conster in consts:
            if str(conster) not in assignment:
                # that means the consts can be any value
                # 处理常量, 如果常量不在字典里面, 就随机赋值
                assignment[conster] = IntVal(random.randint(-10, 10))

        return Substitute(PT, assignment)

    elif r == z3.unsat:  # Not Cool
        return None
    else:
        raise TimeoutError("template solving is OOT:    " + str(PT))
