# A pipeline framework to realize the RL Pruning Tool for loop invariant inference
import argparse
import time

from PT_generators.RL_Prunning.PT_generator import PT_generator
from SMT_Solver import Template_solver
from SMT_Solver.Config import config
from SMT_Solver.SMT_verifier import SMT_verifier
from loginit import logger
from Utilities.ArgParser import parseArgs



def main(path2CFile, path2CFG, path2SMT):
    logger.info(f'start: main')
    logger.info(f'main: path2CFile = {path2CFile}, path2CFG = {path2CFG}, path2SMT = {path2SMT} ')
    start_time = time.time()
    # Step 1. Input the three formation of the code.
    #path2CFile, path2CFG, path2SMT = parseArgs()
    # Step 2. Load the Partial Template Generator.
    logger.info(f'start: main -> PT_generator()')
    # 初始化模板, LSTM, Embedding, RewardPredictor 等网络
    pT_generator = PT_generator(path2CFile, path2CFG, path2SMT)
    sMT_verifier = SMT_verifier()                               # 负责读取 smt-lib 文件和进行 SMT 验证
    # Step 3. ENTER the ICE Solving Loop
    solved = False
    CE = {'p': [],  # positive counterexample
          'n': [],  # negative counterexample
          'i': []}  # inductive counterexample
    print("Begin_process:   ", path2CFile)
    Iteration = 0
    counterNumber = 0
    while not solved:
        current_time = time.time()
        if current_time - start_time >= config.Limited_time:
            print("Loop invariant Inference is OOT")
            return None,None
        Iteration += 1
        # Step 3.1 Generate A partial template
        PT = pT_generator.generate_next(CE)
        logger.info(f'main: Iteration = {Iteration}, Template: {PT.__str__()}')
        print("Template: ", PT.__str__())
        if PT is None:
            # 如果生成的模板是None，说明已经没有可用的模板了，只能放弃了
            logger.info(f'! Iteration = {Iteration}, give up now')
            print("The only way is to give up now")
            return None,None
        # Step 3.2 Solving the partial template
        try:
            # 根据
            Can_I = Template_solver.solve(PT, CE)
            logger.info(f'Iteration = {Iteration}, Can_I = {Can_I}')
            #raise TimeoutError # try this thing out
        except TimeoutError as OOT:  # Out Of Time, we punish

            pT_generator.punish('STRPTICT', 'VERY', 'S')  # Case 1: Template Solving is out of time.
            print("Solving timeout")
            logger.warning(f'Iteration = {Iteration}, Case 1: Template OOT')
            continue
        if Can_I is None:  # Specified too much, we loose.

            print("Solving unsat")
            pT_generator.punish('LOOSE', 'MEDIUM', 'S')
            logger.warning(f'Iteration = {Iteration}, Case 2: Template Unsat')   # Case 2: Template Solving is unsat.
            continue
        # Step 3.3 Check if we bingo
        try:
            print("Candidate: ", Can_I.__str__())
            logger.info(f'Iteration = {Iteration}, Candidate = {Can_I.__str__()} ')
            Counter_example = sMT_verifier.verify(Can_I, path2SMT)
        except TimeoutError as OOT:  # Out Of Time, we punish
            print("Checking timeout")
            pT_generator.punish('STRICT', 'LITTLE', 'V')    # Case 3: Inv Candidate Checking is out of time.
            logger.warning(f'Iteration = {Iteration}, Case 3: Inv Candidate Check OOT')
            continue
        if Counter_example is None:  # Bingo, we prise
            solved = True
            print("The answer is :  ", str(Can_I))
            pT_generator.prise('VERY')
            current_time = time.time()
            print("Time cost is :  ", str(current_time - start_time))
            logger.info(f'Iteration = {Iteration}, Time cost is : {str(current_time - start_time)}, Case 5: Success')
            return current_time - start_time, str(Can_I)
        else:  # progressed anyway, we prise
            if Counter_example.assignment not in CE[Counter_example.kind]:  # Case 4: Candidate Inv SAT
                CE[Counter_example.kind].append(Counter_example.assignment)
            counterNumber += 1
            logger.info(f'Iteration = {Iteration}, Size of CE: {counterNumber}, Case 4: Candidate Inv SAT')
            print("Size of CE: ", counterNumber)
            pT_generator.prise('LITTLE')
            continue
if __name__ == "__main__":
    path2CFile=r"Benchmarks/Linear/c/4.c"
    path2CFG=r"Benchmarks/Linear/c_graph/4.c.json"
    path2SMT=r"Benchmarks/Linear/c_smt2/4.c.smt"
    main(path2CFile, path2CFG, path2SMT)