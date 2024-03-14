from z3 import *
set_param('parallel.enable', True)  # 启用并行求解
from SMT_Solver.Config import config
from Utilities.TimeController import time_limit_calling
from loginit import logger

class Counterexample:
    # 反例类
    kind = "?"
    assignment = {}


class SMT_verifier:
    tpl = []

    def initTpl(self, path2SMT):
        """
        读取一个SMT文件，并根据特定的分隔符将文件内容分割成不同的部分，用于构造求解时需要的模板。
        Args:
            path2SMT: SMT 文件的路径

        Returns:无返回, 修改 tpl

        """
        vc_sections = [""]
        with open(path2SMT, 'r') as vc:
            for vc_line in vc.readlines():
                if "SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop" in vc_line:
                    vc_sections.append("")
                else:
                    # logger.info(f'vc_sections[-1] += vc_line = {vc_line}')
                    vc_sections[-1] += vc_line
        assert len(vc_sections) == 5

        self.tpl = [vc_sections[0]]
        # logger.info(f'tpl = {self.tpl}, vc_sections[0] = {vc_sections[0]}')

        for i in range(2, 5):
            self.tpl.append(vc_sections[1] + vc_sections[i])
            # logger.info(f'i = {i}, vc_sections[i] = {vc_sections[i]}, tpl = {self.tpl}')


    def verify(self, Can_I, path2SMT):
        """
        验证函数: 将Can_I转换为SMT-LIB字符串，并根据之前准备的模板构造完整的SMT问题，然后交给Z3 Solver求解，并等待求解器的响应。
        Args:
            Can_I: Z3 表达式
            path2SMT: SMT 文件的路径

        Returns: Counterexample实例，表示找到了一个反例；或者返回None，表示未找到反例或验证通过

        """
        sol = z3.Solver()
        sol.set(auto_config=False)
        sol.set("timeout", config.SMT_CHECK_TIME)

        if len(self.tpl) <= 0:
            self.initTpl(path2SMT)

        # Z3表达式（Can_I）转换为SMT-LIB字符串
        Can_I_smt = \
            Z3_benchmark_to_smtlib_string(Can_I.ctx_ref(), "benchmark", "NIA", "unknown", "", 0, (Ast * 0)() , Can_I.as_ast())
        # logger.info(f'Can_I.ctx_ref() = {Can_I.ctx_ref()}, Ast * 0 = {(Ast * 0)()}, Can_I.as_ast() = {Can_I.as_ast()}')
        # logger.info(f'Can_I_smt = {Can_I_smt}')
        # Z3表达式的上下文引用（Can_I.ctx_ref()），一个字符串（"benchmark"），
        # 逻辑名称（"NIA"），状态（"unknown"），
        # 一个空字符串，一个零，一个空的Ast数组，以及Z3表达式的抽象语法树（Can_I.as_ast()）
        Can_I_smt = Can_I_smt.split('(assert\n')[-1].split('(check-sat)')[0][:-2]
        # logger.info(f'after spilt: Can_I_smt = {Can_I_smt}')
        # 使用(assert\n)作为分隔符，取最后一部分
        # 使用(check-sat)作为分隔符，取第一部分。
        # 最后，去掉字符串末尾的两个字符。这样，我们就得到了一个只包含Z3表达式的SMT-LIB字符串。
        for i in range(3):
            s = self.tpl[0] + Can_I_smt + self.tpl[i + 1]  # 0: pre, 1: post, 2: inv
            sol.reset() # reset the solver
            decl = z3.parse_smt2_string(s) # parse the string
            sol.add(decl) # add the string to the solver
            #try:
            r = time_limit_calling(sol.check, (), config.SMT_CHECK_TIME) # check the string
            #     r = sol.check()
            # except Exception as e:
            #     r = z3.unknown
            kind = "?"
            ce = {}
            if z3.sat == r:             # we got a counterexample
                m = sol.model()         # get the model
                if i == 0 or i == 2:    # pre or inv
                    for x in m:
                        v = str(x)
                        if v in ['inv-f', 'post-f', 'pre-f', 'trans-f', 'div0', 'mod0']:
                            """
                            这些字符串['inv-f', 'post-f', 'pre-f', 'trans-f', 'div0', 'mod0']在这段代码中被用作变量名的过滤条件。
                            在Z3求解器返回的模型中，这些字符串可能是变量名。然而，在这段代码中，它们被视为特殊的变量，不应被包含在反例中。  
                            具体来说：  
                            'inv-f'，'post-f'，'pre-f'，'trans-f'可能是与程序的不变性、后置条件、前置条件和转换函数相关的变量。
                            'div0'和'mod0'可能是与除法和模运算有关的特殊情况，例如除数为0的情况。
                            这些变量可能是在Z3表达式中定义的，用于描述和解决特定的问题。然而，这些变量的具体含义和用途取决于它们在Z3表达式中的使用方式。
                            """
                            continue
                        if "_" in v:
                            continue
                        ce[v] = m[x]
                    kind = "p" if i == 0 else "n" # pre or inv
                else: # post
                    m1, m2 = {}, {}
                    for x in m:
                        v = str(x)
                        if v in ['inv-f', 'post-f', 'pre-f', 'trans-f', 'div0', 'mod0']:
                            continue
                        if "_" in v:
                            continue
                        const = m[x]
                        if v.endswith("!"):
                            m2[v[:-1]] = const
                        else:
                            m1[v] = const
                    ce = (m1, m2)
                    kind = "i"
                counterexample = Counterexample()
                counterexample.kind = kind
                counterexample.assignment = ce
                return counterexample

            elif z3.unknown == r: # timeout
                print(i)
                raise TimeoutError("SMT Checking is OOT:    " + str(Can_I))

            else:
                assert z3.unsat == r #we pass this one, continue
                continue

        return None  # we find the answer




