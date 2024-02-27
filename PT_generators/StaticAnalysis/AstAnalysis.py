from pycparser import c_ast, parse_file, c_parser


class AstVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.loop_vars = set()
        self.loop_consts = []
        self.whiles = []
        self.in_while = False

    def visit_ID(self, node):
        if self.in_while:
            self.loop_vars.add(node.name)

    def visit_Constant(self, node):
        if self.in_while:
            self.loop_consts.append(node.value)

    def visit_While(self, node):
        self.whiles.append(node)
        self.in_while = True
        self.visit(node.stmt)
        self.in_while = False


def get_loop_var(path2CFile):
    """
    根据原 C 程序代码获得 AST, 分析 AST 中的 while, for 循环中的变量
    Args:
        path2CFile: C 程序代码
    Returns: 循环中的 loop_body, vars_in_loop, cons_in_loop
    """
    try:
        astnode = parse_file(path2CFile, use_cpp=True, cpp_path='/opt/homebrew/bin/cpp-13')
    except c_parser.ParseError as e:
        return "Parse error:" + str(e)

    v = AstVisitor()
    v.visit(astnode)
    loop = v.whiles

    vars_in_loop = v.loop_vars
    consts_in_loop = v.loop_consts

    return loop, list(vars_in_loop), list(consts_in_loop)


# loop_body, vars_in_loop, cons_in_loop = get_loop_var("59.c")
# print(loop_body, vars_in_loop, cons_in_loop)

def check_for_pre_condition_comment(path2CFile):
    with open(path2CFile, 'r') as file:
        for line in file:
            if "// pre-conditions" in line:
                return True
    return False

from pycparser import c_parser, c_ast, parse_file

class PreConditionVisitor(c_ast.NodeVisitor):
    def __init__(self, has_pre_condition_comment):
        self.pre_conditions = []
        self.in_pre_condition = False
        self.has_pre_condition_comment = has_pre_condition_comment

    def visit_FuncDef(self, node):
        if not self.has_pre_condition_comment:
            # 如果没有“// pre-conditions”注释，假设整个函数体都是前置条件
            self.in_pre_condition = True
        self.generic_visit(node)

    def visit_While(self, node):
        if not self.has_pre_condition_comment or self.in_pre_condition:
            # 当进入while循环时，如果之前已经在收集前置条件，则停止收集
            self.in_pre_condition = False
        # 如果有“// pre-conditions”注释，不做任何操作，因为收集逻辑由注释控制

    # visit_Assignment和visit_FuncCall方法与之前相同

def get_pre_conditions_from_source_code(path2CFile):
    has_pre_condition_comment = check_for_pre_condition_comment(path2CFile)
    astnode = parse_file(path2CFile, use_cpp=True, cpp_path='/opt/homebrew/bin/cpp-13')
    v = PreConditionVisitor(has_pre_condition_comment)
    v.visit(astnode)
    return v.pre_conditions

# 示例使用
# pre_conditions = get_pre_conditions_from_source_code("3.c")
# print(pre_conditions)
