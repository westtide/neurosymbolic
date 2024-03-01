from pycparser import c_ast, parse_file, c_parser, c_generator
import sys
import os
sys.path.extend(['.', '..'])




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
# pre_conditions = get_pre_conditions_from_source_code("59.c")
# print(pre_conditions)


import re

with open('59.c', 'r') as file:
    content = file.read()

pre_exp = ""
loop_exp = ""
post_exp = ""

# 提取 pre-conditions 部分的代码
pre_conditions = re.search('// pre-conditions(.*?)// loop body', content, re.DOTALL)
if pre_conditions:
    pre_exp = pre_conditions.group(1)
    # print(f'pre_conditions = {pre_conditions.group(1)}')



# 提取 loop body 部分的代码
loop_body = re.search('// loop body(.*?)// post-condition', content, re.DOTALL)
if loop_body:
    loop_exp = loop_body.group(1)
    # print(f'loop_body: {loop_body.group(1)}')

# 提取 post-condition 部分的代码
post_condition = re.search('// post-condition(.*?)(?=})', content, re.DOTALL)
if post_condition:
    post_exp = post_condition.group(1)
    # print(f'post_condition: {post_condition.group(1)}')


base1 = """
int main(){ """

base2 = """}"""

ccode = [""] * 4
ccode[0] = base1 + pre_exp + base2
ccode[1] = base1 + loop_exp + base2  
ccode[2] = base1 + post_exp + base2




for i in range(0,3):
    parser = c_parser.CParser()

    astnode = parser.parse(text = ccode[i], filename='<none>')

    file_ast = astnode
    # file_ast.show()
    # 定义一个用于遍历AST节点的函数
    exp_assume = []
    exp_assert = []
    def find_func_calls(ast):
        if isinstance(ast, c_ast.FuncCall):
            # 打印函数名称
            if ast.name.name == "assume":
                exp_assume.append(ast.args)
            if ast.name.name == "assert":
                exp_assert.append(ast.args)
            print(f"函数调用: {ast.name.name}")
            # 如果 name == assume, 提取表达式
            # 如果 name == assert, 提取表达式
        for _, child in ast.children():
            find_func_calls(child)
    # 使用定义的函数遍历AST
    find_func_calls(file_ast)
    for item in exp_assume:
        generator = c_generator.CGenerator()
        expr = c_ast.ExprList(item)
        print(f'exp_assume: {generator.visit(expr)}')

    for item in exp_assert:
        generator = c_generator.CGenerator()
        expr = c_ast.ExprList(item)
        print(f'exp_assert: {generator.visit(expr)}')


    # 定义一个用于遍历AST节点并查找for循环的函数
    def find_for_loops(node):
        if isinstance(node, c_ast.For):
            print("找到一个for循环")
            # 可以进一步分析循环的初始化、条件和迭代部分
        for _, child in node.children():
            find_for_loops(child)

    # 使用定义的函数遍历AST
    find_for_loops(file_ast)

    # 定义一个用于遍历AST节点并查找while循环的函数
    def find_while_loops(node):
        if isinstance(node, c_ast.While):
            print("找到一个while循环")
            # 可以进一步分析循环的条件部分
        for _, child in node.children():
            find_while_loops(child)

    # 使用定义的函数遍历AST
    find_while_loops(file_ast)



