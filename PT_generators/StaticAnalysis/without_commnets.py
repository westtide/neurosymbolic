code = """int main() {
    int d1 = 1;
    int d2 = 1;
    int d3 = 1;
    int x1 = 1;
    int x2,x3;

    assume(x1 > 0);
    assume(x2 > 0);
    assume(x3 > 0);
    while( x1 > 0) {
        if(x2 > 0) {
            if(x3 > 0) {
                x1 = x1 - d1;
                x2 = x2 - d2;
                x3 = x3 - d3;
            }
        }
    }

    //assert (x2 >= 0);
    assert (x3 >= 0);
}
"""


import re
from pycparser import c_parser, c_generator, c_ast

# 提取 "int main() {" 到 "while(" 之间的代码
pre = re.search('int main\(\) \{(.*?)while\(', code, re.DOTALL)
if code:
    print(pre.group(1))


base1 = """
int main(){ """

base2 = """}"""

ccode = base1 + pre.group(1) + base2
parser = c_parser.CParser()

if 1:
    astnode = parser.parse(text = ccode, filename='<none>')

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
