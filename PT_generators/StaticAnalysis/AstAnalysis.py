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

loop_body, vars_in_loop, cons_in_loop = get_loop_var("59.c")

class PreConditionVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.pre_conditions = []
        self.record = False

    def visit_Compound(self, node):
        for i, child in enumerate(node.block_items):
            if isinstance(child, c_ast.Comment) and child.coord.line == i+1:
                if "// pre-conditions" in child.text:
                    self.record = True
                elif "// loop body" in child.text:
                    self.record = False
            elif self.record:
                self.pre_conditions.append(child)

def get_pre_conditions_from_source_code(path2CFile):
    try:
        astnode = parse_file(path2CFile, use_cpp=True, cpp_path='/opt/homebrew/bin/cpp-13')
    except c_parser.ParseError as e:
        return "Parse error:" + str(e)
    v = PreConditionVisitor()
    v.visit(astnode)
    return v.pre_conditions

pre_conditions = get_pre_conditions_from_source_code("59.c")