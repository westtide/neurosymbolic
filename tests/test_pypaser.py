from pycparser import c_parser, c_ast

# 示例C代码
c_code = """
int main() {
  /* variable declarations */
  int x;
  int y;
  /* pre-conditions */
  (x = 1);
  (y = 0);
  /* loop body */
  while ((y < 100000)) {
    {
    (x  = (x + y));
    (y  = (y + 1));
    }

  }
  /* post-condition */
assert( (x >= y) );
}

"""

# 解析C代码
parser = c_parser.CParser()
ast = parser.parse(c_code)

# 定义访问者类来查找特定的函数调用
class FuncCallVisitor(c_ast.NodeVisitor):
    def visit_FuncCall(self, node):
        if node.name.name in ['assume', 'assert']:
            print(f"Found {node.name.name} statement: {node}")
        self.generic_visit(node)

# 创建访问者实例并遍历AST
visitor = FuncCallVisitor()
visitor.visit(ast)
