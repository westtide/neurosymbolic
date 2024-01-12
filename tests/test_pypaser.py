# 这里是 AssertAssumeVisitor 类和 get_asserts_assumes_from_source_code 函数的定义
from pycparser import c_ast, parse_file


from pycparser import c_ast, parse_file

class AssertAssumeVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.asserts = []
        self.current_path = []

    def visit(self, node):
        # 添加当前节点到路径
        self.current_path.append(node)

        # 访问当前节点的子节点
        super(AssertAssumeVisitor, self).visit(node)

        # 从路径中移除当前节点
        self.current_path.pop()

    def visit_FuncCall(self, node):
        if isinstance(node.name, c_ast.ID):
            if node.name.name == "assert":
                # 检查父节点是否为 if 类型
                if len(self.current_path) >= 2 and isinstance(self.current_path[-2], c_ast.If):
                    self.asserts.append(self.current_path[-2])
                else:
                    self.asserts.append(node)

    # 同样处理 assume 语句
    def visit_Assume(self, node):
        if isinstance(node.name, c_ast.ID):
            if node.name.name == "assume":
                # 检查父节点是否为 if 类型
                if len(self.current_path) >= 2 and isinstance(self.current_path[-2], c_ast.If):
                    self.asserts.append(self.current_path[-2])
                else:
                    self.asserts.append(node)

    # 其他方法根据需要添加



def get_asserts_assumes_from_source_code(path2CFile):
    try:
        astnode = parse_file(path2CFile, use_cpp=True, cpp_path='/opt/homebrew/bin/cpp-13')
    except OSError as e:
        return "Parse error:" + str(e)
    v = AssertAssumeVisitor()
    v.visit(astnode)
    return v.asserts, v.visit_Assume


# 测试样例文件路径
path_to_test_case = "../Benchmarks/Linear/c/59.c"

# 提取 assert 和 assume 语句
asserts, assumes = get_asserts_assumes_from_source_code(path_to_test_case)

# 打印结果
print("Asserts:")
for a in asserts:
    print(a)
print("\nAssumes:")
for a in assumes:
    print(a)

"""
int main() {
  // variable declarations
  int c;
  int n;
  // pre-conditions
  (c = 0);
  assume((n > 0));
  // loop body
  while (unknown()) {
    {
      if ( unknown() ) {
        if ( (c != n) )
        {
        (c  = (c + 1));
        }
      } else {
        if ( (c == n) )
        {
        (c  = 1);
        }
      }

    }

  }
  // post-condition
if ( (c != n) )
assert( (c <= n) );
}

"""