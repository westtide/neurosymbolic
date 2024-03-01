from pycparser import parse_file
import sys
sys.path.extend(['.', '..'])
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open ('59.c', 'r') as file:
    ccode = file.read()

astnode = parse_file('59.c', use_cpp=True, cpp_path = '/opt/homebrew/bin/cpp-13')
file_ast = astnode
func = file_ast.ext[0]
func_decl = func.decl
func_body = func.body

print("func_decl: 函数声明")
func_decl.show()

print('func_body: 函数体')
func_body.show()

for decl in func_decl.body.block_items:
    decl.show()

    