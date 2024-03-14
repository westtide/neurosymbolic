from z3 import *

x = Real('x')
x.from_string('10')
print(x)  # 输出：10


# Arithmetic

x = Real('x')
y = Int('y')
a, b, c = Reals('a b c')
s = Int('s')
r = Int('r')
# 在需要时自动添加强制将整数表达式转换为实数
print(x + y + 1 + (a + s))      # x + ToReal(y) + 1 + a + ToReal(s)
# ToReal 将整数表达式转换为实数表达式
print(ToReal(y) + c)

x = Int('x')
y = Int('y')
f = Function('f', IntSort(), IntSort())         # IntSort() 用于创建一个整数类型
s = Solver()
s.add(f(f(x)) == x, f(x) == y, x != y)
print (s.check())
m = s.model()
print ("f(f(x)) =", m.evaluate(f(f(x))))        # f(f(x)) = 0
print ("f(x)    =", m.evaluate(f(x)))           # f(x) = 1
