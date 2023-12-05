from z3 import *


# Z3 中的变量
x = Int('x')
y = Int('y')
# [y = 0, x = 7]
solve(x > 2, y < 10, x + 2*y == 7)


# 3 + 3*x + y
print (simplify(x + y + 2*x + 3))


# Not(y <= -2) is equivalent to y > -2
print(simplify(x < y + x + 2))
print(simplify(y > -2))


# And 是逻辑与，前缀表达式，接受两个参数
print(simplify(And(x + 1 >= 3, x**2 + x**2 + y**2 + 2 >= 5)))


# x**2 + y**2 >= 1
print (x**2 + y**2 >= 1)
# 关闭HTML模式
set_option(html_mode=False)
# x**2 + y**2 >= 1
print (x**2 + y**2 >= 1)

# [y = 0, x = 2]
solve(simplify(And(And(x + 1 >= 3, y >= 0), x**2 + x**2 + y**2 + 2 >= 5)))


