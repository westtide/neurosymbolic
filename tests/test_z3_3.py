from z3 import *

x = Int('x')
y = Int('y')
solve(x > 2, y < 10, x + 2*y == 7)


# 表达式化简
x = Int('x')
y = Int('y')
exp1 = x + y + 2 * x + 3 == 0
exp2 = x < y + x + 2
# And 表示逻辑与，必须连接两个布尔表达式
exp3 = simplify(And(exp1, exp2))
print(exp3)

# 默认情况下，Z3Py（用于Web）使用数学符号显示公式和表达式
x = Int('x')
y = Int('y')
print (x**2 + y**2 >= 1)
set_option(html_mode=False)
print (x**2 + y**2 >= 1)

# 遍历表达式的函数
x = Int('x')
y = Int('y')
exp4 = x + y >= 3
print(f'num args: {exp4.num_args()}')   # 表达式的参数个数
print(f'children: {exp4.children()}')   # 表达式的子表达式
print(f'1st child: {exp4.arg(0)}')      # 表达式的第一个参数
print(f'2nd child: {exp4.arg(1)}')      # 表达式的第二个参数
print(f'operator: {exp4.decl()}')       # 表达式的操作符
print(f'op name: {exp4.decl().name()}') # 表达式的操作符名称


# Real 实数类型
x = Real('x')
y = Real('y')
solve(x **2 + y **2 > 3, x**3 + y < 5)
# [y = 2, x = 1/8]

# 设置精度
x = Real('x')
y = Real('y')
# 设置显示结果时使用的小数位数
set_option(precision=30)
print ("Solving, and displaying result with 30 decimal places")
solve(x**2 + y**2 == 3, x**3 == 2)
# [y = -1.188528059421316533710369365015?, x = 1.259921049894873164767210607278?] ?表示不确定的数字

# 创建有理数类型
print(f'1/3')
# RealVal 创建一个Z3实数
print(f'1/3 = {RealVal(1)/3}')  # RealVal(1) 创建一个Z3实数，表示数字 1
# Q(num, den) 创建一个Z3有理数
print(f'5/3 = {Q(5,3)}')        # Q(5,3) 创建一个Z3有理数，表示 5/3


x = Real('x')
solve(x > 4, x < 0)
# no solution


