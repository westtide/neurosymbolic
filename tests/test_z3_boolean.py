from z3 import *

p = Bool('p')
q = Bool('q')
r = Bool('r')
solve(Implies(p, q), r == Not(q), Or(Not(p), r))
# [q = True, p = False, r = False]


# 布尔常量 True 和 False
p = Bool('p')
q = Bool('q')
print (And(p, q, True))                 # And(p, q, True)
print (simplify(And(p, q, True)))       # And(p, q)
print (simplify(And(p, False)))         # False


# 混合多项式和布尔约束的组合
p = Bool('p')
x = Real('x')
exp = (Or(x < 5, x > 10), Or(p, x**2 == 2), Not(p))
solve(exp)
# [x = -1.4142135623?, p = False]

# Solver
# add: 添加断言，check(): 检查可满足性(SAT, UNAST, UNKNOWN)
x = Int('x')
y = Int('y')
s = Solver()
print(s)            # []
print(s.check())    # SAT

s.add(x > 10, y == x + 2)
print(s)            # [x > 10, y == x + 2]
print(s.check())    # SAT

# 维护断言堆栈
# push: 保存当前状态，pop: 恢复到上一个状态
s.push()
s.add(y < 11)
print(f'push: {s}') # push: [x > 10, y == x + 2, y < 11]
print(s.check())    # UNSAT

s.pop()
print(f'pop: {s}')  # [x > 10, y == x + 2]
print(s.check())    # SAT


# unknown 情况
# Z3可以求解非线性多项式约束，但 2**x 不是多项式
x = Real('x')
s = Solver()
s.add(2**x == 3)
print(s.check())    # UNKNOWN

# 遍历断言的约束
x = Real('x')
y = Real('y')
s = Solver()
s.add(x > 1, y > 1, Or(x + y > 3, x - y < 2))
print(f'asstered constraints:')
for c in s.assertions():
    print(' ' + c)
print(s.check())    # SAT
# print (s.statistics())
for k, v in s.statistics():
    print(f'{k} : {v}')
"""
decisions : 2           # 决策次数
final checks : 1        # 最终检查次数
mk clause binary : 1    # 创建二元子句
num checks : 1          # 检查次数
mk bool var : 4         # 创建布尔变量
arith-lower : 1         # 线性规划下界
arith-upper : 3         # 线性规划上界
arith-make-feasible : 3 # 线性规划可行性
arith-max-columns : 8   # 线性规划最大列数
arith-max-rows : 2      # 线性规划最大行数
num allocs : 54182769   # 内存分配次数
rlimit count : 1217     # 限制计数
max memory : 21.33      # 最大内存
memory : 20.33          # 内存
time : 0.004            # 时间
"""