from z3 import *

x = Real('x')
y = Real('y')
z = Real('z')
s = Solver()
s.add(x > 1, y >1, x + y > 3, z - x < 10)
print(s.check())

# model
m = s.model()
print(f'traverse model: ')
for d in m.decls():
    print(f'{d.name()} = {m[d]}')
"""
SAT: Z3 finds a solution that satisfied the set of constraints
Solution: A model for the set of assertes constraints
Model: An interpretation that makes each asserted constraint true
traverse model: 
y = 2
x = 3/2
z = 0 
"""