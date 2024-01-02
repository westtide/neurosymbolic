from z3 import *
a = Int('a')
b = Int('b')
s = Solver()
s.add(a+b == 88, a*b == 1095)
