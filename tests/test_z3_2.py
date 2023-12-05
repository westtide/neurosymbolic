from z3 import *

s1 = Int('s1')
s2 = Int('s2')
c = Int('c')

solve(s1 >= -1, s2 >= -1, s1 <= 1, s2 <= 1,
      And((s1*100 + s2*0 <= c),
         And(Not(s1*0 + s2*1 <= c),
            And(Not(s1*0 + s2*2 <= c),
                Implies((s1*1 + s2*0 <= c), (s1*0 + s2*1 <= c))
        ))))
# [s1 = -1, s2 = -1, c = -3]