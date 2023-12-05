from z3 import *

def funa():
    x = Int('x')
    y = Int('y')

    solve(And(Or(-2*y*y + 0*y*y + 0*y*y + 0*y*y <= 0,
           0*y*y + 0*y*y + -3*y*y + 0*y*y <= 0,
           0*y*y + 0*y*y + -2*y*y + 0*y*y <= 0),
        Or(0*y*y + 0*y*y + -2*y*y + 0*y*y <= 0,
           0*y*y + -2*y*y + 0*y*y + 0*y*y <= 0,
           0*y*y + -3*y*y + 0*y*y + 0*y*y <= 0),
        Or(-1*y*y + 2*y*y + 0*y*y + 1*y*y <= 0,
           0*y*y + 0*y*y + 2*y*x + x <= 0,
           x < 99999)),
          )

    # [y = 196331, x = 90837]
    # [y = 131064, x = 3]


    solve( And(Or(3*y*y*y + 3*y*y*y + 3*y*y*y <= 3,
           3*y*y*y + 3*y*y*y + 3*y*y*y <= 3,
           0*x + 1*x + 0*x + 0*x <= 99991))
    )

funa()