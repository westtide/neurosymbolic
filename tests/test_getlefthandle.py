from z3 import *

def getLeftHandle2(PT):
    PT.decl()
    print(f'PT.decl() = {PT.decl()} ')
    """
    getLeftHandle函数的作用是在AST中找到最左边的"Handle"。
    这里的"Handle"是指可以被替换或者归约的部分，它的标志是节点名称以'non_'开头。
    函数通过递归的方式遍历AST，当找到一个节点名称以'non_'开头时，就返回这个节点，否则就继续遍历其子节点。
    在自底向上的语法分析过程中，"Handle"是指在待归约串中，可以按某个产生式直接归约的部分。
    也就是说，"Handle"是待归约串中与某个产生式右部匹配的那一部分。
    if PT = And(Bool('non_nd'), Or(Bool('non_p'), Bool('non_p')))
    return Bool('non_nd')

    /Users/westtide/opt/anaconda3/envs/py311-torch/bin/python /Users/westtide/Developer/LIPuS/tests/test_getlefthandle.py

    TemplateCenter: getLeftHandle = non_ in non_nd1
    pt = And, left = non_nd1
    
    TemplateCenter: getLeftHandle = non_ in non_p1
    pt = Or, left = non_p1
    
    TemplateCenter: getLeftHandle = non_ in non_p2
    pt = non_p2, left = non_p2
    
    TemplateCenter: getLeftHandle = non_ in non_p2
    pt = ==, left = non_p2
    
    TemplateCenter: getLeftHandle = non_ in non_p1
    pt = And, left = non_p1

    """
    if 'non_' in str(PT.decl()):    # decl() 方法用于获取构成该表达式的函数或操作符的名字
        print(f'')
        print(f'TemplateCenter: getLeftHandle = non_ in {PT.decl()}')
        return PT                   # 如果一个表达式的声明以 'non_' 开头，那么这个表达式就是一个非终结符，可以根据生成规则进一步展开。
    else:
        for child in PT.children():
            l = getLeftHandle2(child)
            if l is not None:
                return l
    return None
RULE = {
    # conjunction: 1元/2元/3元的，最高3元
    'non_nc': [And(Bool('non_nd')), And(Bool('non_nd'), Bool('non_nd')),
               And(Bool('non_nd'), Bool('non_nd'), Bool('non_nd'))],
    # disjunction: 1元/2元/3元的，最高3元
    'non_nd': [Or(Bool('non_p')), Or(Bool('non_p'), Bool('non_p')), Or(Bool('non_p'), Bool('non_p'), Bool('non_p'))],
    # predicate 谓词 p := t < s | t <= s | t == s
    'non_p': [Int('non_t') < Int('non_s'),
              Int('non_t') <= Int('non_s'),
              Int('non_t') == Int('non_s')],
    # t := term | term+term | term+term+term | term+term+term+term，最高4元的加法：可以合并吗？
    'non_t': [Int('non_term'),
              Int('non_term') + Int('non_term'),
              Int('non_term') + Int('non_term') + Int('non_term'),
              Int('non_term') + Int('non_term') + Int('non_term') + Int('non_term')],
    # term := v | s*v | s*v*v | s*v*v*v | s*v*v*v*v
    'non_term': [Int('non_v'),
                 Int('non_s') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v') * Int('non_v'),
                 Int('non_s') * Int('non_v') * Int('non_v') * Int('non_v') * Int('non_v')],
    # 'non_op1': [-Int('non_t')],  # 'Rule_op1_abs'],
    # s := undecided
    'non_s': [Int('undecided')],  # Int('non_decided')
    # 'non_decided': ['VALUE'],
    # v := [], 运行时初始化
    'non_v': []  # dynamically initialize this one
}

PTs = [And(Bool('non_nd'), Or(Bool('non_p2'), Bool('non_p3'))),
       Or(Bool('non_p'), Bool('non_p2')),
       Bool('non_p'),
Bool('non_p') == Bool('non_p2'),
       And(And(Bool('non_p'), Bool('non_p2')), Bool('non_p3'))]

for pt in PTs:
    print('')
    left = getLeftHandle2(pt)
    print(f'pt = {pt.decl()}, left = {left}')
    if RULE[str(left)]:
        print(f'RULE = {RULE[str(left)]}')