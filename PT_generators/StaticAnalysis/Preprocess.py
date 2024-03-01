#!/usr/bin/env bash
# MIT License
# Copyright (c) 2023 <westtide>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# #############################################
# @Author    :   westtide
# @Contact   :   tocokeo@outlook.com
# @Date      :   2024/1/16
# @License   :   MIT License
# @Desc      :   结合 Pre, Post 条件, 推理 Loop Inv 的上近似
# #############################################

import os
import re
import sys

sys.path.extend([".", ".."])
from pycparser import c_parser, c_ast, parse_file, c_generator

filename = "3.c"
base1 = """int main(){ """
base2 = """}"""


def find_func_calls(ast, exp_assume, exp_assert):
    if isinstance(ast, c_ast.FuncCall):
        # 如果 name == assume, 提取表达式
        if ast.name.name == "assume":
            exp_assume.append(ast.args)
        # 如果 name == assert, 提取表达式
        if ast.name.name == "assert":
            exp_assert.append(ast.args)
        print(f"函数调用: {ast.name.name}")

    for _, child in ast.children():
        find_func_calls(child, exp_assume, exp_assert)


# 用于遍历AST节点并查找for循环
def find_for_loops(node):
    if isinstance(node, c_ast.For):
        print("找到一个for循环")
        # 可以进一步分析循环的初始化、条件和迭代部分
    for _, child in node.children():
        find_for_loops(child)


# 用于遍历AST节点并查找while循环
def find_while_loops(node):
    if isinstance(node, c_ast.While):
        print("找到一个while循环")
        # 可以进一步分析循环的条件部分
    for _, child in node.children():
        find_while_loops(child)


def ast2exp(type, astlist):
    for item in astlist:
        generator = c_generator.CGenerator()
        expr = c_ast.ExprList(item)
        print(f'{type}: {generator.visit(expr)}')


def exp2ccode(pre_exp, loop_exp, post_exp):
    ccode = ["", "", ""]
    # pre-condition
    ccode[0] = base1 + pre_exp + base2
    # loop body
    ccode[1] = base1 + loop_exp + base2
    # post-condition
    ccode[2] = base1 + post_exp + base2
    return ccode


def process_all_node(ccode):
    for i in range(0, 3):
        parser = c_parser.CParser()
        astnode = parser.parse(text=ccode[i], filename='<none>')
        file_ast = astnode

        exp_assume = []
        exp_assert = []

        # 使用定义的函数遍历AST
        find_func_calls(file_ast, exp_assume, exp_assert)
        # 使用c_generate将ast转换为exp，输出 assume
        ast2exp("exp_assume", exp_assume)
        # 使用c_generate将ast转换为exp，输出所有的 assert
        ast2exp("exp_assert", exp_assert)
        # 使用定义的函数遍历AST
        find_for_loops(file_ast)
        # 使用定义的函数遍历AST
        find_while_loops(file_ast)


def process_nude(content):
    print("into: process_nude")
    pre_exp = ""
    loop_exp = ""
    post_exp = ""

    # 提取 pre-conditions 部分的代码
    pre_conditions = re.search('int main\s*\(\)\s*\{(.*?)while', content, re.DOTALL)
    if pre_conditions:
        pre_exp = pre_conditions.group(1).lstrip('{')
        print(f'pre_conditions = {pre_conditions.group(1)}')

    # 提取 loop body 部分的代码: 定义一个用于遍历AST节点并查找while循环的函数
    def find_while_loops(node):
        if isinstance(node, c_ast.While):
            print("找到一个while循环")
            loop_exp = node.stmt
            # node.show()
            # print(f'loop_body: {node.stmt}')
            # 可以进一步分析循环的条件部分
        for _, child in node.children():
            find_while_loops(child)

    # 遍历AST
    parser = c_parser.CParser()
    # parser.parse() 不支持注释行，使用 re 去除所有注释行
    astnode = parser.parse(text=re.sub(r'//.*$', '', content, flags=re.MULTILINE), filename='<none>')
    find_while_loops(astnode)

    # 提取 post-condition 部分的代码
    post_condition = re.search('// post-condition(.*?)(?=})', content, re.DOTALL)
    if post_condition:
        post_exp = post_condition.group(1)
        print(f'post_condition: {post_condition.group(1)}')

    ccode = exp2ccode(pre_exp, loop_exp, post_exp)
    process_all_node(ccode)


def process_with_comment(content):
    print("into: process_with_comment")
    pre_exp = ""
    loop_exp = ""
    post_exp = ""

    # 提取 pre-conditions 部分的代码
    pre_conditions = re.search('// pre-conditions(.*?)// loop body', content, re.DOTALL)
    if pre_conditions:
        pre_exp = pre_conditions.group(1)
        print(f'pre_conditions = {pre_conditions.group(1)}')

    # 提取 loop body 部分的代码
    loop_body = re.search('// loop body(.*?)// post-condition', content, re.DOTALL)
    if loop_body:
        loop_exp = loop_body.group(1)
        print(f'loop_body: {loop_body.group(1)}')

    # 提取 post-condition 部分的代码
    post_condition = re.search('// post-condition(.*?)(?=})', content, re.DOTALL)
    if post_condition:
        post_exp = post_condition.group(1)
        print(f'post_condition: {post_condition.group(1)}')

    ccode = exp2ccode(pre_exp, loop_exp, post_exp)
    process_all_node(ccode)


def preprocess_file(filename):
    with open(filename, "r") as f:
        code = f.read()

    with open(filename, "r") as f:
        containPre = False
        containLoop = False

        for line in f:

            if "// pre-conditions" in line:
                containPre = True

            if "// loop body" in line:
                containLoop = True

            if containPre and containLoop:
                print(f'{filename} contains pre-conditions and loop')
                break

        if not containPre and not containLoop:
            process_nude(code)

        if containPre and containLoop:
            process_with_comment(code)


preprocess_file(filename)