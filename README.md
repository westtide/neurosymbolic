# LIPuS

A Reinforcement-Learning-based loop invariant inference tool.

作者论文: Loop Invariant Inference through SMT Solving Enhanced Reinforcement Learning

Reference: Shiwen Yu, Ting Wang, and Ji Wang. 2023. Loop Invariant Inference through SMT Solving Enhanced Reinforcement Learning. In Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA ’23), July 17–21, 2023, Seattle, WA, USA. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3597926.3598047

原地址: https://zenodo.org/records/7909725

### Install

0. `git clone https://neurosymbolic.coding.net/p/lipus/d/LIPuS/git`, could take a long time.

1. prepare a python environment with python version=3.7.10
2. prepare `make` and `g++`. If you are using Windows, you need prepare the above two in git bash.
3. `cd /LIPuS/; pip install -r requirements.txt`
4. `cd /LIPuS/code2inv/graph_encoder/`
   1. if you are using Windows, delete the "Makefile" , and rename the "Makefile_win" as "Makefile" and run `make clean ; make` on git bash.
   2. if you are using Linux, just run `make clean ;make`

### Run

1. `cd /LIPuS/`
2. `python RunAllLinear.py` if you want to run all linear benchmarks
3. `python RunAllNonLinear.py` if you want to run all nonlinear benchmarks
4. The results can be checked in `Results` directory.
5. check out "main.py" if you want to run specific one benchmark.

### Benchmarks

1. All benchmarks are put in "Benchmarks/", each instance has three files: c source file, CFG json file, and SMT file.
2. If you want to add new instance, you only need to prepare the three files, and LIPuS will automatically do the rest.
3. As for how to prepare the CFG json file and SMT file, please refer to [Code2Inv](https://github.com/PL-ML/code2inv), which use Clang to do it automatically. Also, you can manually do it just like us.
