from PT_generators.StaticAnalysis.Preprocess import preprocess_file
import sys
sys.path.extend([".", ".."])

benchmarks_c_path = r"/Users/westtide/Developer/LIPuS/Benchmarks/Linear/c/"


for i in range(1, 134):
    if i in [26, 27, 31, 32, 61, 62, 72, 75, 106]:  # not solvable
        continue
    # 创建文件路径

    cfilename = str(i) + ".c"
    path_c = benchmarks_c_path + cfilename
    print(f'i = {i} benchmark start: path_c = {cfilename}')
    preprocess_file(path_c)
