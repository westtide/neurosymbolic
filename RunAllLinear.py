from main import main
from loginit import logger

# CUDA_VISIBLE_DEVICES="0"
if __name__ == "__main__":
    logger.info(f'start: RUNALLLINEAR')
    result_file = open(r"Result/RL_Prunning_Linear.txt", "w")
    benchmarks_c_path = r"Benchmarks/Linear/c/"
    benchmarks_g_path = r"Benchmarks/Linear/c_graph/"
    benchmarks_s_path = r"Benchmarks/Linear/c_smt2/"
    results = []

    for i in range(1,134):
        if i in [26, 27, 31, 32, 61, 62, 72, 75, 106]:  # not solvable
            continue
        # 创建文件路径
        gfilename = str(i) + ".c.json"
        path_g = benchmarks_g_path + gfilename
        sfilename = str(i) + ".c.smt"
        path_s = benchmarks_s_path + sfilename
        cfilename = str(i) + ".c"
        path_c = benchmarks_c_path + cfilename
        logger.info(f'i = {i} benchmark start: path_c = {path_c}, path_g = {path_g}, path_s = {path_s}')


        timeUsed, answer = main(path_c, path_g, path_s)

        results.append(timeUsed)
        result_file.write(str(timeUsed) + '\t' + str(answer) + '\n')
        result_file.flush()

    result_file.close()
    print(results)
