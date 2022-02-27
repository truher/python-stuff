"""experiment with multiprocessing an enormous dataframe"""
import os
from multiprocessing import Pool
from multiprocessing import Manager
from numpy.random import default_rng
import numpy as np
import pandas as pd # type:ignore
import psutil # type:ignore

def worker_fn(wns, wstart, wend):
    """do some scoring"""
    print(f"worker {os.getpid()} {len(wns.df)} start {wstart} end {wend}")
    result_1 = []
    result_2 = []
    for df_row in range(wstart,wend):
        row = wns.df.iloc[[df_row]]
        # TODO pick cols by name?
        result_1.append(np.median(row.values))
        result_2.append(np.mean(row.values))
    return pd.DataFrame({'median': result_1, 'mean': result_2})

if __name__ == '__main__':
    #COLUMNS = 40
    COLUMNS = 4
    #ROWS = 100000000
    ROWS = 100
    CPU = 6

    big_dataframe = pd.DataFrame()
    rng = default_rng()
    for i in range(COLUMNS): # yields about 30GB
        print(f"i: {i:3d} memory (GB): {psutil.Process(os.getpid()).memory_info().rss/1e9:5.2f}")
        big_dataframe[f'key{i}'] = rng.standard_normal(ROWS)

    mgr = Manager()
    ns = mgr.Namespace()
    ns.df = big_dataframe
    partition_ranges = [[cpu * ROWS // CPU, (cpu + 1) * ROWS // CPU] for cpu in range(CPU)]
    with Pool(CPU) as p:
        y = p.starmap(worker_fn, [[ns, start, end] for start, end in partition_ranges])

    yy = pd.concat(y, axis=0).reset_index(drop=True)
    big_dataframe = pd.concat([big_dataframe, yy], axis=1)

    print("===== OUTPUT")
    print(big_dataframe)
