""" Experiment with multiprocessing an enormous dataframe. """
import os
from multiprocessing import Pool
from numpy.random import default_rng
import numpy as np
import pandas as pd # type:ignore
import psutil # type:ignore
from multiprocessing import set_start_method

def worker_fn(chunk):
    """ Scoring goes here. """
    result_1 = []
    result_2 = []
    # TODO pick cols by name.
    for i,row in chunk.iterrows():
        result_1.append(np.median(row.values))
        result_2.append(np.mean(row.values))
    print(f"worker {os.getpid()} finished "
          f"RSS (GB) {psutil.Process(os.getpid()).memory_info().rss/1e9:5.2f} "
          f"start {min(chunk.index):10d} end {max(chunk.index):10d}")
    return (result_1, result_2)

def chunks(x, chunk_size):
    for i in range(0, len(x), chunk_size):
        yield x[i:i + chunk_size]

def make_df(cols = 40, rows = 100000000):
    big_df = pd.DataFrame()
    rng = default_rng()
    for i in range(cols):
        print(f"column: {i:3d} RSS (GB): {psutil.Process(os.getpid()).memory_info().rss/1e9:5.2f}")
        big_df[f'key{i}'] = rng.standard_normal(rows)
    return big_df

if __name__ == '__main__':
    set_start_method("spawn")
    big_dataframe = make_df()
    with Pool(processes = 6) as pool:
        result_shards = pool.imap(worker_fn, chunks(big_dataframe, 10000))
        result_columns = np.concatenate(list(result_shards), axis=1)
        big_dataframe['median'] = result_columns[0]
        big_dataframe['mean'] = result_columns[1]
        print("===== OUTPUT")
        print(big_dataframe)
