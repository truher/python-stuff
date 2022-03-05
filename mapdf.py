"""
Classify the candidates (parallel map).

On my 6-core machine, runs about 1.2M row/sec
"""
from multiprocessing import Pool
from multiprocessing import set_start_method
import pickle
import pandas as pd # type:ignore

INPUT_FILE = 'all-similarities-rescored.csv'
#INPUT_FILE = 'all-similarities-rescored-test.csv'
CHUNK_SIZE = 10000
#CHUNK_SIZE = 2
#OUTPUT_FILE = 'predictions.csv'
OUTPUT_FILE = 'predictions-test.csv'
MODEL_FILE = 'model2.pkl'

def worker_fn(scores_chunk_df):
    """accept a chunk from the main reader, map each col, and append to the output"""
    x_data = scores_chunk_df.drop(columns=scores_chunk_df.columns[0:2])
    predictions = scores_chunk_df.loc[:,['left_index','right_index']]
    del scores_chunk_df
    with open(MODEL_FILE, 'rb') as model_f:
        model = pickle.load(model_f)
    predictions['prediction_prob'] = model.predict_proba(x_data.values)
    predictions = predictions[predictions['prediction_prob'] > 0.5]
    with open(OUTPUT_FILE, 'a', encoding='utf8') as output_f:
        predictions.to_csv(output_f, float_format='%.4f', index=False, header=False)
        output_f.flush()

if __name__ == '__main__':
    set_start_method('spawn')

    result_df = pd.DataFrame(columns=['left_index','right_index', 'prediction_prob'])
    with open(OUTPUT_FILE, 'w', encoding='utf8') as output_header_f :
        result_df.to_csv(output_header_f, index=False)
        output_header_f.flush()
    scores_reader = pd.read_csv(INPUT_FILE, index_col=0, chunksize=CHUNK_SIZE)
    with Pool(processes = 6) as pool:
        res_list = []
        # use this loop with apply to avoid exhausting the iterator in map()
        for chunk in scores_reader:
            res_list.append(pool.apply_async(worker_fn, (chunk,)))
        pool.close()
        pool.join()
