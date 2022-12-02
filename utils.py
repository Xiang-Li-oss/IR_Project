import pandas as pd
import numpy as np

def load_tsv(path):
    tsv_file = pd.read_csv(path, sep='\t', header=None)
    data = np.array(tsv_file)
    return data

def load_source(path):
    source = load_tsv(path)
    pairs = dict(zip(source[:,0], source[:,1]))
    return pairs
