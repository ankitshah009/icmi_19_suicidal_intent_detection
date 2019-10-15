import os
import argparse
import pandas as pd
import shlex
import subprocess
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--bert', type=str, default='language_data/trimmed/bert_features/', help="data set to be used")
parser.add_argument('--sentiments', type=str, default='language_data/trimmed/avg_neg_sentiments.npy', help="old punctuated")
parser.add_argument('--dump_loc', type=str, default='language_data/trimmed/trimmed_language_features.npz',
                    help="location of dumping")
args = parser.parse_args()


def load(path):
    f = {}
    for filename in tqdm(os.listdir(path)):
        video_id = int(filename.replace(".npy", ""))
        if video_id not in f:
            f[video_id] = {}
        f[video_id]['bert'] = np.load(path + filename)

    for vid, sentiment in np.load(args.sentiments).item().items():
        f[vid]['sentiment'] = sentiment
    np.savez(args.dump_loc, f)


def main():
    # Load data
    print("Merging...")
    load(args.bert)
    print("Merging...[OK]")


if __name__ == '__main__':
    main()
