import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='language_data/sentiments/', help="data set to be used")
parser.add_argument('--dump_loc', type=str, default='language_data/negative_sentences/',
                    help="location of dumping")
args = parser.parse_args()

THETA = 0.8


def load(path):
    for filename in os.listdir(path):
        vid = filename.replace(".json", "")
        negative_sentences = []
        with open(path + filename) as json_file:
            data = json.load(json_file)
            for sentence, sentiment in data.items():
                if sentiment['label'] == 'neg' and sentiment['probability']['neg'] >= THETA:
                    negative_sentences.append(sentence)
        write_to_file(vid, negative_sentences)


def write_to_file(vid, sents):
    if len(sents) == 0:
        sents.append("neutral")
    with open(args.dump_loc + vid + '.txt', 'w') as out:
        for sent in sents:
            out.write(sent + "\n")


def main():
    # Load data
    print("Filter negative sentences...")
    negative_scores = load(args.data)
    #mu = np.mean(negative_scores)
    #std = np.std(negative_scores)
    #print("Mean: {}".format(mu))    # 0.679
    #print("Std: {}".format(std))    # 0.98
    #print("Suggested Threshold: {}".format(mu + 1.5 * std)) # 0.826
    print("Filter negative sentences...[OK]")


if __name__ == '__main__':
    main()
