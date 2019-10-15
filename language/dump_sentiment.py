import argparse
import shlex
import subprocess
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='language_data/punctuated_transcripts.tsv', help="data set to be used")
parser.add_argument('--dump_loc', type=str, default='language_data/sentiments/',
                    help="location of dumping")
args = parser.parse_args()


def get_sentiment(text):
    cmd = "curl -d \"text=" + text + "\" http://text-processing.com/api/sentiment/"
    process = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    process.kill()
    return stdout.decode("utf-8")


def load(path):
    with open(path, 'r') as file:
        for no, line in enumerate(file):
            vid, transcript = line.split("\t")
            sentiments = {}
            sentences = transcript.lower().split(".")
            for sentence in tqdm(sentences):
                sentence = sentence.strip()
                if sentence != '':
                    ss = get_sentiment(sentence)
                    sentiments[sentence] = json.loads(ss)
            print(no + 1, sentiments)
            write_to_file(vid, sentiments)


def write_to_file(vid, sentiments):
    with open(args.dump_loc + str(vid) + '.json', 'w') as out:
        json.dump(sentiments, out)


def main():
    # Load data
    print("Getting sentiment for sentences in data...")
    load(args.data)
    print("Getting sentiment for sentences in data...[OK]")


if __name__ == '__main__':
    main()
