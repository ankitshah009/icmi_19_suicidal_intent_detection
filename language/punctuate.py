import argparse
import pandas as pd
import shlex
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='language_data/transcripts.tsv', help="data set to be used")
parser.add_argument('--dump_loc', type=str, default='language_data/punctuated_transcripts.tsv',
                    help="location of dumping")
parser.add_argument('--merged', type=str, default='language_data/Merged_updated.tsv', help="location of merged sheet")
args = parser.parse_args()


def url_to_video_id(url):
    url = url.split("=")[1].lower()
    for k, v in mapping.items():
        if url in k:
            return v


def punctuate(text):
    cmd = "curl -d \"text=" + text + "\" http://bark.phon.ioc.ee/punctuator"
    process = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    process.kill()
    return stdout.decode("utf-8")


def load(path):
    data = []
    with open(path, 'r') as file:
        for no, line in tqdm(enumerate(file)):
            url, _, _, transcript, _ = line.split("\t")
            video_id = url_to_video_id(url)
            if no == 63:
                # This one is already punctuated for some reason and it breaks the API
                punctuated_transcript = transcript
            else:
                punctuated_transcript = punctuate(transcript)
            tqdm.write(punctuated_transcript)
            data.append((video_id, punctuated_transcript))
    return data


def write_to_file(data):
    out = open(args.dump_loc, 'w')
    for vid, text in data:
        out.write(str(vid) + "\t" + text + "\n")
    out.close()


def main():
    # Load data
    print("Punctuating data...")
    data = load(args.data)
    write_to_file(data)
    print("Punctuating data...[OK]")


if __name__ == '__main__':
    # Read the merged sheet
    df = pd.read_csv(args.merged, delimiter="\t")
    mapping = {}
    for _, row in df.iterrows():
        mapping[row['Name']] = row['New Video Id']
    main()
