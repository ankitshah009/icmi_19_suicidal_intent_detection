import os
import sys
from multiprocessing import Pool

# NUMBER OF PROCESSES TO RUN 
N_PROCESSES = 8

def run(args):
    args[0].run(args[1:])

# responsible for running the commands
def run_command(indir, outdir, cmd):
    cmd2run = None

    if cmd == "resample":
        cmd2run = Resample(indir, outdir)

    if cmd == "opensmile":
        from opensmile import OpenSmile
        cmd2run = OpenSmile(indir, outdir)

    if cmd == "get_audio":
        from get_audio import GetAudio
        cmd2run = GetAudio(indir, outdir)

    print("Got the data, about to process...")
    pool = Pool(N_PROCESSES)
    print("Processing the data...")
    pool.map(run, cmd2run.expose_data())
    print("Done processing the data\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("USAGE: extract_script.py <input_dir> <output_dir> <command>")
        exit()
    run_command(sys.argv[1], sys.argv[2], sys.argv[3])
