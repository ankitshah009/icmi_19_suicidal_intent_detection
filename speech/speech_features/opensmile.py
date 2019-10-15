import os
import sys
import json
import tempfile
import subprocess as sh

class OpenSmile():
    def __init__(self,
                indir,
                outdir,
                opensmile_path = "/share/workhorse3/mahmoudi/11776_Intervening_before_its_too_late/speech_features/SMILExtract",
                config_path = "/share/workhorse3/mahmoudi/11776_Intervening_before_its_too_late/speech_features/config"):

        self.indir = []
        self.outdir = []
        self.opensmile_path = opensmile_path
        self.config_paths = self.get_config_files(config_path)
        self.dirs = set()
        self.prep_data(indir, outdir)
        self.create_dirs()

    # returns a list of configs to use for each file
    def read_configs(self, config_path, configfiles_path):
        configs = []
        with open(configfiles_path, "r") as file:
            for line in file.readlines():
                configs.append(os.path.join(config_path.strip(), line.strip()))
        return configs
    
    # gets all the config files in a directory 
    def get_config_files(self, config_path):
        all_paths = []
        for root, dirs, files in os.walk(config_path):
            for f in files:
                if f.endswith(".conf"):
                    all_paths.append(os.path.join(root, f))
        return all_paths
                    
    # reads all files and index them
    def prep_data(self, indir, outdir):
        for root, dirs, files in os.walk(indir):
            for f in files:
                if f.endswith(".wav"):
                    filename = os.path.join(root, f)
                    self.indir.append(filename)
                    outfile = os.path.join(filename.replace(indir, outdir))
                    fname, ext = os.path.splitext(outfile)
                    outfile = fname + ".json"
                    self.outdir.append(outfile)
                    self.dirs.add(os.path.dirname(outfile))

    # creates all output directories
    def create_dirs(self):
        for d in self.dirs:
            if not os.path.exists(d):
                os.makedirs(d)

    # exposes data to process to main process
    def expose_data(self):
        return zip([self]*len(self.indir), self.indir, self.outdir)

    # writes json to disk
    def write2disk(self, data, outfile):
        result = data[0]
        for i in range(1, len(data)):
            result = dict(result, **data[i])

        # write json to file
        with open(outfile, "w") as fp:
            fp.write(json.dumps(result))
            # json.dumps(result, fp)

    # reads opensmile output file
    def read_outfile(self, f, config):
        lines = f.readlines()
        if len(lines) == 0 or len(lines) > 2:
            # print("ignoring the line...")
            return dict()
        #try:
        #    lines[2]
        #except:
        #    print("bugged")
        #    exit(1)
        #    return dict()
        # print(type(lines[0].strip()))
        feats = str(lines[0].strip()).split(";")
        vals = str(lines[1].strip()).split(";")
        feats = [config+"__"+x for x in feats] 
        corr = dict(zip(feats, vals))
        return corr

    def run(self, args):
        # extract all opensmile features from a single file
        output = []
        for config in self.config_paths:
            temp = tempfile.NamedTemporaryFile()
            # print(" ".join([self.opensmile_path, '-C', config, '-I', args[0], '-csvoutput', temp.name]))
            # print(config)
            sh.call([self.opensmile_path, '-C', config, '-I', args[0], '-csvoutput', temp.name, '-appendcsv', '0', '-noconsoleoutput', '1'])
            output.append(self.read_outfile(temp, os.path.basename(config)))
            temp.close()
        self.write2disk(output, args[1])


