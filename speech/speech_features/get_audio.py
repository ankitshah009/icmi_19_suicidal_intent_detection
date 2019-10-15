import os
import sys
from pydub import AudioSegment 

# Extracts audio out of video
class GetAudio():
    def __init__(self, indir, outdir, out_format="wav", sample_rate=16000, channel_num=1):
        self.indir = []
        self.outdir = []
        self.out_format = out_format
        self.sample_rate = sample_rate
        self.channel_num = channel_num
        self.dirs = set()
        self.prep_data(indir, outdir)
        self.create_dirs()

    # reads all files and index them
    def prep_data(self, indir, outdir):
        for root, dirs, files in os.walk(indir):
            for f in files:
                filename = os.path.join(root, f)
                self.indir.append(filename)
                outfile = os.path.join(filename.replace(indir, outdir))
                fname, ext = os.path.splitext(outfile)
                outfile = fname + "." + self.out_format
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

    def run(self, args):
        # args[0] --> input 
        # args[1] --> output 
        try:
            audio_segment = AudioSegment.from_file(args[0])
            audio_segment = audio_segment.set_channels(self.channel_num)
            audio_segment = audio_segment.set_frame_rate(self.sample_rate)
            audio_segment.export(args[1], format=self.out_format)
        except:
            pass

