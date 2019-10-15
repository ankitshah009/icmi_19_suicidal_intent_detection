import numpy as np
import os
import sys

def time_sec(input):
	splitted=input.split(':')
	return str(int(splitted[1]) + int(int(splitted[0])*60))

def ffmpeg_trim(input,start_time,end_time,output,output1):
	#cmd='ffmpeg -i %s -vcodec copy -acodec copy -ss %s -t %s %s' %(input,start_time,str(int(end_time)-int(start_time)),output)
	cmd='ffmpeg -y  -i %s -ss %s -t %s -c:v copy -c:a copy %s' %(input,start_time,str(int(end_time)-int(start_time)),output)
	audio_conversion_cmd='ffmpeg -y -i %s -vn -c:a copy -ss %s -t %s %s' %(input,start_time,str(int(end_time)-int(start_time)),output1)
	#print(cmd)
	os.system(cmd)
	os.system(audio_conversion_cmd)

def new_videos_trimmed(input_file,directory_files,trimmed_directory,audio_trimmed_directory):
	with open(input_file,'r') as f:
		a=f.readlines()
	if not os.path.exists(trimmed_directory):
		os.makedirs(trimmed_directory)
	if not os.path.exists(audio_trimmed_directory):
		os.makedirs(audio_trimmed_directory)
	for item in a:
		line_split=item.split('\t')
		extension='.'+line_split[1].rsplit('.',1)[-1]
		filename=line_split[-1].strip()
		new_filename=directory_files + '/' + filename + extension
		start_time=line_split[23]
		end_time=line_split[24]
		start_time_new = time_sec(start_time)
		end_time_new=time_sec(end_time)
		print(start_time_new,end_time_new)
		output_filename=trimmed_directory + '/' + filename + '_' + start_time_new + '_' + end_time_new + extension
		output_filename1=audio_trimmed_directory + '/' + filename + '_' + start_time_new + '_' + end_time_new + '.wav'
		ffmpeg_trim(new_filename,start_time_new,end_time_new,output_filename,output_filename1)	

if __name__=="__main__":
	new_videos_trimmed(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
