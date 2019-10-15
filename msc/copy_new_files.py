import shutil
import os
import sys

def copy_new_files(input_file,output_path):
	with open(input_file,'r') as f:
		a=f.readlines()
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	for item in a:
		item_split=item.split('\t')
		print(item_split)
		path=item_split[0] + '/' + item_split[1]
		destination=output_path + '/' + item_split[-1].strip() + '.' + item_split[1].rsplit('.',1)[-1]
		if not os.path.exists(destination):
			shutil.copy(path,destination)
		

if __name__=="__main__":
	copy_new_files(sys.argv[1],sys.argv[2])
