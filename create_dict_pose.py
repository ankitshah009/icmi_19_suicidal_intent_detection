import csv
import numpy as np
import sys
def extract_pose(pose_csv,dict_name):
	with open(pose_csv,'r') as f:
		lines=f.readlines()
	d={}
	for line in lines:
		line_split=line.split(',',2)
		d[line_split[1]]=line_split[2].strip()
	np.savez(dict_name,d)		

if __name__=="__main__":
	extract_pose(sys.argv[1],sys.argv[2])
