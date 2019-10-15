import os,json,sys
import pandas as pd
import math
def read_jsons(path):
	json_files=[i for i in sorted(os.listdir(path)) if i.endswith('json')]
	#print(json_files)
	json_data=pd.DataFrame(columns=['x1','y1','prob'])
	l1=[]
	for i,json_file in enumerate(json_files):
		with open(os.path.join(path,json_file)) as jf:
			data=json.load(jf)
			part=data['part_candidates']
			l1.append(part)
	diff=[]
	for i1,i2 in zip(l1,l1[1:]):
		k=[{x: [(ai-bi)**2 for ai,bi in zip(i2[0][x],i1[0][x])] for x in i2[0] if x in i1[0]}]
		diff.append(k)
	with open('all_video_resolution.list','r') as f:
		all=f.readlines()
	all_split=[]
	for item in all:
		all_split_instance=item.strip().split('x')
		all_split.append(all_split_instance)
	sum={}
	frame_count={}
	#print(all_split)
	path_file=path.rsplit('/',2)[-2]
	#nprint(path)
	print(path_file)
	file_number=int(path_file.rsplit('_',2)[0])
	if file_number>36:
		file_number=file_number-1
	resolution=all_split[file_number-1]
	print(resolution)
	s1=0
	#pdb.set_trace()
	for item in diff:
		for key in item[0]:
			s1=0
			if key in sum:
				for i in range(2):
					if len(item[0][key])>0:
						if i==0:
							frame_count[key]+=1
						item[0][key][i] = float(item[0][key][i])/(float(resolution[i])**2)
						s1+=item[0][key][i]
				sum[key]+=math.sqrt(s1)
			else:
				for i in range(2):
					if len(item[0][key])>0:
						if i==0:
							frame_count[key]=1
						item[0][key][i] = float(item[0][key][i])/(float(resolution[i])**2)
						s1+=item[0][key][i]
					else:
						frame_count[key]=0
				sum[key]=math.sqrt(s1)
	for key in sum:
		with open(path_file + '.txt','a') as f:
			f.write(str(frame_count[key])+'\t' + str(sum[key]) + '\n' )
	#for key in sum:
	#	xsum=0
	#	ysum=0
	#	psum=0
	#	for i in range(len(sum[key])):
	#		#if i%3==0:
	#		#	xsum+=sum[key][i]
	#		#if i%3 ==1:
	#		#	ysum+=sum[key][i]
	#		#if i%3==2:
	#		#	psum+=sum[key][i]
	#	print(i,xsum,ysum,psum)	
	#	with open(path_file + '.list','a') as f:
	#		f.write(str(i)+'\t' + str(xsum) + '\t' + str(ysum) + '\t' + str(psum) + '\n')
		#print(len(sum[key]))
		#print(key)
	#print(sum)			
	

if __name__=="__main__":
	read_jsons(sys.argv[1])
