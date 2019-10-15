import os
import sys
import csv
import collections
def new_sub_dict(dict_all,i,values):
	d1={}
	d1['Distress (0) vs Suicide (1)'] = values[i]
	for key,value in dict_all.items():
		d1[key]=value[i]					
	return d1


def plot(all_text_files,tsv):
	with open(all_text_files,'r') as f:
		all=f.readlines()
	with open(tsv,'r') as f:
		tsvs=f.readlines()
	all_1=[]
	id_mapping={}
	for i,item in enumerate(tsvs):
		i1=item.split('\t')
		print(i1[0],i1[16])
		print(type(i1[16]))
		if i > 0:
			id_mapping[i1[0]]=int(i1[16])
	print(id_mapping)
	for item in all:
		all_1.append(item.strip('\n'))
	#print(all_1)
	dict_suicide={}
	dict_depressed={}
	dict_all={}
	for i in range(len(all)):
		#print(all_1[i])			
		filename=str(int(all_1[i].split('_')[0]))
		with open(all_1[i],'r') as f:
			lines=f.readlines()
		for j,line in enumerate(lines):
			line=line.strip()
			line_split=line.split('\t')
			#print(line_split)
			if id_mapping[filename] == 0:
				if j in dict_depressed.keys():
					dict_depressed[j].append(line_split[1])		
				else:
					dict_depressed[j]=[line_split[1]]
			else:
				if j in dict_suicide.keys():
					dict_suicide[j].append(line_split[1])
				else:
					dict_suicide[j]=[line_split[1]]
			if j in dict_all.keys():
				dict_all[j].append(line_split[1])
			else:
				dict_all[j]=[line_split[1]]
	#print(dict_suicide)
	#print(dict_depressed)			
	print(dict_all)
	values=[]
	#id_sorted={int(k):v for k,v in id_mapping.items()}
	#print
	#for k,v in id_sorted.items():
	#	print(k,v)
	sorted_dict=collections.OrderedDict(sorted(id_mapping.items(),key=lambda x:int(x[0])))
	for k,v in sorted_dict.items():
		print(k,v)
		values.append(v)
	print(values)
	with open('merged_pose.csv','w') as f:
		fieldnames=list(dict_all.keys())
		fieldnames.insert(0,'Distress (0) vs Suicide (1)')
		writer=csv.DictWriter(f,fieldnames=fieldnames)
		writer.writeheader()
		for i in range(89):
			d1=new_sub_dict(dict_all,i,values)			
			writer.writerow(d1)
		
if __name__=="__main__":
	plot(sys.argv[1],sys.argv[2])
