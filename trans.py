
import os
import json
import argparse
from PIL import Image

def read_jsonFile(input_json_dir,fname):
	info_list = []
	path = os.path.join(input_json_dir,fname)
	with open(path, 'r',encoding="utf-8") as f:
		json_i = json.load(f)
	shapes = json_i['shapes']
	for i in range(len(shapes)):
		temp = {}
		temp['group_id']=shapes[i]['group_id']
		temp['points']=shapes[i]['points']
		temp['label']=shapes[i]['label']
		info_list.append(temp)

	return info_list

def getBboxInfo(i_list):
	# 0:中文字串 1:中文單字 2:英數字串 3:中英數字串 4:中文單字字串 5:其他 255:don't care
	allBbox = []
	for info in i_list:
		x1,y1 = round(info['points'][0][0]),round(info['points'][0][1])
		x2,y2 = round(info['points'][1][0]),round(info['points'][1][1])
		x3,y3 = round(info['points'][2][0]),round(info['points'][2][1])
		x4,y4 = round(info['points'][3][0]),round(info['points'][3][1])
		allBbox.append((x1,y1,x2,y2,x3,y3,x4,y4,info['label'],info['group_id']))
		
	return allBbox

def writeToFile(output_path,file_name, result):
	path = os.path.join(output_path,file_name)
	with open(path, "w", encoding="utf-8") as writeFile:  
		for box in result:
			string = ",".join(str(p) for p in box)+"\n"
			writeFile.write(string)
		writeFile.close()
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_json_dir',type=str,default="./datasets/AICUP1/train_jsons",help='path of GT json directory')
	parser.add_argument('--output_txt_dir',type=str,default="./datasets/AICUP1/train_gts",help='path of GT txt directory')
	parser.add_argument('--image_list',type=str,default="./datasets/AICUP1/train_list.txt",help='path of image list')
	arg = parser.parse_args()

	json_list = os.listdir(arg.input_json_dir)

	#create output directory
	if not os.path.exists(arg.output_txt_dir):
		os.mkdir(arg.output_txt_dir)
		
	for json_file in sorted(json_list):
		outputFileName = json_file.replace('json', 'jpg')+'.txt'
		info_list = read_jsonFile(arg.input_json_dir,json_file)
		bboxlist = getBboxInfo(info_list)
		writeToFile(arg.output_txt_dir,outputFileName, bboxlist)

		#create image list txt file
		with open(arg.image_list,'a',encoding="utf-8") as f:
			f.write(json_file.replace('json', 'jpg')+'\n')
