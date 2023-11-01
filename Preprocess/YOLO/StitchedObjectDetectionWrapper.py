import argparse
import os
from tqdm import tqdm 
import time

parser = argparse.ArgumentParser()
curpath=os.getcwd()
parser.add_argument('--source', required=True, help='The directory where the stitched projection of all frames are.')
parser.add_argument('--output',required=True,help='The file where metadata will be stored')
args=parser.parse_args()

path, dirs, files = next(os.walk(args.source))
file_count = len(files)

file_count = 26

for i in tqdm(range(file_count)):
	filename = str(i) + ".png"
	if i % 25 == 0:  # draw every 25th frame
		os.system("python yolo.py --image-path=" + args.source + "/" + filename + " --storefilename " + args.output + " --framenum " + str(i) + " --draw")
	else:
		os.system("python yolo.py --image-path=" + args.source + "/" + filename + " --storefilename " + args.output + " --framenum " + str(i))

# i = 1
# filename = str(i) + ".png"

# os.system("python yolo.py --image-path=" + args.source + "/" + filename + " --storefilename " + args.output + " --framenum " + str(i))


