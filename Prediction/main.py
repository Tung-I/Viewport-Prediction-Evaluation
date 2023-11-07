import numpy as np 
import math
import pickle
import random
import argparse
import json
from parima import build_model
from bitrate import alloc_bitrate
from qoe import calc_qoe
import os

VIEW_PATH = '/home/tungi/datasets/vr_dataset/viewport/'
OBJ_PATH = '/home/tungi/datasets/vr_dataset/object_tracking/'

PRED_PATH = '/home/tungi/datasets/vr_dataset/viewport/6'


def get_data(data, frame_nos, dataset, topic, usernum, fps, milisec, width, height, view_width, view_height):

	######################
	offset = 0
	#######################

	# view_info: list of tuples (timestamp, (x,y))
	# ex. [0.0, (246.21661637948282, 55.39030055772059)], [0.021, (246.21661637948282, 55.39030055772059)], ...
	# obj_info: dictionary of dictionaries {frame_number: {object_id: (x,y)}}
	# ex. obj_info[0] = {0: array([1466,  703]), 1: array([2167,  703]), 2: array([2091,  702]), 3: array([1350,  716]), 4: array([360, 727]), 5: array([1845,  706])}
	obj_info = np.load(OBJ_PATH + 'ds{}_topic{}.npy'.format(dataset, topic), allow_pickle=True,  encoding='latin1').item()
	view_info = pickle.load(open(VIEW_PATH + '{}/viewport_ds{}_topic{}_user{}'.format(topic, dataset, topic, usernum), 'rb'), encoding='latin1')

	n_objects = []
	for i in obj_info.keys():
		try:
			n_objects.append(max(obj_info[i].keys()))
		except:
			n_objects.append(0)
	total_objects=max(n_objects)

	if dataset == 1:
		max_frame = int(view_info[-1][0]*1.0*fps/milisec)

		for i in range(len(view_info)-1):
			frame = int(view_info[i][0]*1.0*fps/milisec)
			frame += int(offset*1.0*fps/milisec)

			frame_nos.append(frame)
			if(frame > max_frame):
				break
			X={}
			X['VIEWPORT_x']=int(view_info[i][1][1]*width/view_width)
			X['VIEWPORT_y']=int(view_info[i][1][0]*height/view_height)
			for j in range(total_objects):
				try:
					centroid = obj_info[frame][j]

					if obj_info[frame][j] == None:
						X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
						X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)
					else:
						X['OBJ_'+str(j)+'_x']=centroid[0]
						X['OBJ_'+str(j)+'_y']=centroid[1]

				except:
					X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
					X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)


			data.append((X, int(view_info[i+1][1][1]*width/view_width),int(view_info[i+1][1][0]*height/view_height)))

	elif dataset == 2:
		# Find the frame number corresponding to the offset
		# ex., 60 sec duration, 29 fps, min_index=1 and max_fram=1739
		duration = 100
		for k in range(len(view_info)-1):
			if view_info[k][0]<=offset+duration and view_info[k+1][0]>offset+duration:
				max_frame = int(view_info[k][0]*1.0*fps/milisec)
				break
		min_index = 1
		for k in range(len(view_info)-1):
			if view_info[k][0]<=offset and view_info[k+1][0]>offset:
				min_index = k+1
				break
		

		prev_frame = 0
		for i in range(min_index, len(view_info)-1):
			# get the current frame number
			frame = int((view_info[i][0])*1.0*fps/milisec)
			if frame == prev_frame:
				continue
			if(frame > max_frame):
				break
			# insert the frame number into the list
			frame_nos.append(frame)
		
			X={}
			X['VIEWPORT_x']=int(view_info[i][1][1]*width/view_width)
			X['VIEWPORT_y']=int(view_info[i][1][0]*height/view_height)
			print("X['VIEWPORT_x'] = " + str(X['VIEWPORT_x']))
			for j in range(total_objects):
				try:
					centroid = obj_info[frame][j]

					if obj_info[frame][j] == None:
						X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
						X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)
					else:
						X['OBJ_'+str(j)+'_x']=centroid[0]
						X['OBJ_'+str(j)+'_y']=centroid[1]

				except:
					X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
					X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)


			data.append((X, int(view_info[i+1][1][1]*width/view_width),int(view_info[i+1][1][0]*height/view_height)))
			prev_frame = frame
			
	return data, frame_nos, max_frame, total_objects



def main():

	parser = argparse.ArgumentParser(description='Run PARIMA algorithm and calculate QoE of a video for a single user')

	parser.add_argument('-D', '--dataset', type=int, required=True, help='Dataset ID (1 or 2)')
	parser.add_argument('-T', '--topic', required=True, help='Topic in the particular Dataset (video name)')
	parser.add_argument('--fps', type=int, required=True, help='fps of the video')
	parser.add_argument('-O', '--offset', type=int, default=0, help='Offset for the start of the video in seconds (when the data was logged in the dataset) [default: 0]')
	parser.add_argument('-U', '--user', type=int, default=0, help='User ID on which the algorithm will be run [default: 0]')
	parser.add_argument('-Q', '--quality', required=True, help='Preferred bitrate quality of the video (360p, 480p, 720p, 1080p, 1440p)')

	args = parser.parse_args()

	if args.dataset != 1 and args.dataset != 2:
		print("Incorrect value of the Dataset ID provided!!...")
		print("======= EXIT ===========")
		exit()

	# Get the necessary information regarding the dimensions of the video
	print("Reading JSON...")
	file = open('./meta.json', )
	jsonRead = json.load(file)

	width = jsonRead["dataset"][args.dataset-1]["width"]
	height = jsonRead["dataset"][args.dataset-1]["height"]
	view_width = jsonRead["dataset"][args.dataset-1]["view_width"]
	view_height = jsonRead["dataset"][args.dataset-1]["view_height"]
	milisec = jsonRead["dataset"][args.dataset-1]["milisec"]

	pref_bitrate = jsonRead["bitrates"][args.quality]
	ncol_tiles = jsonRead["ncol_tiles"]
	nrow_tiles = jsonRead["nrow_tiles"]
	player_width = jsonRead["player_width"]
	player_height = jsonRead["player_height"]

	# player_tiles_x is the number of tiles in the x direction that the player can see
	player_tiles_x = math.ceil(player_width*ncol_tiles*1.0/width)
	player_tiles_y = math.ceil(player_height*nrow_tiles*1.0/height)
	
	# Initialize variables
	pred_nframe = args.fps  # prediction window size
	data, frame_nos = [],[]


	
	####################################################################################
	####### PARIMA

	print("PARIMA ...")
	# Read Data
	print("Reading Viewport Data and Object Trajectories...")
	data, frame_nos, max_frame, tot_objects = \
		get_data(data, frame_nos, args.dataset, args.topic, args.user, args.fps, \
		milisec, width, height, view_width, view_height)
	print("Data read\n")

	print("Build Model...")  # frame_nos: the list frame numbers; tot_objects: the number of objects
	act_tiles, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae, \
		gof, chunk_itm_xy_pred, chunk_final_xy_pred, chunk_gt_xy = \
		build_model(data, frame_nos, max_frame, tot_objects, width, height, \
		nrow_tiles, ncol_tiles, args.fps, pred_nframe)
	
	# print out the dimensions of each array
	print("gof: " + str(gof.shape))
	# print("chunk_itm_xy_pred: " + str(chunk_itm_xy_pred.shape))
	print("chunk_final_xy_pred: " + str(chunk_final_xy_pred.shape))
	print("chunk_gt_xy: " + str(chunk_gt_xy.shape))

	# Save the output np arrays to PRED_PATH
	OUTPUT_PATH = os.path.join(PRED_PATH, 'PARIMA')
	np.save(OUTPUT_PATH + f'/gof_{args.user}.npy', chunk_frames)
	# np.save(PRED_PATH + f'/chunk_itm_xy_pred_{args.user}.npy', chunk_itm_xy_pred)
	np.save(OUTPUT_PATH+ f'/chunk_final_xy_pred_{args.user}.npy', chunk_final_xy_pred)
	np.save(OUTPUT_PATH+ f'/chunk_gt_xy_{args.user}.npy', chunk_gt_xy)

	####################################################################################

	####################################################################################
	####### ARIMA

	print("ARIMA ...")
	# Read Data
	print("Reading Viewport Data and Object Trajectories...")
	data, frame_nos, max_frame, tot_objects = \
		get_data(data, frame_nos, args.dataset, args.topic, args.user, args.fps, \
		milisec, width, height, view_width, view_height)
	print("Data read\n")

	# preserve X['VIEWPORT_x'] and X['VIEWPORT_y'] in X, and remove the rest keys in X
	for i in range(len(data)):
		for key in list(data[i][0].keys()):
			if key != 'VIEWPORT_x' and key != 'VIEWPORT_y':
				del data[i][0][key]

	print("Build Model...")  # frame_nos: the list frame numbers; tot_objects: the number of objects
	act_tiles, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae, \
		gof, chunk_itm_xy_pred, chunk_final_xy_pred, chunk_gt_xy = \
		build_model(data, frame_nos, max_frame, tot_objects, width, height, \
		nrow_tiles, ncol_tiles, args.fps, pred_nframe)


	# print out the dimensions of each array
	print("gof: " + str(gof.shape))
	# print("chunk_itm_xy_pred: " + str(chunk_itm_xy_pred.shape))
	print("chunk_final_xy_pred: " + str(chunk_final_xy_pred.shape))
	print("chunk_gt_xy: " + str(chunk_gt_xy.shape))

	# Save the output np arrays to PRED_PATH
	OUTPUT_PATH = os.path.join(PRED_PATH, 'ARIMA')
	np.save(OUTPUT_PATH + f'/gof_{args.user}.npy', chunk_frames)
	# np.save(PRED_PATH + f'/chunk_itm_xy_pred_{args.user}.npy', chunk_itm_xy_pred)
	np.save(OUTPUT_PATH+ f'/chunk_final_xy_pred_{args.user}.npy', chunk_final_xy_pred)
	np.save(OUTPUT_PATH+ f'/chunk_gt_xy_{args.user}.npy', chunk_gt_xy)

	####################################################################################

	####################################################################################
	####### OBJ

	print("OBJ ...")
	# Read Data
	print("Reading Viewport Data and Object Trajectories...")
	data, frame_nos, max_frame, tot_objects = \
		get_data(data, frame_nos, args.dataset, args.topic, args.user, args.fps, \
		milisec, width, height, view_width, view_height)
	print("Data read\n")

	# remove X['VIEWPORT_x'] and X['VIEWPORT_y'] in X,
	for i in range(len(data)):
		for key in list(data[i][0].keys()):
			if key == 'VIEWPORT_x' or key == 'VIEWPORT_y':
				del data[i][0][key]

	print("Build Model...")  # frame_nos: the list frame numbers; tot_objects: the number of objects
	act_tiles, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae, \
		gof, chunk_itm_xy_pred, chunk_final_xy_pred, chunk_gt_xy = \
		build_model(data, frame_nos, max_frame, tot_objects, width, height, \
		nrow_tiles, ncol_tiles, args.fps, pred_nframe, wo_vp=True)

	# print out the dimensions of each array
	print("gof: " + str(gof.shape))
	# print("chunk_itm_xy_pred: " + str(chunk_itm_xy_pred.shape))
	print("chunk_final_xy_pred: " + str(chunk_final_xy_pred.shape))
	print("chunk_gt_xy: " + str(chunk_gt_xy.shape))

	# Save the output np arrays to PRED_PATH
	OUTPUT_PATH = os.path.join(PRED_PATH, 'OBJ')
	np.save(OUTPUT_PATH + f'/gof_{args.user}.npy', chunk_frames)
	# np.save(PRED_PATH + f'/chunk_itm_xy_pred_{args.user}.npy', chunk_itm_xy_pred)
	np.save(OUTPUT_PATH+ f'/chunk_final_xy_pred_{args.user}.npy', chunk_final_xy_pred)
	np.save(OUTPUT_PATH+ f'/chunk_gt_xy_{args.user}.npy', chunk_gt_xy)

	####################################################################################

	####################################################################################
	# raise Exception("Stop here")
	####################################################################################

	i = 0
	while True:
		curr_frame=frame_nos[i]
		if curr_frame < 5*args.fps:
			i += 1
		else:
			break

	frame_nos = frame_nos[i:]
	print("Allocate Bitrates...")
	vid_bitrate = alloc_bitrate(pred_tiles, chunk_frames, nrow_tiles, ncol_tiles, pref_bitrate, player_tiles_x, player_tiles_y)
	
	print("Calculate QoE...")
	qoe = calc_qoe(vid_bitrate, act_tiles, chunk_frames, width, height, nrow_tiles, ncol_tiles, player_width, player_height)

	print(qoe)
	#Print averaged results
	print("\n======= RESULTS ============")
	print('Dataset: {}'.format(args.dataset))
	print('Topic: {}'.format(args.topic))
	print('User ID: {}'.format(args.user))
	print('QoE: {}'.format(qoe))

	print('\n\n')

if __name__ == '__main__':
	main()
