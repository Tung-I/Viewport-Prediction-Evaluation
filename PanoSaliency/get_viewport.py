import os
import numpy as np
import pickle
import header
import argparse

import head_orientation_lib
import saldat_head_orientation


if __name__ == "__main__":

	#specify dataset & video name to extract
	parser = argparse.ArgumentParser(description='Run Viewport Extraction Algorithm')

	parser.add_argument('-D', '--dataset', type=int, required=True, help='Dataset ID (1 or 2)')
	parser.add_argument('-T', '--topic', required=True, help='Topic in the particular Dataset (video name)')
	parser.add_argument('--fps', type=int, required=True, help='fps of the video')
	
	args = parser.parse_args()

	dataset = args.dataset
	topic = args.topic 	#dataset 1: paris, roller, venise,diving,timelapse, 
				   		#dataset 2: '0', '1', '2', '3', '4', '5', '6', '7', '8'
	fps = args.fps

	if args.dataset != 1 and args.dataset != 2:
		print("Incorrect value of the Dataset ID provided!!...")
		print("======= EXIT ===========")
		exit()

	PATH = '../Viewport/ds{}/'.format(dataset)
	if not os.path.exists(PATH):
		os.makedirs(PATH)
	
	#initialize head_orentiation
	print ("Extract Viewport for ds={}, topic={}".format(dataset, topic))
	dirpath1 = header.dirpath1	#u'./data/head-orientation/dataset1'
	dirpath2 = header.dirpath2	#u'./data/head-orientation/dataset2/Experiment_1'
	ext1 = header.ext1
	ext2 = header.ext2

	headoren = saldat_head_orientation.HeadOrientation(dirpath1, dirpath2, ext1, ext2)
	dirpath, filename_list, f_parse, f_extract_direction = headoren.load_filename_list(dataset, topic)
	##########################################################################
	# dirpath: viewport log directory
	# filename_list: all the user's csv files
	# f_parse: the parsing function for the csv file
	# f_extract_direction: the function to extract the head direction from the csv file


	##########################################################################
	
	series_ds = headoren.load_series_ds(filename_list, f_parse, dataset)
	# series: [n_user][n_pb_timing][n_dim_of_filtered_data]

	vector_ds = headoren.headpos_to_headvec(series_ds, f_extract_direction, dataset)
	# vector_ds[n_user][n_pb_timing] = [pb_time, v, 0, 0]

	_, vlength, _, _ = head_orientation_lib.topic_info_dict[topic]  # [video_name, video_length, width, height]

	if(dataset == 1):
		user = 1
		for vector in vector_ds:
			viewport = []
			print("Usernum={}".format(user))
			max_frame = vector[-1][0]
			for f in np.arange(0, max_frame):
				dt = 1.0/fps
				series_t = []
				series_v = []
				print(f, end='\r')
				for item in vector:
					if item[0] == f:
						series_t.append(item[0])
						series_v.append(item[1])

				if(len(series_t) != 0):
					mean_series_t = np.mean(np.array(series_t)) * dt
					mean_series_v = np.mean(np.array(series_v), axis=0)

					# transform a vector to a point on the sphere
					theta, phi = head_orientation_lib.vector_to_ang(mean_series_v)
					# transform a point on the sphere to a point on the viewport
					x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)

					viewport.append([mean_series_t, (x,y)])

			pickle.dump(viewport, open(PATH + 'viewport_ds{}_topic{}_user{}'.format(dataset, topic, user), 'wb'))
			user += 1
	else:
		user = 1
		for vector in vector_ds:  # for each user
			# vector: [pb_time][pb_time, v, 0, 0]
			viewport = []
			print("Usernum={}".format(user))
			dt = fps
			for item in vector:  # [pb_time, v, 0, 0]
				print(item[0], end='\r')  
				mean_series_t = item[0]  # pb_time
				mean_series_v = np.mean(np.array([item[1]]), axis=0) # [vx, vy, vz]

				theta, phi = head_orientation_lib.vector_to_ang(mean_series_v)
				x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)

				viewport.append([mean_series_t, (x,y)])

			pickle.dump(viewport, open(PATH + 'viewport_ds{}_topic{}_user{}'.format(dataset, topic, user), 'wb'))
			user += 1
