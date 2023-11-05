from creme import linear_model
from creme import compose
from creme import compat
from creme import metrics
from creme import model_selection
from creme import optim
from creme import preprocessing
from creme import stream
from sklearn import datasets
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np 
import math
import pickle
import random
import warnings
import time


def pred_frames(data, model, metric_X, metric_Y, frames, prev_frames, 
	tile_manhattan_error, act_tiles, pred_tiles, count, width, height, 
	nrow_tiles, ncol_tiles):
	"""
	Args:
		metrix_X: MAE metric for x coordinate
		metrix_Y: MAE metric for y coordinate
		frame: list of frame numbers within the current pred window
		prev_frames: list of previous frame numbers
	
	Returns:
		tile_manhattan_error: Manhattan error for tiles
		count: number of frames
		act_tiles
		pred_tiles
	"""

	x_pred, y_pred = 0,0
	series = []
	shift_x = False
	intermediate_xy_pred = []
	final_xy_pred = []
	gt_xy = []
	

	# 
	for k in range(len(prev_frames)):
		# get the current viewport xy
		[inp_k, x_act, y_act] = data[prev_frames[k]]
		a = [-1 for i in range(2)]  # a = [x,y]
		for key, value in inp_k.items():
			if key == 'VIEWPORT_x':
				a[0] = value
			if key == 'VIEWPORT_y':
				a[1] = value
		# if the first frame, just append to series
		if k==0:
			series.append(a)
			continue

		# determine whether to shift x or not
		[prev_x,prev_y] = series[-1]
		
		if prev_x < a[0]:
			new_x_dif = a[0]-width
			if a[0] - prev_x > prev_x - new_x_dif:
				a[0] = new_x_dif
		else:
			new_x_dif = a[0]+width
			if prev_x - a[0] > new_x_dif - prev_x:
				a[0] = new_x_dif

		if a[0]<0:
			shift_x = True

		series.append(a)

	if shift_x == True:
		for i in range(len(series)):
			series[i][0] = series[i][0]+width

	series = np.array(series, dtype=np.float64)
	series_copy = series

	# transform into log values
	result = len(series[:, 0]) > 0 and \
		all(elem == series[0][0] for elem in series[:, 0])
	if(result):
		series[:, 0] = [elem + random.random()/10 for elem in series[:, 0]]
	for i in range(len(series[:, 0])):
		if series[i, 0] == 0:
			series[i, 0] += random.random()
	# each element is the difference between the current and previous y
	series_log = np.log(series[:, 0])
	series_x = np.diff(series_log, 1)  

	result = len(series[:, 1]) > 0 and \
		all(elem == series[0][1] for elem in series[:, 1])
	if(result):
		series[:, 1] = [elem + random.random()/10 for elem in series[:, 1]]
	for i in range(len(series[:, 1])):
		if series[i, 1] == 0:
			series[i, 1] += random.random()
	
	series_log = np.log(series[:, 1])
	series_y = np.diff(series_log, 1)
	

	series_new = []
	for i in range(len(series_x)):
		series_new.append([series_x[i], series_y[i]])

	# train and predict using ARIMA
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		
		model_x = SARIMAX(series_x, order=(2,0,1))
		model_fit_x = model_x.fit(maxiter=1000, disp=0, method='nm')
		model_pred_x = model_fit_x.forecast(len(frames))

		model_y = SARIMAX(series_y, order=(3,0,0))
		model_fit_y = model_y.fit(maxiter=1000, disp=0, method='nm')
		model_pred_y = model_fit_y.forecast(len(frames))

	# The prediction is the log difference: log(x_t) - log(x_t-1) = log(x_t/x_t-1)
	# So, x_t = exp(log(x_t) - log(x_t-1)) * x_t-1
	x_pred_list, y_pred_list = [], []
	for k in range(len(model_pred_x)):
		if k == 0:
			x_pred_list.append(np.exp(model_pred_x[k]) * series_copy[-1][0])
			y_pred_list.append(np.exp(model_pred_y[k]) * series_copy[-1][1])
		else:
			x_pred_list.append(np.exp(model_pred_x[k]) * x_pred_list[k-1])
			y_pred_list.append(np.exp(model_pred_y[k]) * y_pred_list[k-1])

	# package x_pred_list and y_pred_list into intermediate_xy_pred
	for k in range(len(frames)):
		intermediate_xy_pred.append([x_pred_list[k], y_pred_list[k]])


	# LR prediction
	# predict_one(inp_k, update, x_act, y_act)
	for k in range(len(frames)):
		[inp_k, x_act, y_act] = data[frames[k]]

		gt_xy.append([x_act, y_act])

		# update the model at the first frame	
		if(k == 0):
			x_pred, y_pred = model.predict_one(inp_k, True, x_act, y_act)
		else:
			if(shift_x==True):
				x_pred_list[k] = x_pred_list[k] - width
			inp_k['VIEWPORT_x'] = x_pred_list[k-1]
			inp_k['VIEWPORT_y'] = y_pred_list[k-1]
			x_pred, y_pred = model.predict_one(inp_k, False, None, None)	

		shift = 0
		if(x_act > x_pred):
			if(abs(x_act - x_pred) > abs(x_act - (x_pred+width))):
				x_pred = x_pred+width
				shift = 1
		else:
			if(abs(x_act - x_pred) > abs(x_act - (x_pred-width))):
				x_pred = x_pred-width
				shift = 2

		metric_X = metric_X.update(x_act, x_pred)
		metric_Y = metric_Y.update(y_act, y_pred)

		final_xy_pred.append([x_pred, y_pred])

		# get the tile number that the actual and predicted viewport belongs to
		if(shift == 0):
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int(x_pred * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)
		elif(shift == 1):
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int((x_pred - width) * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)
		else:
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int((x_pred + width) * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)

		actual_tile_row = actual_tile_row - nrow_tiles if(actual_tile_row >= nrow_tiles) else actual_tile_row
		actual_tile_col = actual_tile_col - ncol_tiles if(actual_tile_col >= ncol_tiles) else actual_tile_col
		actual_tile_row = actual_tile_row + nrow_tiles if actual_tile_row < 0 else actual_tile_row
		actual_tile_col = actual_tile_col + ncol_tiles if actual_tile_col < 0 else actual_tile_col

		######################################################
		# print("x: "+str(x_act))
		# print("x_pred: "+str(x_pred))
		# print("y: "+str(y_act))	
		# print("y_pred: "+str(y_pred))
		# print("("+str(actual_tile_row)+","+str(actual_tile_col)+"),("+str(pred_tile_row)+","+str(pred_tile_col)+")")
		# # ######################################################
		
		act_tiles.append((actual_tile_row, actual_tile_col))
		pred_tiles.append((pred_tile_row, pred_tile_col))

		tile_col_dif = ncol_tiles

		if actual_tile_col < pred_tile_col:
			tile_col_dif = \
			min(pred_tile_col - actual_tile_col, actual_tile_col + \
			ncol_tiles - pred_tile_col)
		else:
			tile_col_dif = min(actual_tile_col - pred_tile_col, ncol_tiles + \
			pred_tile_col - actual_tile_col)

		tile_manhattan_error += abs(actual_tile_row - pred_tile_row) + abs(tile_col_dif)
		count = count+1

	return metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles, \
		intermediate_xy_pred, final_xy_pred, gt_xy



def build_model(data, frame_nos, max_frame, tot_objects, width, height, nrow_tiles, ncol_tiles, fps, pred_nframe):
	"""
	Args:
		frame_nos: list of frame numbers
		max_frame: maximum frame number
		tot_objects: total number of objects in the video
		pred_nframe: number of frames to be predicted
	"""
	model = linear_model.PARegressor(C=0.01, mode=2, eps=0.001, data=data, learning_rate=0.005, rho=0.99)
	metric_X = metrics.MAE()
	metric_Y = metrics.MAE()
	manhattan_error = []
	x_mae = []
	y_mae = []
	count = 0

	i = 0
	tile_manhattan_error = 0
	act_tiles, pred_tiles = [],[]
	chunk_frames = []
	chunk_itm_xy_pred = []
	chunk_final_xy_pred = []
	chunk_gt_xy = []
	gof = []

	# Warm-up training of first 5 seconds
	prev_frames = {0}
	while True:
		curr_frame = frame_nos[i]
		prev_frames.add(i)
		# fit the model
		if curr_frame < 5*fps:
			i=i+1
			[inp_i,x,y]=data[curr_frame]
			model = model.fit_one(inp_i,x,y)
		else:
			break
	prev_frames = sorted(prev_frames)
	cnt = 0

	# Start predicting frames while updating model
	
	while True:
		curr_frame = frame_nos[i]
		print("The head frame of the current prediction: "+str(curr_frame))
		# nframe: number of frames to be predicted
		nframe = min(pred_nframe, max_frame - frame_nos[i])  
		if(nframe < 1):  # if not remaining frames
			break
		# frames: frame numbers within the current prediction window
		frames = {i}
		for k in range(i+1, len(frame_nos)):
			if(frame_nos[k] < curr_frame + nframe):
				frames.add(k)
			else:
				i=k
				break
		if(i!=k):
			i=k
		if(i==(len(frame_nos)-1)):
			break
			
		# chunk_frames = [[fid of chunk1], [fid of chunk2], ...
		frames = sorted(frames)
		chunk_frames.append(frames)
		gof.append([frame_nos[f] for f in frames])
		# get predictions
		metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles, \
			intermediate_xy_pred, final_xy_pred, gt_xy = \
			pred_frames(data, model, metric_X, metric_Y, \
			frames, prev_frames, tile_manhattan_error, act_tiles, pred_tiles, count, width, height, nrow_tiles, ncol_tiles)
		# Batch update
		model = model.fit_n(frames)

		# Store results
		chunk_itm_xy_pred.append(intermediate_xy_pred)
		chunk_final_xy_pred.append(final_xy_pred)
		chunk_gt_xy.append(gt_xy)

		prev_frames = prev_frames+frames
		manhattan_error.append(tile_manhattan_error*1.0 / count)
		x_mae.append(metric_X.get())
		y_mae.append(metric_Y.get())

		print("Manhattan Tile Error: "+str(tile_manhattan_error*1.0 / count))
		print(metric_X, metric_Y)
		print("\n")

		# till 60 seconds
		cnt = cnt + 1
		if cnt == 60:
			break

	# transform chunk_frames, chunk_itm_xy_pred, chunk_final_xy_pred, chunk_gt_xy into np arrays
	gof = np.array(gof)
	chunk_itm_xy_pred = np.array(chunk_itm_xy_pred)
	chunk_final_xy_pred = np.array(chunk_final_xy_pred)
	chunk_gt_xy = np.array(chunk_gt_xy)

	
	return act_tiles, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae, gof, chunk_itm_xy_pred, chunk_final_xy_pred, chunk_gt_xy

