import cv2
import numpy as np
import os
from tqdm import tqdm
import json
import argparse

VIDEO_DIR = '/home/tungi/datasets/vr_dataset/videos'
video_file_name =  '1-7-Cooking Battle.mp4'
PRED_DIR = '/home/tungi/datasets/vr_dataset/predictions'
OUTPUT_DIR = '/home/tungi/datasets/vr_dataset/visualize'


def draw_viewport(frame, center, color):
    
    # center = (int(center[0] * width / display_width), int(center[1] * height / display_height))
    # top_left = (int(center[0] - display_width/2), int(center[1] - display_height/2))
    # bottom_right = (int(center[0] + display_width/2), int(center[1] + display_height/2))
    # cv2.rectangle(frame, top_left, bottom_right, color, 4)

    # center = (int(center[0] * width / view_width), int(center[1] * height / view_height))
    center = (int(center[0]), int(center[1]))
    cv2.circle(frame, center, 24, color, -1)  # -1 means filled

def main():

    parser = argparse.ArgumentParser(description='Run PARIMA algorithm and calculate QoE of a video for a single user')
    parser.add_argument('-T', '--topic', required=True, help='Topic in the particular Dataset (video name)')
    parser.add_argument('-U', '--user', type=int, default=0, help='User ID on which the algorithm will be run [default: 0]')
    args = parser.parse_args()

    global PRED_DIR
    global OUTPUT_DIR
    
    PRED_DIR = os.path.join(PRED_DIR, f'{args.topic}_{args.user}')
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'{args.topic}_{args.user}')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


    output_video_path = os.path.join(OUTPUT_DIR, f'output.mp4')

    file = open('./meta.json', )
    jsonRead = json.load(file)

    width = jsonRead["dataset"][1]["width"]
    height = jsonRead["dataset"][1]["height"]
    view_width = jsonRead["dataset"][1]["view_width"]
    view_height = jsonRead["dataset"][1]["view_height"]

    # Load viewport data
    viewport_pred_parima = np.load(os.path.join(PRED_DIR, f'parima_chunk_final_xy_pred.npy'))
    viewport_pred_arima = np.load(os.path.join(PRED_DIR, f'arima_chunk_final_xy_pred.npy'))
    # viewport_pred_obj = np.load(os.path.join(PRED_DIR, f'obj_chunk_final_xy_pred.npy'))
    viewport_gt = np.load(os.path.join(PRED_DIR, f'chunk_gt_xy.npy'))
    gof = np.load(os.path.join(PRED_DIR, f'gof.npy'))

    # Flatten the frame groups and viewport arrays
    frame_ids = gof.flatten()
    viewport_pred_parima_flat = viewport_pred_parima.reshape(-1, viewport_pred_parima.shape[-1])
    viewport_pred_arima_flat = viewport_pred_arima.reshape(-1, viewport_pred_arima.shape[-1])
    # viewport_pred_obj_flat = viewport_pred_obj.reshape(-1, viewport_pred_obj.shape[-1])
    viewport_gt_flat = viewport_gt.reshape(-1, viewport_gt.shape[-1])

    # Build a mapping of frame ID to viewport data
    vp_parima = {frame_id: viewport for frame_id, viewport in zip(frame_ids, viewport_pred_parima_flat)}
    vp_arima = {frame_id: viewport for frame_id, viewport in zip(frame_ids, viewport_pred_arima_flat)}
    # vp_obj = {frame_id: viewport for frame_id, viewport in zip(frame_ids, viewport_pred_obj_flat)}
    viewport_gt_map = {frame_id: viewport for frame_id, viewport in zip(frame_ids, viewport_gt_flat)}

    # Fill in the missing viewport data
    max_frame_id = max(frame_ids)
    vp_parima_filled = [vp_parima.get(i, None) for i in range(max_frame_id + 1)]
    vp_arima_filled = [vp_arima.get(i, None) for i in range(max_frame_id + 1)]
    # vp_obj_filled = [vp_obj.get(i, None) for i in range(max_frame_id + 1)]
    viewport_gt_filled = [viewport_gt_map.get(i, None) for i in range(max_frame_id + 1)]


    # Forward fill the missing data
    for i in range(1, len(viewport_gt_filled)):
        vp_parima_filled[i] = vp_parima_filled[i] if vp_parima_filled[i] is not None else vp_parima_filled[i-1]
        vp_arima_filled[i] = vp_arima_filled[i] if vp_arima_filled[i] is not None else vp_arima_filled[i-1]
        # vp_obj_filled[i] = vp_obj_filled[i] if vp_obj_filled[i] is not None else vp_obj_filled[i-1]
        viewport_gt_filled[i] = viewport_gt_filled[i] if viewport_gt_filled[i] is not None else viewport_gt_filled[i-1]



    # Open the video file
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_file_name))
    # Get the properties of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Prepare the output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' if mp4v doesn't work
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    # Skip to the start frame
    start_frame = min(frame_ids)
    # start_frame = START_FRAME
    for i in range(start_frame):
        ret, frame = cap.read()

    # Process the video and draw the viewports
    for frame_index in range(start_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        print(frame_index)
        if frame_index > max_frame_id:
            break

        ret, frame = cap.read()

        if not ret:
            break
        
        draw_viewport(frame, vp_parima_filled[frame_index], (0, 0, 255))  
        draw_viewport(frame, vp_arima_filled[frame_index], (26, 160, 252))  
        # draw_viewport(frame, vp_obj_filled[frame_index], (0, 255, 0)) 
        draw_viewport(frame, viewport_gt_filled[frame_index], (0, 0, 0))  
        
        # Write the frame to the output video file
        out.write(frame)

    # Release everything
    cap.release()
    out.release()

    print(f"Video processing complete. The output video is stored as {output_video_path}")


if __name__ == '__main__':
	main()
