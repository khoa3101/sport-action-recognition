import os, cv2, sys, time, datetime, threading, glob, argparse
import numpy as np
from tqdm import tqdm
from pool import ActivePool
from utils import make_path, progress_bar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='inp', type=str, help='Path to input folder')
    parser.add_argument('-o', dest='out', type=str, help='Path to output folder')
    args = parser.parse_args()
    return args


############################################################
##################### Build the data #######################
############################################################
def build_data(video_list, save_path, width_OF=320, workers=15, flow_method='DeepFlow'):
    make_path(save_path)

    # Extract Frames
    extract_frames(video_list, save_path, width_OF)

    # Compute DeepFlow
    compute_DeepFlow(video_list, save_path, workers)

    # # Compute ROI
    compute_ROI(video_list, save_path, workers, flow_method=flow_method)


##################### RGB #######################
def extract_frames(video_list, save_path, width_OF):
    # Chrono
    start_time = time.time()

    print('Extracting frame')
    for idx, video_path in enumerate(tqdm(video_list)):
        
        video_name = os.path.basename(video_path)

        path_data_video = os.path.join(save_path, video_name.split('.')[0])
        make_path(path_data_video)
        path_RGB = os.path.join(path_data_video, 'RGB')
        make_path(path_RGB)

        # Load Video
        cap = cv2.VideoCapture(video_path)
        length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        
        # Check if video uploaded
        if not cap.isOpened():
            sys.exit("Unable to open the video, check the path.\n")

        while frame_number < length_video:
            # Load video
            ret, rgb = cap.read()

            # Check if load Properly
            if ret == 1:
                # Resizing and Save
                rgb = cv2.resize(rgb, (width_OF, rgb.shape[0] * width_OF // rgb.shape[1]))
                cv2.imwrite(os.path.join(path_RGB, '%08d.png' % frame_number), rgb)
                frame_number += 1
        cap.release()

    print('Frame extraction completed in %s' % (datetime.timedelta(seconds=int(time.time() - start_time))))

##################### Deep Flow #######################
def compute_DeepFlow(video_list, save_path, workers):
    start_time = time.time()
    DeepFlow_pool = ActivePool()

    print('\nComputing DeepFlow')
    for idx, video_path in enumerate(tqdm(video_list)):
        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)

        # Split the calculation in severals process
        while threading.activeCount() > workers:
            time.sleep(0.1)

        if threading.activeCount() <= workers:
            job = threading.Thread(
                target=compute_DeepFlow_video, 
                name=idx, 
                args=(
                    DeepFlow_pool, 
                    os.path.join(path_data_video, 'RGB'), 
                    os.path.join(path_data_video, 'DeepFlow')
                )
            )
            job.daemon=True
            job.start()

    while threading.activeCount()>1:
        time.sleep(0.1)

    print('DeepFlow computation done in %s' % (datetime.timedelta(seconds=int(time.time() - start_time))))


def compute_DeepFlow_video(pool, path_RGB, path_Flow):
    name = threading.current_thread().name
    pool.makeActive(name)
    os.system('python deep_flow.py -i %s -o %s' % (path_RGB, path_Flow))
    pool.makeInactive(name)


##################### ROI #######################
def compute_ROI(video_list, save_path, workers, flow_method='DeepFlow'):
    start_time = time.time()
    ROI_pool = ActivePool()

    print('\nComputing ROI for %s' % (flow_method))
    for idx, video_path in enumerate(tqdm(video_list)):
        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)

        # Split the calculation in severals process
        while threading.activeCount() > workers:
            time.sleep(0.1)

        if threading.activeCount() <= workers:
            job = threading.Thread(
                target=compute_roi_video, 
                name=idx, 
                args=(ROI_pool, path_data_video, flow_method)
            )
            job.daemon=True
            job.start()

    while threading.activeCount()>1:
        time.sleep(0.1)

    #join_values_flow(video_list, 'values_flow_%s' % flow_method)
    print('ROI computation for %s completed in %s' % (flow_method, datetime.timedelta(seconds=int(time.time() - start_time))))


def compute_roi_video(pool, path_data_video, flow_method):
    name = threading.current_thread().name
    pool.makeActive(name)
    os.system('python roi_flow.py -v %s -m %s' % (path_data_video, flow_method))
    pool.makeInactive(name)


def join_values_flow(video_list, name_values, save_path):
    values_flow = []
    for video_path in video_list:
        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)
        values_flow_video = np.load(os.path.join(path_data_video, '%s.npy' % name_values))
        values_flow.extend(values_flow_video)
    np.save(os.path.join(save_path, name_values), values_flow)


if __name__ == "__main__":
    args = parse_args()

    video_list = glob.glob(args.inp)
    start_time = time.time()

    build_data(video_list, args.out)

    print('\nDone in %s' % (datetime.timedelta(seconds=int(time.time() - start_time))))