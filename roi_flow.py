import numpy as np
import os, glob, argparse, cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='video', type=str, required=True, help='Path to video folder')
    parser.add_argument('-m', dest='method', type=str, required=True, help='Optical flow method')
    args = parser.parse_args()
    return args

def func(omega, u, V):
    return max(min(u, V - omega//2), omega//2)


def center(flow, alpha=0.6):
    norm = np.sum(np.abs(flow), axis=-1)
    #argmax = np.argmax(norm)
    #X_max = np.array([argmax % norm.shape[0], argmax // norm.shape[1]])
    delta = norm > 1e-5
    row, column = np.indices((180, 320))
    x = np.sum(row*delta) // np.sum(delta)
    y = np.sum(column*delta) // np.sum(delta)
    #x_roi = alpha*func(320, X_max[0], 120) + (1-alpha)*func(180, X_g[0], 120)
    #y_roi = alpha*func(320, X_max[1], 120) + (1-alpha)*func(180, X_g[1], 120)
    return np.array([x, y])#np.array([x_roi, y_roi])


if __name__ == '__main__':
    args = parse_args()

    # Load computed optical flows
    flow_method_path = os.path.join(args.video, args.method)
    num_videos = len(glob.glob(flow_method_path + '/*')) // 2
    flows = []
    for path in range(1, num_videos+1):
        flow_x = cv2.imread(os.path.join(flow_method_path, '%08d_x.png' % (path)), 0)
        flow_y = cv2.imread(os.path.join(flow_method_path, '%08d_y.png' % (path)), 0)
        flow = np.stack([flow_x, flow_y], axis=-1).astype(np.float32)
        flow[flow==128] = 127.5
        flow[flow==127] = 127.5
        flow = cv2.normalize(flow, None, -1, 1, cv2.NORM_MINMAX)
        flows.append(flow)
    
    # Compute ROI centers
    roi_centers = []
    for flow in flows:
        roi_center = center(flow)
        roi_centers.append(roi_center)
    np.save(os.path.join(flow_method_path, 'roi_centers.npy'), roi_centers)