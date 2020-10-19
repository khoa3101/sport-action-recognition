import numpy as np 
import os, cv2, glob, argparse
from utils import make_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='inp', type=str, required=True, help='Path to input images folder')
    parser.add_argument('-o', dest='out', type=str, required=True, help='Path to ouptut folder')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    make_path(args.out)

    # Load the images
    image_list = glob.glob(args.inp + '/*')
    image_list.sort()
    images = []
    masks = []
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for path in image_list:
        image = cv2.imread(path)
        mask = fgbg.apply(image)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        masks.append(np.expand_dims(mask/255, axis=-1))

    # Compute optical flow filterd by background subtractor
    flow_masks = []
    for frame in range(1, len(images)):
        flow = cv2.calcOpticalFlowFarneback(
            images[frame-1], 
            images[frame], 
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_mask = flow * masks[frame]
        flow_masks.append(flow_mask)

    # Compute mean and standard deviation
    flow_masks = np.array(flow_masks)
    mean = np.mean(flow_masks, axis=0, keepdims=True)
    std = np.std(flow_masks, axis=0, keepdims=True)
    np.seterr(divide='ignore', invalid='ignore')
    flow_masks /= mean + 3*std
    flow_masks[np.isnan(flow_masks)] = 0
    # Normalize flow
    flow_masks[flow_masks<-1] = -1
    flow_masks[flow_masks>1] = 1
    # Save flow
    for idx, flow in enumerate(flow_masks):
        flow_norm = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(args.out, '%08d_x.png' % (idx+1)), flow_norm[:, :, 0])
        cv2.imwrite(os.path.join(args.out, '%08d_y.png' % (idx+1)), flow_norm[:, :, 1])
        
