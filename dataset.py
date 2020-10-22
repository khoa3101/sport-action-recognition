import torch
import glob, json, cv2, os, random
import numpy as np
import albumentations as A

class TableTennis(torch.utils.data.Dataset):
    def __init__(self, mode, __C):
        self.mode = mode
        self.__C = __C

        with open(__C.LABEL_DICT, 'r') as f:
            self.label_dict = json.load(f)
        self._construct()


    def _construct(self):
        self.path = []
        self.label = []
        with open(os.path.join(self.__C.PATH_DATA, '{}.csv'.format(self.mode)), 'r') as f:
            for path_label in f.read().splitlines():
                path, label = path_label.split(self.__C.DATA_SEPARATOR)
                self.path.append(path)
                self.label.append(self.label_dict[label])


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        rgb, flow, label = self.get_annotation_data(idx)
        sample = {
            'rgb': torch.from_numpy(rgb), 
            'flow': torch.from_numpy(flow), 
            'label': label
        }
        return sample


    def get_annotation_data(self, idx):
        [T, H, W] = self.__C.SIZE_DATA
        # Get indices of taken frames
        path_video = self.path[idx]
        num_frames = len(glob.glob(path_video + '/RGB/*'))
        
        # if self.__C.AUGMENTATION:
        #     frames_idx = random.sample(range(1, num_frames), T)
        #     frames_idx.sort()
        # else:
        interval = (num_frames - T) // 2
        frames_idx = range(interval, interval+T)

        # Augmentation
        seed = random.randint(0, 999999999)
        if self.mode == 'train':
            if self.__C.AUGMENTATION:
                transform = A.Compose([
                    A.Flip(),
                    #A.RandomScale(),
                    A.Rotate(limit=10)
                ])

        # Get videos
        rgb_video = []
        flow_video = []
        roi_centers = np.load(os.path.join(path_video, self.__C.FLOW + '/roi_centers.npy'))
        for frame in frames_idx:
            # RGB
            frame_rgb = cv2.imread(os.path.join(path_video, 'RGB/%08d.png' % (frame))).astype(np.float32)
            frame_rgb_norm = frame_rgb / 255.
            frame_rgb_norm = np.pad(
                frame_rgb_norm, 
                ((H//2, H//2), (W//2, W//2), (0, 0))
            )

            # Flow
            flow_x = cv2.imread(os.path.join(path_video, self.__C.FLOW + '/%08d_x.png' % (frame)), 0)
            flow_y = cv2.imread(os.path.join(path_video, self.__C.FLOW + '/%08d_y.png' % (frame)), 0)
            flow = np.stack([flow_x, flow_y], axis=-1).astype(np.float32)
            flow[flow==128] = 127.5
            flow[flow==127] = 127.5
            frame_flow = cv2.normalize(flow, None, -1, 1, cv2.NORM_MINMAX)
            frame_low = np.pad(
                frame_flow, 
                ((H//2, H//2), (W//2, W//2), (0, 0))
            )

            # Augmentation
            if self.mode == 'train':
                if self.__C.AUGMENTATION:
                    random.seed(seed)
                    transformed = transform(image=frame_rgb_norm, mask=frame_flow)
                    frame_rgb_norm = transformed['image']
                    frame_flow = transformed['mask']

            # Crop ROI
            x_start = roi_centers[frame-1, 0]
            y_start = roi_centers[frame-1, 1]
            x_end = roi_centers[frame-1, 0] + H
            y_end = roi_centers[frame-1, 1] + W
            frame_rgb_norm = frame_rgb_norm[x_start:x_end, y_start:y_end, :]
            frame_flow = frame_flow[x_start:x_end, y_start:y_end, :]

            # Append
            rgb_video.append(frame_rgb_norm)
            flow_video.append(frame_flow)

        rgb_video = np.transpose(np.array(rgb_video), (3, 0, 1, 2))
        flow_video = np.transpose(np.array(flow_video), (3, 0, 1, 2))
        label = self.label[idx]
        return rgb_video, flow_video, label
