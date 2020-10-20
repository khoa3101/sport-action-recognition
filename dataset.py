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
        with open(os.path.join(self.__C.PATH_DATA, '{}.csv'.format(self.__C.MODE)), 'r') as f:
            for path_label in f.read().splitlines():
                path, label = path_label.split(self.__C.DATA_SEPARATOR)
                self.path.append(path)
                self.label.append(label)


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
        [T, W, H] = self.__C.SIZE_DATA
        # Get indices of taken frames
        path_video = self.path[idx]
        num_frames = len(glob.glob(path_video + '/RGB/*'))
        if self.__C.AUGMENTATION:
            frames_idx = random.sample(range(1, num_frames), T)
            frames_idx.sort()
        else:
            interval = (num_frames - T) // 2
            frames_idx = range(interval, interval+100)

        # Get videos
        rgb_video = []
        flow_video = []
        for frame in frames_idx:
            # RGB
            frame_rgb = cv2.imread(os.path.join(path_video, 'RGB/%08d.png' % (frame))).astype(np.float32)
            rgb_video.append(frame / 255.)
            # Flow
            flow_x = cv2.imread(os.path.join(path_video, self.__C.FLOW + '/%08d_x.png' % (frame)), 0)
            flow_y = cv2.imread(os.path.join(path_video, self.__C.FLOW + '/%08d_y.png' % (frame)), 0)
            flow = np.stack([flow_x, flow_y], axis=-1).astype(np.float32)
            flow[flow==128] = 127.5
            flow[flow==127] = 127.5
            frame_flow = cv2.normalize(flow, None, -1, 1, cv2.NORM_MINMAX)
            flow_video.append(frame_flow)

        if not self.mode == 'test':
            if self.__C.AUGMENTATION:
                transform = A.compose([
                    A.Flip(),
                    A.RandomScale(),
                    A.Rotate(limit=10)
                ])
                transformed = transform(images=rgb_video, masks=flow_video)
                rgb_video = trasnformed['images']
                flow_video = transform['masks']

        # Crop ROI
        roi_centers = np.load(os.path.join(path_video, self.__C.FLOW + '/roi_centers.npy'))
        for i, c in enumerate(roi_centers):
            rgb_video[i] = rgb_video[i][c[0]-W//2:c[0]+W//2][c[1]-H//2:c[1]+H//2]
            flow_video[i] = flow_video[i][c[0]-W//2:c[0]+W//2][c[1]-H//2:c[1]+H//2]

        rgb_video = np.transpose(np.array(rgb_video), (3, 0, 1, 2))
        flow_video = np.transpose(np.array(flow_video), (3, 0, 1, 2))
        label = self.label[idx]
        return rgb_video, flow_video, label