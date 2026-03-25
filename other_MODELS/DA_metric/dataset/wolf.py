import os,cv2
import numpy as np
import torch
#import torchvision.io as tvio
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop, CustomAugmentation
#from torchvision.io import read_image
from glob import glob



class WolfDataset(Dataset):
    def __init__(self, data_dir, npy_dir, mode=None, transform=None, size=(518, 518)):
        """
        Args:
            data_dir (str): Directory containing folders with extracted frames.
            transform (callable, optional): Optional transform to apply to the frames.
        """
        self.data_dir = data_dir
        self.npy_dir = npy_dir
        #self.frames_dir = os.path.join(data_dir, 'frames')
        #self.masks_dir = os.path.join(data_dir, 'masks')
        # if mode.upper()=='TRAIN':
        #     with open(os.path.join(self.data_dir, 'train.txt'), 'r') as f:
        #         self.video_folders = sorted([os.path.join(self.data_dir, line.strip()) for line in f if line.strip()])
        #     with open(os.path.join(self.data_dir, 'train.txt'), 'r') as f:
        #         self.npy_folders = sorted([os.path.join(npy_dir, line.strip()) for line in f if line.strip()])

        # else:
        #     with open(os.path.join(self.data_dir, 'test.txt'), 'r') as f:
        #         self.video_folders = [os.path.join(self.data_dir, line.strip()) for line in f if line.strip()]
        #     with open(os.path.join(self.data_dir, 'test.txt'), 'r') as f:
        #         self.npy_folders = sorted([os.path.join(npy_dir, line.strip()) for line in f if line.strip()])

        ignore_dirs = ['06090172','06090148','06090246','06090248','06090314','06090326','06100339','06100341','06100355','06100357','06100447']
        
        self.video_folders = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, f)) and f not in ignore_dirs])
        self.npy_folders = sorted([os.path.join(npy_dir, f) for f in os.listdir(npy_dir) 
            if os.path.isdir(os.path.join(npy_dir, f))and f not in ignore_dirs])

        self.dataset_mean = np.load(os.path.join(npy_dir,'robust_mean_stage_17.npy'))

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        self.augmentations = CustomAugmentation(size=size)
        
        # Collect a list of all frame file paths
        self.frames_list, self.npy_list = self._get_all_frames()
        assert len(self.frames_list)==len(self.npy_list)

    def _get_all_frames(self):
        """
        Collects all frame file paths from the directory structure.
        
        Returns:
            List of file paths to the frames.
        """
        frames_list = []
        npy_list = []
        for img_folder, npy_folder in zip(self.video_folders, self.npy_folders):
            frame_files = sorted(glob(os.path.join(img_folder, '*.png')))
            npy_files = sorted(glob(os.path.join(npy_folder,'*.npy')))

            frames_list.extend(frame_files)
            npy_list.extend(npy_files)
        return frames_list, npy_list

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        
        frame_path = self.frames_list[idx]
        npy_path = self.npy_list[idx]

        mask_path = frame_path.replace('FRAMES', 'MASKS')

        frame = cv2.imread(frame_path)
        #npy_file = np.load(npy_path)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            sample = self.transform({'image':frame,'mask': mask,'depth':self.dataset_mean})
            frame = sample['image']
            mask = sample['mask']
            gt_mean = sample['depth']

        frame = torch.from_numpy(frame)
        mask = torch.from_numpy(mask)/255.0
        mask = mask.unsqueeze(0)
        gt_mean = torch.from_numpy(self.dataset_mean).unsqueeze(0)
        
        if self.transform:
            frame, mask, gt_mean = self.augmentations(frame, mask, gt_mean)
            
        #npy_file = torch.from_numpy(npy_file)#.permute(2, 0, 1).float() # Convert to (C, H, W) and float

        return frame, mask.squeeze(), gt_mean.squeeze()


# # Example usage
# from torchvision import transforms

# # Define any image transforms you want to apply to the frames (e.g., resizing, normalization)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize frames to 224x224
#     transforms.ToTensor(),          # Convert frames to PyTorch tensor
# ])

# video_frame_dataset = WolfDataset(video_dir="path_to_videos", transform=transform, frame_rate=30)

# # Define a DataLoader to load the frames in batches
# video_frame_loader = torch.utils.data.DataLoader(video_frame_dataset, batch_size=16, shuffle=True)

# # Example loop to load video frames
# for frames in video_frame_loader:
#     print(frames.shape)  # Each batch will contain [batch_size, channels, height, width]
