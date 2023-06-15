import os
import torch 
import torchvision.transforms as transforms
import numpy as np
from data_processing.base_dataset import BaseDataset
import fnmatch
import cv2

class NRGGBRecurrentDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        """
        Creates an iterator over dataset.
        :param root: path to dataset root
        """
        self.dirs = [os.path.join(os.path.join(opt.dataroot), a) for a in os.listdir(os.path.join(opt.dataroot))]
        self.test_on_txt = opt.test_on_txt
        self.data = []
        for i, dir_name in enumerate(self.dirs):
            files = sorted([os.path.join(dir_name, a) for a in os.listdir(dir_name)])
            print(dir_name, len(files))
            self.data.append(files)
        self.dataset_size = len(self.data)
        print(self.dataset_size)

        self.num_bins = opt.num_bins
        self.normalization = opt.event_norm

    def __len__(self):        
        return self.dataset_size
    
    def process(self, gt_path, ldr_path, ev_path, sig_s = 'RAN', sig_c = 'RAN'):
        if gt_path.endswith('.hdr'):
            hdr = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
        else :
            hdr = cv2.imread(gt_path) / 255

        if ev_path.endswith('.npy'):
            ev_rep = np.load(ev_path)
            ev_rep = torch.from_numpy(ev_rep)
        elif ev_path.endswith('.txt'):
            events = np.loadtxt(ev_path).reshape(-1, 4)
            ev_rep = None
            height, width = hdr.shape[:2]
            if events.shape[0] == 0:
                ev_rep = torch.zeros((self.num_bins, height, width))
            else :
                ev_rep = self.__events_to_voxel_grid_pytorch(events, self.num_bins, width, height, self.normalization)
        
        ldr = cv2.imread(ldr_path).transpose(2, 0, 1) / 255
        ldr_tensor = torch.from_numpy(ldr)        
        hdr_tensor = torch.from_numpy(hdr.transpose(2, 0, 1))
        return ev_rep, hdr_tensor, ldr_tensor, sig_s, sig_c

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        sig_s = 'RAN'
        sig_c = 'RAN'

        files = self.data[index % self.dataset_size]
        idx = 0
        if not self.test_on_txt:
            self.ldr_paths = sorted(fnmatch.filter(files, '*_ldr_*.jpg'))
            self.gt_paths = sorted(fnmatch.filter(files, '*_gt.jpg' ) + fnmatch.filter(files, '*_gt.png') + fnmatch.filter(files, '*_gt.hdr'))
            self.ev_paths = sorted(fnmatch.filter(files, '*.npy'))
        else :
            self.gt_paths =  sorted(fnmatch.filter(files, '*.jpg' ) + fnmatch.filter(files, '*.png'), key = lambda x: int(x.split('/')[-1].split('_')[-1].split('-')[-1][:-4]))
            self.ldr_paths = self.gt_paths
            self.ev_paths = sorted(fnmatch.filter(files, '*.txt'), key = lambda x: int(x.split('/')[-1].split('_')[-1][:-4]))
        self.num_img = len(self.gt_paths)
        print(len(self.gt_paths), len(self.ev_paths), files[0])

        evs = []
        ldr = []
        gt_test = []
        ts = []
        for i in range(self.num_img):
            evs_, gt, ldr_, sig_s, sig_c = self.process(self.gt_paths[idx + i], self.ldr_paths[idx + i], self.ev_paths[idx + i], sig_s, sig_c)
            evs.append(evs_)
            ldr.append(ldr_)
            gt_test.append(gt)
            fhdr = np.power(ldr[-1], 2.2)
            t = 0.1 / np.average(fhdr)
            ts.append(t)
        name = self.gt_paths[idx]
        name = (name.split('/')[-3] + name.split('/')[-2] + '_' + name.split('/')[-1]).replace('_gt', '')
        gt = gt_test
        return {
            "ev": evs,
            "ldr": ldr,
            "hdr": gt,
            "t" : ts,
            "ldr_path": name
        }
    
    def __events_to_voxel_grid_pytorch(self, events, num_bins, width, height, normalization=False):
        assert(events.shape[1] == 4)
        assert(num_bins > 0)
        assert(width > 0)
        assert(height > 0)

        with torch.no_grad():

            events_torch = torch.from_numpy(events)
            events_torch = events_torch

            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
            if events.shape[0] == 0:
                return voxel_grid

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                index=xs[valid_indices] + ys[valid_indices]
                                * width + tis_long[valid_indices] * width * height,
                                source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                index=xs[valid_indices] + ys[valid_indices] * width
                                + (tis_long[valid_indices] + 1) * width * height,
                                source=vals_right[valid_indices])

            if normalization:
                # Normalize the event tensor (voxel grid) so that
                # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
                mean, stddev = voxel_grid[voxel_grid != 0].mean(), voxel_grid[voxel_grid != 0].std()
                voxel_grid[voxel_grid != 0] = (voxel_grid[voxel_grid != 0] - mean) / stddev
            
            voxel_grid = voxel_grid.view(num_bins, height, width)

        return voxel_grid    