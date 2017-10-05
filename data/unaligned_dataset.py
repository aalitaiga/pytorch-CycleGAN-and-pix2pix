import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

import torch
from fuel.datasets import H5PYDataset
import PIL
from PIL import Image
from pdb import set_trace as st
import random

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

class FuelUnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.transform = get_transform(opt)

        phase = 'train' if opt.isTrain else 'valid'
        self.f = H5PYDataset(opt.dataroot, which_sets=(phase,))
        self.A_size = self.f.num_examples
        self.B_size = self.f.num_examples

        # Get sequence length
        handle = self.f.open()
        data = self.f.get_data(handle, slice(0, 1))
        self.seq_len = data[1].shape[1]
        self.f.close(handle)

    def __getitem__(self, index):
        idx_A = index % self.A_size
        idx_A = (idx_A // self.seq_len, idx_A % self.seq_len)
        idx_B = random.randint(0, self.B_size - 1)
        idx_B = (idx_B // self.seq_len, idx_B % self.seq_len)
        handle = self.f.open()

        # print(index, idx_A)
        # import ipdb; ipdb.set_trace()
        # Get img and joints from domain A and B
        A_data = self.f.get_data(handle, slice(idx_A[0], idx_A[0]+1))
        B_data = self.f.get_data(handle, slice(idx_B[0], idx_B[0]+1))
        # import ipdb; ipdb.set_trace()
        A_img, B_img = A_data[1][0, idx_A[1], :, :, :], B_data[3][0, idx_B[1], :, :, :]
        A_img, B_img = Image.fromarray(A_img), Image.fromarray(B_img)
        A_img, B_img = self.transform(A_img), self.transform(B_img)

        d = {
            'A': A_img,
            'B': B_img,
            'A_paths': 0,
            'B_paths': 0
        }

        if self.opt.add_state:
            A_joints, B_joints = A_data[8][0, idx_A[1], :], B_data[4][0, idx_B[1], :]
            A_joints, B_joints = torch.from_numpy(A_joints).float(), torch.from_numpy(A_joints).float()
            d.update({'A_joints': A_joints, 'B_joints': B_joints})
        self.f.close(handle)

        return d

    def __len__(self):
        return self.f.num_examples * self.seq_len

    def name(self):
        return 'FuelUnalignedDataset'
