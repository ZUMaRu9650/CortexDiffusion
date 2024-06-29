import os
import glob
import numpy as np
import random
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
from .geometry import compute_operators, compute_hks_autoscale


class Data(Dataset):
    def __init__(self, root, folds, k):
        super(Data, self).__init__()
        self.k = k
        if type(folds) == list:
            self.file_paths = []
            self.path_list = []
            for fold in folds:
                self.file_paths = self.file_paths + sorted(glob.glob(os.path.join(root, fold, '*.pt')))
                self.path_list = self.path_list + sorted(os.listdir(os.path.join(root, fold)))

        elif type(folds) == str:
            self.file_paths = sorted(glob.glob(os.path.join(root, folds, '*.pt')))
            self.path_list = sorted(os.listdir(os.path.join(root, folds)))
    
    def __getitem__(self,index):
        file_path = self.file_paths[index]
        basename, _ = os.path.splitext(self.path_list[index])

        data = torch.load(file_path)
        coords, labels, mass, evals, evecs, gradX, gradY = data["vertices"], data["label"], data["massvec"], data["evals"][0:self.k], data["evecs"][:,0:self.k], data["gradX"], data["gradY"]

        return basename, coords.float(), labels.long(), mass, evals, evecs, gradX, gradY
    
    def __len__(self):
        return len(self.file_paths)