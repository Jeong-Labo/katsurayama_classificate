import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def create_path_list(file_dir, phase):
    return_path_list = []
    for one_dir in file_dir:
        file_name_list = os.listdir(one_dir)
        path_list = [os.path.join(one_dir, file_name) for file_name in file_name_list]
        return_path_list.extend(path_list)

    random.shuffle(return_path_list)
    train_val_threshold = int(len(return_path_list) * 0.7)
    if phase == 'train':
        return return_path_list[:train_val_threshold]
    elif phase == 'val':
        return return_path_list[train_val_threshold:]
    else:
        exit(code='phase Error! Choose one {train, val}')


class OkL5Dataset(Dataset):
    def __init__(self, ok_dir, l5_dir, phase='train'):
        self.phase = phase
        self.filepath = create_path_list((ok_dir, l5_dir), phase=self.phase)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, index):
        img_path = self.filepath[index]
        img = Image.open(img_path)
        resized_img = to_tensor(img.resize(size=(256, 256)))
        resized_img = resized_img.to(self.device)

        if 'level5' in img_path:
            label = 1
        else:
            label = 0

        return resized_img, label
