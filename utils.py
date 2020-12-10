import os
from glob import glob
from torchvision import transforms


def make_datapath_list(phase='train'):
    rootpath = "./dataset/"
    target_path = os.path.join(rootpath + phase + '/**/*.jpg')
    path_list = [path for path in glob(target_path)]
    return path_list


class BaseTransfrom():
    """
    resize <int>    : リサイズ先の大きさ
    mean <(R,G,B)>  : 各チャンネルの平均値
    std <(R,G,B)>   : 各チャンネルの標準偏差
    """
    def __init__(self, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.base_transform(img)