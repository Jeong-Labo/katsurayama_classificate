from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        """
        phase: 'train' or 'val'
        """
        return self.data_transform[phase](img)


class OriginalDataset(data.Dataset):
    """
    オリジナルのデータセットを呼び出し。
    ------
    file_list <list>    : ファイルパスのリスト
    transform <object>  : 前処理クラスのインスタンス
    phase <str>         : 'train' or 'test' (default=='train')
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        # 画像の枚数を返す
        return len(self.file_list)

    def __getitem__(self, index):
        """
        前処理後の画像テンソルデータとそのラベルを返す。
        """
        # 画像ロード
        img_path = self.file_list[index]
        img = Image.open(img_path)

        # 前処理の実施
        img_transformed = self.transform(img, self.phase).to(device)  # torch.size([3,224,224])

        # ラベルを確保
        # ./dataset/{train, val}/{ok, level1, level3, level5}/***.jpg
        # 　↑の{ok, ..., level5}を確保したい（'/'で分けたときの後ろから二番目の要素）
        label = img_path.split('\\')[-2]
        # print(label)

        # ラベルの数値化
        label_index = -1
        if label == 'ok':
            label_index = 0
        elif label == 'level1':
            label_index = 1
        elif label == 'level3':
            label_index = 2
        elif label == 'level5':
            label_index = 3
        else:
            print('Error! Unknown label has appeared.')
            exit(code=1)

        return img_transformed, label_index
