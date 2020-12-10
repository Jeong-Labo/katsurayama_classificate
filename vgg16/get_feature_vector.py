import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torchvision import models
from utils import *


def main():
    # モデル定義（最後のドロップアウトとソフトマックスを排除した特徴量抽出器）
    model = models.vgg16_bn(pretrained=True)
    layers = list(model.classifier.children())[:-2]
    model.classifier = nn.Sequential(*layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    model.eval()

    # 画像データ
    ok_image_dir = '../../dataset/katsurayama-data/YAN111-04_ok/'
    image_paths = os.listdir(ok_image_dir)
    mask_img = Image.open('../../dataset/katsurayama-data/mask.jpg').convert('1')
    paste_mask_img = Image.new("RGB", mask_img.size, (0, 0, 0))

    # 前処理の定義
    transform = BaseTransfrom()  # use ILSVRC2012 default params

    # 特徴量をまとめる
    features_tensor = []
    for img_path in tqdm(image_paths):
        img = Image.open(ok_image_dir + img_path)
        img = Image.composite(img, paste_mask_img, mask_img)

        transformed_img = transform(img).to(device)  # torch.size([3, 224, 224])
        inputs = transformed_img.unsqueeze_(0)  # torch.size([1, 3, 224, 224])

        # 推論実行して特徴ベクトルの保存
        output_feature_tensor = model(inputs)  # torch.size([1, 4096])
        output_feature_tensor = output_feature_tensor.to('cpu')
        output_feature_tensor_np = output_feature_tensor.detach().numpy()
        output_feature_tensor_list = output_feature_tensor_np.tolist()
        features_tensor.append(output_feature_tensor_list[0])

    df = pd.DataFrame(features_tensor, index=None, columns=None)
    print(df.head())
    df.to_csv('features_vec/features_not_masked.csv', header=False, index=False)


if __name__ == '__main__':
    main()