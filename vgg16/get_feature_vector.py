import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torchvision import models
from utils import *


def save_features(features_tensor, save_dir):
    # 特徴量の保存
    df = pd.DataFrame(features_tensor, index=None, columns=None)
    print(df.head())
    df.to_csv(save_dir + 'features.csv', header=False, index=False)

    # 平均値の保存
    df_avg = df.mean()
    df_avg.to_csv(save_dir + 'features_avg.csv', header=False, index=False)


def main():
    # モデル定義（最後のドロップアウトとソフトマックスを排除した特徴量抽出器）
    model = models.vgg16_bn(pretrained=True)
    layers = list(model.classifier.children())[:-2]
    model.classifier = nn.Sequential(*layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    model.eval()

    # 画像データ(正解データだけ抽出)
    image_paths = make_datapath_list(phase="**", label='ok')
    # mask_path = '../../dataset/katsurayama-data/mask.jpg'
    mask_path = '../dataset/mask.jpg'
    mask_img = Image.open(mask_path).convert('1')
    paste_mask_img = Image.new("RGB", mask_img.size, (0, 0, 0))

    # 前処理の定義
    transform = BaseTransfrom()  # use ILSVRC2012 default params

    # 特徴量をまとめる
    features_tensor = []
    for img_path in tqdm(image_paths):
        img = Image.open(img_path)
        img = Image.composite(img, paste_mask_img, mask_img)

        transformed_img = transform(img).to(device)  # torch.size([3, 224, 224])
        inputs = transformed_img.unsqueeze_(0)  # torch.size([1, 3, 224, 224])

        # 推論実行して特徴ベクトルの保存
        output_feature_tensor = model(inputs)  # torch.size([1, 4096])
        output_feature_tensor = output_feature_tensor.to('cpu')
        output_feature_tensor_np = output_feature_tensor.detach().numpy()
        output_feature_tensor_list = output_feature_tensor_np.tolist()
        features_tensor.append(output_feature_tensor_list[0])

    save_features(features_tensor=features_tensor, save_dir='features_vec/')


if __name__ == '__main__':
    main()