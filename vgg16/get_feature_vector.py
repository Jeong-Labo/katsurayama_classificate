import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torchvision import models
from utils import *

# デバイス
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 前処理の定義
transform = BaseTransfrom()  # use ILSVRC2012 default params

# マスク画像
mask_img = Image.open('../../dataset/katsurayama-data/mask.jpg').convert('1')
paste_mask_img = Image.new("RGB", mask_img.size, (0, 0, 0))


def get_features(model, image_dir, image_paths, rank):
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

    save_features(features_tensor=features_tensor, save_dir='features_vec/', rank=rank)


def save_features(features_tensor, save_dir, rank):
    # 特徴量の保存
    df = pd.DataFrame(features_tensor, index=None, columns=None)
    print(df.head())
    df.to_csv(save_dir + f'features_{rank}.csv', header=False, index=False)

    # 平均値の保存
    df_avg = df.mean()
    df_avg.to_csv(save_dir + f'features_{rank}_avg.csv', header=False, index=False)


def main():
    # モデル定義（最後のドロップアウトとソフトマックスを排除した特徴量抽出器）
    model = models.vgg16_bn(pretrained=True)
    layers = list(model.classifier.children())[:-2]
    model.classifier = nn.Sequential(*layers)
    model.to(device=device)
    model.eval()

    # 画像データ
    ok_image_dir = '../../dataset/katsurayama-data/YAN111-04_ok/'
    # l1_image_dir = '../../dataset/katsurayama-data/YAN111-04_ng_level1/'
    # l3_image_dir = '../../dataset/katsurayama-data/YAN111-04_ng_level3/'
    l5_image_dir = '../../dataset/katsurayama-data/YAN111-04_ng_level5/'
    ok_image_paths = os.listdir(ok_image_dir)
    # l1_image_paths = os.listdir(l1_image_dir)
    # l3_image_paths = os.listdir(l3_image_dir)
    l5_image_paths = os.listdir(l5_image_dir)

    get_features(model, ok_image_dir, ok_image_paths, 'ok')
    # get_features(model, l1_image_dir, l1_image_paths, 'l1')
    # get_features(model, l3_image_dir, l3_image_paths, 'l3')
    get_features(model, l5_image_dir, l5_image_paths, 'l5')


if __name__ == '__main__':
    main()