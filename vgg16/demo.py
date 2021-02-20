from PIL import Image
from tqdm import tqdm
import torch
# from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms

from utils import *


def main():
    # vgg-16モデルの呼び出し
    use_pretrained = False
    net = models.vgg16(pretrained=use_pretrained)

    # 1000クラス分類を4クラス分類に変更
    net.classifier[6] = torch.nn.Linear(net.classifier[6].in_features, 4)
    print(net)

    # CPU or GPUに移動させ、推論モードに変更
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device=device)
    net.eval()
    print("##### VGG-16 model load has done and set EVAL mode. #####")

    # 自作modelのロード
    load_path = 'model_weights/weights_2020-10-26_143533_3.46_.pth'
    load_weights = torch.load(load_path, map_location=device)
    net.load_state_dict(load_weights)
    print("##### Dataset load has done. #####")

    # ラベル定義と画像データの整理
    out_labels = ['ok', 'level1', 'level3', 'level5']
    image_paths = make_datapath_list('val')

    # 一枚ずつ取り出して推論を実行
    num_preds = len(image_paths)
    num_correct = 0
    for img_path in tqdm(image_paths):
        img = Image.open(img_path)
        transform = BaseTransfrom()     # use ILSVRC2012 default params
        transformed_img = transform(img).to(device)     # torch.size([3, 224, 224])
        inputs = transformed_img.unsqueeze_(0)          # torch.size([1, 3, 224, 224])
        correct = img_path.split("\\")[-2]  # set correct label

        # 推論実行とラベルに変換
        out_tensor = net(inputs)
        pred_index = torch.argmax(out_tensor)
        result = out_labels[pred_index]
        if result == correct:
            num_correct += 1
        # print("pred_result :", result, "| teacher_label :", correct)

    print("num_correct : {}/{}, ACC = {:.3}".format(num_correct, num_preds, num_correct/num_preds))


if __name__ == '__main__':
    main()