import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_tensor
from my_cnn_model.my_utils.my_model import MyModel


def create_input_tensor(file_path, device):
    img = Image.open(file_path)
    resized_img = to_tensor(img.resize(size=(256, 256)))
    resized_img = resized_img.to(device)
    return resized_img.unsqueeze(0)


def main():
    model_path = './model_weights/my_weight.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MyModel(input_ch=1).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    label_name = ['ok', 'NG-level-5']
    ok_data_dir = '../../dataset/katsurayama-data/exp_data/ok_otsu_bin/validation/'
    l5_data_dir = '../../dataset/katsurayama-data/exp_data/ng_level5_otsu_bin/validation/'
    ok_file_path_list = [os.path.join(ok_data_dir, file_name) for file_name in os.listdir(ok_data_dir)]
    l5_file_path_list = [os.path.join(l5_data_dir, file_name) for file_name in os.listdir(l5_data_dir)]
    file_path_list = ok_file_path_list + l5_file_path_list

    predict_ok = 0
    predict_l5 = 0
    true_p = 0  # 結果がエラーで、実際もエラー
    false_p = 0  # 結果がエラーで、実際はセーフ
    false_n = 0  # 結果がセーフで、実際はエラー
    true_n = 0  # 結果がセーフで、実際もセーフ
    for file_path in file_path_list:
        # 正解ラベルのセット
        if 'level5' in file_path:
            true_index = 1
        else:
            true_index = 0

        # 結果の出力
        input_tensor = create_input_tensor(file_path, device)
        output = net(input_tensor)
        label_index = output.argmax()

        # カウント
        if label_index == 0:
            predict_ok += 1
        else:
            predict_l5 += 1

        # 真のとき
        if label_index == true_index:
            if label_index == 0:
                true_n += 1  # 結果がセーフで、実際もセーフ
            else:
                true_p += 1  # 結果がエラーで、実際もエラー
        # 偽のとき
        else:
            if label_index == 0:
                false_n += 1  # 結果がセーフで、実際はエラー
            else:
                false_p += 1  # 結果がエラーで、実際はセーフ
            # ログも出力
            label = label_name[label_index]
            print(f'{os.path.basename(file_path)} => predicted: {label} (true label: {label_name[true_index]})')

    # 結果のログ出力
    print('================================')
    # 適合率：予測の正確さ
    precision = true_p / (true_p + false_p)
    # 再現率：結果として出てくるべきのもののうち、実際に出てきた割合
    recall = true_p / (true_p + false_n)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
    print(f'OK: {len(ok_file_path_list)} samples, NG-LV5: {len(l5_file_path_list)} samples '
          f'=> OK: {predict_ok}, NG-LV5: {predict_l5} classified')


if __name__ == '__main__':
    main()
