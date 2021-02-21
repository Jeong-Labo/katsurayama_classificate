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


def calc_mean_avg_precision(num_ok, num_sample):
    num_error = num_sample - num_ok
    true_p = num_error  # 結果がエラーで、実際もエラー
    false_p = 0         # 結果がエラーで、実際はセーフ
    false_n = num_ok    # 結果がセーフで、実際はエラー
    true_n = 0          # 結果がセーフで、実際もセーフ

    # 適合率：予測の正確さ
    precision = true_p / (true_p + false_p)
    # 再現率：結果として出てくるべきのもののうち、実際に出てきた割合
    recall = true_p / (true_p + false_n)

    print(f'{num_sample} samples => OK: {num_ok}, NG-Lv.5: {num_error}')
    print(f'Simple Acc: {num_error / num_sample:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')


def main():
    model_path = './model_weights/loss-0.0003.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MyModel(input_ch=1).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    label_name = ['ok', 'NG-level-5']
    data_dir = '../../dataset/katsurayama-data/YAN111-04_ng_level5_Otsu_binary/'
    # data_dir = '../../dataset/katsurayama-data/YAN111-04_ok_Otsu_binary/'
    file_path_list = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]

    n = 0
    for file_path in file_path_list:
        input_tensor = create_input_tensor(file_path, device)
        output = net(input_tensor)
        label_index = output.argmax()
        label = label_name[label_index]
        if label_index == 0:
            print(f'{os.path.basename(file_path)} => {label}')
            n += 1
    print('================================')

    calc_mean_avg_precision(n, len(file_path_list))


if __name__ == '__main__':
    main()
