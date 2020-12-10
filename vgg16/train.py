from torchvision import models
from datetime import datetime
from tqdm import tqdm

from utils import *
from vgg16.create_dataset import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=10):
    # epoch ループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('---------------------------------------')

        # ファイル名用の変数（この値に応じて保存するか否かも検討する）
        epoch_loss_for_filename = 0

        # train & val ループ（それぞれやる）
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0      # 損失の和
            epoch_corrects = 0  # 正解数

            # データローダーからデータを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # optimizer初期化
                optimizer.zero_grad()

                # 順伝搬の計算
                with torch.set_grad_enabled(phase == 'train'):
                    out_puts = net(inputs)
                    loss = criterion(out_puts, labels)  # lossの計算
                    _, preds = torch.max(out_puts, 1)   # ラベルの予測

                    # 訓練時は誤差逆伝搬する
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_loss_for_filename = epoch_loss
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('#######################################\n')

        # モデルの保存
        dt_now = datetime.now()
        dt_now_for_filename = dt_now.strftime('%Y-%m-%d_%H%M%S')
        model_save_path = "../models/vgg16/weights_{}_{:.3}_.pth".format(dt_now_for_filename, epoch_loss_for_filename)
        torch.save(net.state_dict(), model_save_path)


def main():
    # パラメータの定義
    # 画像正規化用パラメータ（ILSVRC2012準拠）
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # データセット構築時のバッチサイズ
    batch_size = 32

    # 学習済みモデルを使うか否か（使うなら転移学習）
    use_pretrained = True

    # 学習用のパラメータ
    learning_rate = 5e-3
    momentum = 0.9
    num_epochs = 20

    # データセットの構築
    train_list = make_datapath_list('train')
    val_list = make_datapath_list('val')
    train_dataset = OriginalDataset(
        file_list=train_list,
        transform=ImageTransform(size, mean, std),
        phase='train'
    )
    val_dataset = OriginalDataset(
        file_list=val_list,
        transform=ImageTransform(size, mean, std),
        phase='val'
    )

    # データローダー定義
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    data_loaders_dict = {
        'train' : train_dataloader,
        'val' : val_dataloader
    }

    print("##### Dataset load has done. #####")

    # vgg-16モデルの呼び出し
    net = models.vgg16(pretrained=use_pretrained)

    # 1000クラス分類を4クラス分類に変更
    net.classifier[6] = torch.nn.Linear(net.classifier[6].in_features, 4)
    # print(net)

    # CPU or GPUに移動させ、訓練モードに変更
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net.to(device=device)
    net.train()

    print("##### VGG-16 model load has done and set TRAIN mode. #####")

    # 学習済みモデルを使うときは転移学習 -> パラメータの固定をする
    if use_pretrained:
        # 転移学習用のパラメータの格納先
        params_to_update = []
        # 学習用のパラメータ名 == ここだけ学習するようにif分岐
        update_params_names = ["classifier.6.weight", "classifier.6.bias"]
        for name, param in net.named_parameters():
            if name in update_params_names:
                param.requires_grad = True      # 学習する層のみ勾配計算する
                params_to_update.append(param)
                print(name)
            else:
                param.requires_grad = False
    else:
        params_to_update = []
        for name, param in net.named_parameters():
            param.requires_grad = True
            params_to_update.append(param)

    # 交差エントロピー誤差を使用
    cirterion = torch.nn.CrossEntropyLoss()

    # 最適化手法の設定
    optimizer = torch.optim.SGD(params=params_to_update, lr=learning_rate, momentum=momentum)

    # 学習実行
    train_model(
        net=net,
        dataloaders_dict=data_loaders_dict,
        criterion=cirterion,
        optimizer=optimizer,
        num_epochs=num_epochs
    )


if __name__ == '__main__':
    main()
