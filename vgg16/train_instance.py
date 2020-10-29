from datetime import datetime
from tqdm import tqdm
import torch


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