from os.path import join
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from my_utils.my_model import MyModel
from my_utils.create_dataset import OkL5Dataset


def do_train(net, data_loader, criterion, scheduler, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('================================')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(data_loader[phase]):
                labels = labels.to(device)
                optimizer.zero_grad()

                # if phase == train -> calc gradient
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(data_loader[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(data_loader[phase].dataset)
            print(f'{phase}: Loss = {epoch_loss:.4f}, Acc = {epoch_acc:.4f}')

            # save model
            if phase == 'val' and epoch % 5 == 0:
                save_dir = './model_weights/'
                save_file_name = f'epoch-{epoch}_loss-{epoch_loss:.4}.pth'
                torch.save(net.state_dict(), join(save_dir, save_file_name))


def main():
    # create CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MyModel(input_ch=1).to(device)
    net.train()

    # load dataset
    ok_dir = 'D:/Users/kanazawahironori/Documents/dataset/katsurayama-data/YAN111-04_ok_Otsu_binary/'
    l5_dir = 'D:/Users/kanazawahironori/Documents/dataset/katsurayama-data/YAN111-04_ng_level5_Otsu_binary/'
    train_dataset = OkL5Dataset(ok_dir=ok_dir, l5_dir=l5_dir, phase='train')
    val_dataset = OkL5Dataset(ok_dir=ok_dir, l5_dir=l5_dir, phase='val')

    # create data loader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    data_loader = {'train': train_loader, 'val': val_loader}

    # setting train options
    lr = 1e-3
    optimizer = optim.SGD(params=net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5, step_size=5)
    criterion = nn.CrossEntropyLoss()

    # do train
    num_epochs = 25
    do_train(net=net,
             data_loader=data_loader,
             criterion=criterion,
             scheduler=scheduler,
             optimizer=optimizer,
             num_epochs=num_epochs,
             device=device)


if __name__ == '__main__':
    main()
