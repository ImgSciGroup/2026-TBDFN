import os
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms as Transforms
from model.our import DTMF
from model.My_modul1e import Module
# from torch import optim
from torchvision import utils, transforms
import torch.utils.data as data
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from util.dataset3 import ISBI_Loader
from test import test
def wbce_loss(y_pred, label ,alpha):
    p = torch.sigmoid(y_pred)
    loss = torch.sum(- alpha * torch.log(p) * label - torch.log(1 - p) * (1 - label)) / ((32 ** 2)*3)
    return loss

def train_net(net, device, data_path, epochs=50, batch_size=3, lr=0.0003, ModelName='FC_EF', is_Transfer=False):
    print('Conrently, Traning Model is :::::'+ModelName+':::::')
    if is_Transfer:
        print("Loading Transfer Learning Model.........")
    else:
        print("No Using Transfer Learning Model.........")

    # 创建目录
    os.makedirs('txt', exist_ok=True)
    os.makedirs('mx', exist_ok=True)


    # 加载数据集
    isbi_dataset = ISBI_Loader(data_path=data_path, transform=Transforms.ToTensor())
    train_loader = data.DataLoader(dataset=isbi_dataset, batch_size=batch_size, shuffle=True)

    # 定义优化器与调度器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.9)

    # 损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 写入文件
    f_loss = open('txt/train_loss.txt', 'w')

    # 训练开始
    best_loss = float('inf')
    start = time.time()
    for epoch in range(1, epochs+1):
        net.train()
        n = 0
        total = 0
        print('==========================epoch = '+str(epoch)+'==========================')
        for image1, image2, image3, image4, label in train_loader:
            optimizer.zero_grad()
            image1, image2, image3, image4, label = image1.to(device), image2.to(device), image3.to(device), image4.to(device), label.to(device)

            pred = net(image1, image2, image3, image4)
            total_loss = criterion(pred, label)

            total_loss.backward()
            optimizer.step()

            total += total_loss.item()
            n += 1

            # print(f'{epoch}/{epochs} ::: lr={optimizer.param_groups[0]["lr"]:.6f} ::: batch {n}/{len(train_loader)}')
            # print('Loss/train', total_loss.item())
            # print('-----------------------------------------------------------------------')


        avg_loss = total / n

        f_loss.write(str(avg_loss) + '\n')
        scheduler1.step()
        print(epoch)
        print(avg_loss)
        if(epoch%10 == 0):
            torch.save(net.state_dict(), 'our/uk2-new-1.pth')
            test()
        # if epoch % 10 == 0:
        #     model_path = f'./mx/epoch-m_{epoch}_model.pth'
        #     torch.save(net.state_dict(), model_path)
        #     print(f"Saved model at epoch {epoch} to {model_path}")
        #     # test()


    end = time.time()
    f_loss.write(f'Total training time: {end - start:.2f} seconds\n')
    f_loss.close()


if __name__ == '__main__':
    device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MDSiamFLikeNet(n_classes=1)
    net.to(device)
    data_path = "./data/Landsat"
    train_net(net, device, data_path)
