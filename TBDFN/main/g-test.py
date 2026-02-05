import os
import glob
import torch
import argparse
import torch.nn  as nn
import torch.nn.functional  as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms


# 测试数据集类
class TestDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, "testA") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "testB") + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        # 记录原始尺寸
        orig_size_A = image_A.size  # (width, height)
        orig_size_B = image_B.size

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {
            "A": item_A,
            "B": item_B,
            "orig_size_A": torch.tensor(orig_size_A),
            "orig_size_B": torch.tensor(orig_size_B)
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    # 生成器模型类


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def test():
    # 超参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/shuguang', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--cpu', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='save/datasets/shuguang/G_AB_149.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='save/datasets/shuguang/G_BA_149.pth',
                        help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    # 初始化生成器
    netG_A2B = GeneratorResNet((opt.channels, 256, 256), opt.n_residual_blocks)
    netG_B2A = GeneratorResNet((opt.channels, 256, 256), opt.n_residual_blocks)

    # 设备设置
    if opt.cpu:
        netG_A2B.cpu()
        netG_B2A.cpu()

        # 加载训练好的模型
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    # 设置为评估模式
    netG_A2B.eval()
    netG_B2A.eval()

    # 数据转换
    transforms_ = [
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # 数据加载器
    dataloader = DataLoader(
        TestDataset(opt.dataroot, transforms_=transforms_),
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.n_cpu
    )

    # 创建输出目录
    os.makedirs('output/A', exist_ok=True)
    os.makedirs('output/B', exist_ok=True)

    # 测试过程
    for i, batch in enumerate(dataloader):
        real_A = batch['A']
        real_B = batch['B']
        orig_size_A = batch['orig_size_A'][0].numpy()  # (width, height)
        orig_size_B = batch['orig_size_B'][0].numpy()

        # 通过生成器生成图像
        with torch.no_grad():
            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)

        # 反归一化到[0,1]范围
        fake_A = 0.5 * (fake_A + 1.0)
        fake_B = 0.5 * (fake_B + 1.0)

        # 调整输出图像大小以匹配原始输入尺寸
        fake_A_resized = F.interpolate(
            fake_A,
            size=(orig_size_A[1], orig_size_A[0]),  # (height, width)
            mode='bilinear',
            align_corners=False
        )
        fake_B_resized = F.interpolate(
            fake_B,
            size=(orig_size_B[1], orig_size_B[0]),  # (height, width)
            mode='bilinear',
            align_corners=False
        )

        # 保存图片
        save_image(fake_A_resized, f'output/A/{i + 1:04d}.png')
        save_image(fake_A_resized, f'./data/Landsat/pred/image3/01.bmp')
        save_image(fake_B_resized, f'output/B/{i + 1:04d}.png')
        save_image(fake_B_resized, f'./data/Landsat/pred/image4/01.bmp')
        print(f'processing ({i + 1:04d})-th image... (original size: {orig_size_A[0]}x{orig_size_A[1]})')

    print("测试完成")


if __name__ == '__main__':
    test()
