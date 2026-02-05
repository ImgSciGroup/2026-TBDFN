import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,in_ch):
        super(Encoder, self).__init__()
        self.conv1_sar = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.conv1_op = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.conv1_diff = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64))


        self.conv2_sar = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128))
        self.conv2_op = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128))
        self.conv2_diff = nn.Sequential(
            nn.Conv2d(128*2, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128))


        self.conv3_sar = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        self.conv3_op = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        self.conv3_diff = nn.Sequential(
            nn.Conv2d(256*2, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))

    def forward(self, sar1, sar2, opt1, opt2):
        sar1_1 = self.conv1_sar(sar1)
        sar2_1 = self.conv1_sar(sar2)
        diff_1_sar = abs(sar1_1-sar2_1)

        opt1_1 = self.conv1_op(opt1)
        opt2_1 = self.conv1_op(opt2)
        diff_1_opt = abs(opt1_1 - opt2_1)
        diff_1 = self.conv1_diff(torch.cat([diff_1_sar,diff_1_opt],dim=1))


        sar1_2 = self.conv2_sar(sar1_1)
        sar2_2 = self.conv2_sar(sar2_1)
        diff_2_sar = abs(sar1_2 - sar2_2)


        opt1_2 = self.conv2_op(opt1_1)
        opt2_2 = self.conv2_op(opt2_1)
        diff_2_opt = abs(opt1_2 - opt2_2)
        diff_2=self.conv2_diff(torch.cat([diff_2_sar, diff_2_opt],dim=1))


        sar1_3 = self.conv3_sar(sar1_2)
        sar2_3 = self.conv3_sar(sar2_2)
        diff_3_sar = abs(sar1_3 - sar2_3)

        opt1_3 = self.conv3_op(opt1_2)
        opt2_3 = self.conv3_op(opt2_2)
        diff_3_opt = abs(opt1_3 - opt2_3)
        diff_3 = self.conv3_diff(torch.cat([diff_3_sar, diff_3_opt],dim=1))

        return sar1_3,sar2_3,opt1_3,opt2_3,diff_1,diff_2,diff_3



class DTMF(nn.Module):
    def __init__(self, in_ch):
        super(DTMF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch*2, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))

        self.conv1_diff= nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))

        self.conv =nn.Sequential(
            nn.Conv2d(256*3, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))

    def forward(self, sar1,sar2,opt1,opt2):
        time1 = self.conv1(torch.cat([sar1, opt1], dim=1))
        time2 = self.conv1(torch.cat([sar2, opt2], dim=1))
        diff_1 = self.conv1_diff(time1-time2)
        sar_diff = abs(sar1-sar2)
        opt_diff = abs(opt1 - opt2)

        return self.conv(torch.cat([diff_1, sar_diff, opt_diff], dim=1))


class FusionChangeDetectionNet(nn.Module):
    def __init__(self, n_classes=1):
        super(FusionChangeDetectionNet, self).__init__()
        self.Encoder = Encoder(3)
        self.fusion = DTMF(256)
        self.up1 =  nn.Sequential(
        nn.ConvTranspose2d(256 * 2, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128))

        self.up2 =  nn.Sequential(
        nn.ConvTranspose2d(128 * 2, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64))

        self.up3 =  nn.Sequential(
        nn.ConvTranspose2d(64 * 2, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64))

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, sar1, sar2, opt1, opt2):
        sar1_3,sar2_3,opt1_3,opt2_3,diff_1,diff_2,diff_3 = self.Encoder(sar1, sar2, opt1, opt2)
        fusion_map = self.fusion(sar1_3,sar2_3,opt1_3,opt2_3)
        decoder_3 = self.up1(     torch.cat([fusion_map,diff_3],dim=1))
        decoder_2 = self.up2(     torch.cat([decoder_3, diff_2],dim=1))
        decoder_1 = self.up3(     torch.cat([decoder_2, diff_1],dim=1))

        return self.final_conv(decoder_1)



if __name__ == "__main__":
    sar1 = torch.randn(1, 3, 256, 256)
    sar2 = torch.randn(1, 3, 256, 256)
    opt1 = torch.randn(1, 3, 256, 256)
    opt2 = torch.randn(1, 3, 256, 256)

    model = FusionChangeDetectionNet(n_classes=1)
    output = model(sar1, sar2, opt1, opt2)
    print(output.shape)  # 应输出 torch.Size([1, 1, 256, 256])