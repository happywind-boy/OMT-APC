import time

# import monai
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding,
                              stride=stride)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(Conv3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding,
                              stride=stride)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class Conv9(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4, stride=3):
        super(Conv9, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding,
                              stride=stride)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.ac(x)

        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2):
        super(UpConv, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding,
                                       output_padding=padding,
                                       stride=stride)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.ac(x)

        return x


class PSnet(nn.Module):
    def __init__(self, in_channels,
                 out_channels_IDH=2, out_channels_1p19q=2, out_channels_Grade=2,
                 dropout=0.2):
        super(PSnet, self).__init__()
        self.encoder1 = nn.Sequential(
            Conv3(in_channels, 32),
            nn.Dropout(dropout),
        )
        self.encoder2 = nn.Sequential(
            Conv3(32, 32, stride=2),
            nn.BatchNorm3d(32),
            Conv3(32, 64),
            Conv3(64, 64),
            nn.Dropout(dropout),
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            Conv3(64, 128),
            Conv3(128, 128),
            nn.Dropout(dropout),
        )

        self.encoder4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            Conv3(128, 256),
            Conv3(256, 256),
            nn.Dropout(dropout),
        )

        self.brige = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            Conv3(256, 512),
            Conv3(512, 512),
        )
        self.upbrige = nn.Sequential(
            nn.Dropout(dropout),
            UpConv(512, 256),
        )

        self.decoder1 = nn.Sequential(
            Conv3(512, 256),
            Conv3(256, 256),
            nn.Dropout(dropout),
        )
        self.updecoder1 = nn.Sequential(
            nn.BatchNorm3d(256),
            UpConv(256, 128),
        )

        self.decoder2 = nn.Sequential(
            Conv3(256, 128),
            Conv3(128, 128),
            nn.Dropout(dropout),
        )
        self.updecoder2 = nn.Sequential(
            nn.BatchNorm3d(128),
            UpConv(128, 64),
        )

        self.decoder3 = nn.Sequential(
            Conv3(128, 64),
            Conv3(64, 64),
            nn.Dropout(dropout),
        )
        self.updecoder3 = nn.Sequential(
            nn.BatchNorm3d(64),
            UpConv(64, 32, kernel_size=3, stride=2),
        )

        self.decoder4 = nn.Sequential(
            Conv3(64, 32),
            Conv3(32, 32),
            nn.Dropout(dropout),
        )
        # self.toseg = nn.Sequential(
        #     nn.BatchNorm3d(32),
        #     Conv1(32, out_channels_seg),
        # )

        self.branch_IDH = nn.Sequential(
            nn.Linear(1472, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels_IDH),
            nn.Softmax(dim=1),
        )

        self.branch_1p19q = nn.Sequential(
            nn.Linear(1472, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels_1p19q),
            nn.Softmax(dim=1),
        )
        self.branch_Grade = nn.Sequential(
            nn.Linear(1472, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels_Grade),
            nn.Softmax(dim=1),
        )
        self.maxpol = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.brige(e4)

        b1_2 = self.upbrige(b)

        d1 = self.decoder1(torch.cat([e4, b1_2], dim=1))
        d1_2 = self.updecoder1(d1)

        d2 = self.decoder2(torch.cat([e3, d1_2], dim=1))
        d2_2 = self.updecoder2(d2)

        d3 = self.decoder3(torch.cat([e2, d2_2], dim=1))
        d3_2 = self.updecoder3(d3)

        d4 = self.decoder4(torch.cat([e1, d3_2], dim=1))

        # out_seg = self.toseg(d4)

        fe_all = [e1, e2, e3, e4, b, d1, d2, d3, d4]
        fe_all = [self.maxpol(x).view(x.size(0), -1) for x in fe_all]
        fe_all = torch.cat(fe_all, dim=1)

        fe_all = self.dropout(fe_all)

        out_IDH = self.branch_IDH(fe_all)
        # out_1p19q = self.branch_1p19q(fe_all)
        # out_Grade = self.branch_Grade(fe_all)

        return out_IDH
        

class PSFnet(nn.Module):
    def __init__(self, in_channels, out_channels_IDH=2, out_channels_1p19q=2, out_channels_Grade=2,
                 feature_channels=12, dropout=0.2):
        super(PSFnet, self).__init__()
        self.feature_channels = feature_channels
        self.encoder1 = nn.Sequential(
            Conv3(in_channels, 32),
            nn.Dropout(dropout),
        )
        self.encoder2 = nn.Sequential(
            Conv3(32, 32, stride=2),
            nn.BatchNorm3d(32),
            Conv3(32, 64),
            Conv3(64, 64),
            nn.Dropout(dropout),
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            Conv3(64, 128),
            Conv3(128, 128),
            nn.Dropout(dropout),
        )

        self.encoder4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            Conv3(128, 256),
            Conv3(256, 256),
            nn.Dropout(dropout),
        )

        self.brige = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            Conv3(256, 512),
            Conv3(512, 512),
        )
        self.upbrige = nn.Sequential(
            nn.Dropout(dropout),
            UpConv(512, 256),
        )

        self.decoder1 = nn.Sequential(
            Conv3(512, 256),
            Conv3(256, 256),
            nn.Dropout(dropout),
        )
        self.updecoder1 = nn.Sequential(
            nn.BatchNorm3d(256),
            UpConv(256, 128),
        )

        self.decoder2 = nn.Sequential(
            Conv3(256, 128),
            Conv3(128, 128),
            nn.Dropout(dropout),
        )
        self.updecoder2 = nn.Sequential(
            nn.BatchNorm3d(128),
            UpConv(128, 64),
        )

        self.decoder3 = nn.Sequential(
            Conv3(128, 64),
            Conv3(64, 64),
            nn.Dropout(dropout),
        )
        self.updecoder3 = nn.Sequential(
            nn.BatchNorm3d(64),
            UpConv(64, 32, kernel_size=3, stride=2),
        )

        self.decoder4 = nn.Sequential(
            Conv3(64, 32),
            Conv3(32, 32),
            nn.Dropout(dropout),
        )
        # self.toseg = nn.Sequential(
        #     nn.BatchNorm3d(32),
        #     Conv1(32, out_channels_seg),
        # )

        self.branch_IDH = nn.Sequential(
            nn.Linear(1472, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels_IDH),
            nn.Softmax(dim=1),
        )

        self.branch_1p19q = nn.Sequential(
            nn.Linear(1472, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels_1p19q),
            nn.Softmax(dim=1),
        )
        self.branch_Grade = nn.Sequential(
            nn.Linear(1472, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels_Grade),
            nn.Softmax(dim=1),
        )

        self.branch_Feature = nn.Sequential(
            nn.Linear(64 + self.feature_channels, 8),
            nn.Linear(8, out_channels_IDH),
            nn.Softmax(dim=1),
        )

        self.maxpol = nn.AdaptiveMaxPool3d(1)
        self.FC = nn.Linear(1472, 64)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Softmax(dim=1)

    def forward(self, x, y):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.brige(e4)

        b1_2 = self.upbrige(b)

        d1 = self.decoder1(torch.cat([e4, b1_2], dim=1))
        d1_2 = self.updecoder1(d1)

        d2 = self.decoder2(torch.cat([e3, d1_2], dim=1))
        d2_2 = self.updecoder2(d2)

        d3 = self.decoder3(torch.cat([e2, d2_2], dim=1))
        d3_2 = self.updecoder3(d3)

        d4 = self.decoder4(torch.cat([e1, d3_2], dim=1))

        # out_seg = self.toseg(d4)

        fe_all = [e1, e2, e3, e4, b, d1, d2, d3, d4]
        fe_all = [self.maxpol(x).view(x.size(0), -1) for x in fe_all]
        fe_all = torch.cat(fe_all, dim=1)

        fe_all = self.dropout(fe_all)
        fe_all = self.FC(fe_all)
        fe_cat = torch.cat((fe_all, y), dim=-1)

        out_IDH = self.branch_Feature(fe_cat)
        # out_1p19q = self.branch_1p19q(fe_all)
        # out_Grade = self.branch_Grade(fe_all)

        return out_IDH


if __name__ == '__main__':
    start = time.time()
    model = PSnet(3, 2)

    x = torch.randn((2, 3, 128, 128, 128))

    y1 = model(x)
    print(time.time() - start)

    # print(y.shape)
    print(y1)

