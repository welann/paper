import os
import argparse
from PIL import Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional 
from torch.utils.data import Dataset, DataLoader


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        bias=True,
        norm=False,
        relu=True,
        transpose=False,
    ):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(
                out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True
            ),
            BasicConv(
                out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True
            ),
            BasicConv(
                out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True
            ),
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res=20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res),
                EBlock(base_channel * 2, num_res),
                EBlock(base_channel * 4, num_res),
            ]
        )

        self.feat_extract = nn.ModuleList(
            [
                BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
                BasicConv(
                    base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2
                ),
                BasicConv(
                    base_channel * 2,
                    base_channel * 4,
                    kernel_size=3,
                    relu=True,
                    stride=2,
                ),
                BasicConv(
                    base_channel * 4,
                    base_channel * 2,
                    kernel_size=4,
                    relu=True,
                    stride=2,
                    transpose=True,
                ),
                BasicConv(
                    base_channel * 2,
                    base_channel,
                    kernel_size=4,
                    relu=True,
                    stride=2,
                    transpose=True,
                ),
                BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.Decoder = nn.ModuleList(
            [
                DBlock(base_channel * 4, num_res),
                DBlock(base_channel * 2, num_res),
                DBlock(base_channel, num_res),
            ]
        )

        self.Convs = nn.ModuleList(
            [
                BasicConv(
                    base_channel * 4,
                    base_channel * 2,
                    kernel_size=1,
                    relu=True,
                    stride=1,
                ),
                BasicConv(
                    base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1
                ),
            ]
        )

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList(
            [
                AFF(base_channel * 7, base_channel * 1),
                AFF(base_channel * 7, base_channel * 2),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, "blur/"))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, "blur", self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, "sharp", self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = functional.to_tensor(image)
            label = functional.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split(".")
            if splits[-1] not in ["png", "jpg", "jpeg"]:
                raise ValueError


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, "test")
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MIMOUNetPlus()
    state_dict = torch.load(args.model_dir)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=4)
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            pred = model(input_img)[2]

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # print("pred_numpy: ", pred_numpy)
            # print("label_numpy: ", label_numpy)
            save_name = os.path.join(args.result_dir, name[0])
            pred_clip += 0.5 / 255
            pred = functional.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
            pred.save(save_name)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--data_dir", type=str, default="dataset/GOPRO")
    args = parser.parse_args()
    args.result_dir = os.path.join('mimounet_results/')
    main(args)
