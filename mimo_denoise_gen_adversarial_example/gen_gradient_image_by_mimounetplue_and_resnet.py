import torch
import torch.nn as nn
from torch import optim

import os
from PIL import Image

from torchvision.models import resnet18
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from torch.utils.data import Dataset, DataLoader

from mimo_model import MIMOUNetPlus

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        # 获取文件夹中所有图片文件的路径
        self.image_files = [
            f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))
        ][
            :10
        ]  # 只取前10张

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图片
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image


def test_dataloader(image_dir, batch_size=1, num_workers=0):
    # 定义图像变换
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 调整图片大小
            transforms.ToTensor(),  # 转换为张量
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化（可选）
        ]
    )

    dataset = ImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


def denoise_image(model, model_dir, data_dir, result_dir, save_image=True):
    state_dict = torch.load(model_dir)
    model.load_state_dict(state_dict["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = test_dataloader(data_dir, batch_size=1, num_workers=4)
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, _ = data
            input_img = input_img.to(device)
            _ = model(input_img)

            if iter_idx == 20:
                break
        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)

            pred = model(input_img)[2]

            pred_clip = torch.clamp(pred, 0, 1)

            if save_image:
                save_name = os.path.join(result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
                pred.save(save_name)




class denoise_net(nn.Module):
    def __init__(self, model_dir, data_dir, result_dir):
        super(denoise_net, self).__init__()
        self.result_dir=result_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataloader = test_dataloader(data_dir, batch_size=1, num_workers=4)
        
        self.model = MIMOUNetPlus()
        self.model.load_state_dict(torch.load(model_dir)["model"])
        self.model.to(self.device)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            # Main Evaluation
            for iter_idx, data in enumerate(self.dataloader):
                input_img, label_img, name = data
                input_img = input_img.to(self.device)

                pred = self.model(input_img)[2]

                pred_clip = torch.clamp(pred, 0, 1)

                save_name = os.path.join(self.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
                pred.save(save_name)


class classify_net(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(classify_net, self).__init__()
        self.num_classes = num_classes

        model = resnet18(pretrained=pretrained)

        # 如果需要修改分类层的输出类别数
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    def load_pretrained_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def forward(self, x):
        pass


class gradient_attack:
    def __init__(
        self,
        max_iteration=200,
        lr=0.05,
        gen_model=denoise_net,
        classify_model=classify_net,
        org_img=None,
        target_label=None,
    ):
        self.max_iteration = max_iteration
        self.lr = lr
        self.gen_model = gen_model
        self.classify_model = classify_model
        self.org_img = org_img
        self.target_label = target_label

    def run(self):
        region = self.org_img[:, :224, :224].clone().detach()
        region.requires_grad = True

        lossfunc = nn.MSELoss()
        optimizer = optim.Adam([region], lr=self.lr)
        for i in range(self.max_iteration):

            denoise_image = denoise_net(self.org_img)
            classify_result = classify_net(denoise_image)
            # 方法1：获取预测的类别（概率最大的类别）
            pred_label = torch.argmax(classify_result, dim=1)  # 对于批处理数据

            # 判断是否与目标标签相同
            is_successful = pred_label == self.target_label

            if is_successful:
                print(f"攻击成功！在第{i}次尝试后达到目标")
                return self.org_img

            loss = lossfunc(classify_result, self.target_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.org_img[:, :200, :200] = region.detach()

        print("攻击失败")
        return self.org_img
