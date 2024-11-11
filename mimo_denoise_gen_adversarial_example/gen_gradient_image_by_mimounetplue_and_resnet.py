import torch
import torch.nn as nn
from torch import optim

import os
import argparse
from PIL import Image

from torchvision.models import resnet18
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


from mimo_model import MIMOUNetPlus


class denoise_net(nn.Module):
    def __init__(self, model_dir, result_dir):
        super(denoise_net, self).__init__()
        self.result_dir = result_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)

        self.model = MIMOUNetPlus()
        self.model.load_state_dict(
            torch.load(
                model_dir, map_location=torch.device(self.device), weights_only=True
            )["model"]
        )
        self.model.to(self.device)
        self.model.eval()

    # 这里的image，需要to_tensor一下
    def forward(self, input_img, label_img: str):
        # print(input_img)
        with torch.no_grad():
            input_img = input_img.to(self.device)
            pred = self.model(input_img)[2]
            pred_clip = torch.clamp(pred, 0, 1)
            save_name = os.path.join(self.result_dir, label_img + ".png")
            pred_clip += 0.5 / 255
            pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
            pred.save(save_name)

            return pred_clip


class classify_net(nn.Module):
    def __init__(self, num_classes=1000, classify_model_dir=None):
        super(classify_net, self).__init__()
        self.num_classes = num_classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", device)
        self.model = torch.load(classify_model_dir, map_location=torch.device(device))
        # self.model = resnet18()

        # # 如果需要修改分类层的输出类别数
        # if num_classes != 1000:
        #     self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # self.model.load_state_dict(torch.load(classify_model_dir,map_location=torch.device('cpu')))
        self.model.eval()

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        print("classify input image shape: ", x.shape)
        return self.model(x)


class gradient_attack:
    def __init__(
        self,
        max_iteration=200,
        lr=0.05,
        gen_model=denoise_net,
        classify_model=classify_net,
        org_img=None,
        target_label=None,
        gen_model_dir=None,
        classify_model_dir=None,
    ):
        self.max_iteration = max_iteration
        self.lr = lr
        self.gen_model = gen_model
        self.classify_model = classify_model
        self.org_img = org_img
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_label = target_label.to(self.device)
        self.gen_model_dir = gen_model_dir
        self.classify_model_dir = classify_model_dir

    def run(self):
        dnet = self.gen_model
        cnet = self.classify_model

        region = self.org_img[:, :224, :224].clone().detach()
        region.requires_grad = True
        
        orginal_label = 0
        with torch.no_grad():
            denoise_image = dnet(self.org_img, "org_img_label")
            region = denoise_image[:, :224, :224].clone().detach()
            
            region = region.to(self.device)
            classify_result = cnet(region)
            print("classify_result: ", classify_result)
            orginal_label = torch.argmax(classify_result, dim=1)  # 对于批处理数据
            print(f"pred_label: {orginal_label}")
            
        lossfunc = nn.MSELoss()
        optimizer = optim.Adam([region], lr=self.lr)
        for i in range(self.max_iteration):
            if i % 10 == 0:
                res = F.to_pil_image(self.org_img.squeeze(0).cpu(), "RGB")
                res.save(f"running_attack_{i}.png")

            denoise_image = dnet(self.org_img, "org_img_label")
            region = denoise_image[:, :224, :224].clone().detach()
            region.requires_grad = True
            classify_result = cnet(region)
            # print("classify_result: ", classify_result)
            # print("self.target_label: ", self.target_label)
            pred_label = torch.argmax(classify_result, dim=1)  # 对于批处理数据
            print(f"pred_label: {pred_label}")
            
            if pred_label != orginal_label:
                res = F.to_pil_image(self.org_img.squeeze(0).cpu(), "RGB")
                res.save(f"success_{i}.png")
                print("攻击成功")
                return self.org_img
            # 判断是否与目标标签相同
            is_successful = pred_label == self.target_label

            if is_successful:
                print(f"攻击成功！在第{i}次尝试后达到目标")
                return self.org_img
            
            target_tensor = torch.zeros(classify_result.shape).to(self.device)
            target_tensor[0, int(self.target_label[0])] = 1.0
            loss = -lossfunc(classify_result, target_tensor)
            print(f"{i}. loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("region: ",region)
            self.org_img[:, :224, :224] = region.detach()
            # print(self.org_img)
            
        print("攻击失败")
        return self.org_img


def test_gen_net(args):
    image_path = args.image_path
    test_image = Image.open(image_path)
    test_image_tensor = F.to_tensor(test_image)
    test_image_tensor = test_image_tensor.unsqueeze(0)

    print(test_image_tensor.shape)
    dnet = denoise_net(args.gen_model_dir, args.result_dir)

    predict = dnet(test_image_tensor, "test_img_label")
    print(predict.shape)


def test_classify_net(args):
    image_path = args.image_path
    test_image = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ResNet-18 需要 224x224 的输入
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.unsqueeze(0)

    print(test_image_tensor.shape)
    cnet = classify_net(10, args.classify_model_dir)
    result = cnet(test_image_tensor)
    print("cnet: ", result)
    print("cnet argmax: ", torch.argmax(result))


def test_gradient_attack(args):
    image_path = args.image_path
    test_image = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.unsqueeze(0)

    dnet = denoise_net(args.gen_model_dir, args.result_dir)
    cnet = classify_net(10, args.classify_model_dir)
    attackmet = gradient_attack(
        200,
        0.05,
        dnet,
        cnet,
        test_image_tensor,
        torch.tensor([1], dtype=torch.float),
        args.gen_model_dir,
        args.classify_model_dir,
    )

    result_image = attackmet.run()
    print(result_image.shape)
    res = F.to_pil_image(result_image.squeeze(0).cpu(), "RGB")
    res.save("target_label.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=2, type=int, choices=[0, 1, 2])
    parser.add_argument("--gen_model_dir", default=None, type=str)
    parser.add_argument("--image_path", default=None, type=str)
    parser.add_argument("--result_dir", default=None, type=str)
    parser.add_argument("--classify_model_dir", default=None, type=str)
    args = parser.parse_args()
    if args.mode == 0:
        test_gen_net(args)
    elif args.mode == 1:
        test_classify_net(args)
    elif args.mode == 2:
        test_gradient_attack(args)
    else:
        print("mode is not correct")
