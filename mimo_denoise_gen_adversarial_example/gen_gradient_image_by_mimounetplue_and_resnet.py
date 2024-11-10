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
        self.result_dir=result_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ",self.device)
        
        self.model = MIMOUNetPlus()
        self.model.load_state_dict(torch.load(model_dir,map_location=torch.device(self.device),weights_only=True)["model"])
        self.model.to(self.device)
        self.model.eval()

    #这里的image，需要to_tensor一下
    def forward(self, input_img,label_img:str):
        with torch.no_grad():
            input_img = input_img.to(self.device)
            pred = self.model(input_img)[2]
            pred_clip = torch.clamp(pred, 0, 1)
            save_name = os.path.join(self.result_dir, label_img+".png")
            pred_clip += 0.5 / 255
            pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
            pred.save(save_name)
        
            return pred_clip


class classify_net(nn.Module):
    def __init__(self, num_classes=1000, classify_model_dir=None):
        super(classify_net, self).__init__()
        self.num_classes = num_classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ",device)
        self.model=torch.load(classify_model_dir,map_location=torch.device(device))
        # self.model = resnet18()

        # # 如果需要修改分类层的输出类别数
        # if num_classes != 1000:
        #     self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # self.model.load_state_dict(torch.load(classify_model_dir,map_location=torch.device('cpu')))
        self.model.eval()
        
        
    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x=x.to(device)
        x=x.unsqueeze(0)
        print("classify input image shape: ",x.shape)
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
        self.target_label = target_label
        self.gen_model_dir = gen_model_dir
        self.classify_model_dir = classify_model_dir
        

    def run(self):
        dnet=self.gen_model(self.gen_model_dir,"./result")
        cnet=self.classify_model(10,self.classify_model_dir)
            
        region = self.org_img[:, :224, :224].clone().detach()
        region.requires_grad = True

        lossfunc = nn.MSELoss()
        optimizer = optim.Adam([region], lr=self.lr)
        for i in range(self.max_iteration):

            denoise_image = dnet(self.org_img, "org_img_label")
            region=denoise_image[:, :224, :224]
            classify_result = cnet(region)
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

            self.org_img[:, :224, :224] = region.detach()

        print("攻击失败")
        return self.org_img

def test_gen_net(args):
    image_path=args.image_path
    test_image=Image.open(image_path)
    test_image_tensor = F.to_tensor(test_image)
    test_image_tensor = test_image_tensor.unsqueeze(0)
    
    print(test_image_tensor.shape)
    dnet=denoise_net(args.gen_model_dir,args.result_dir)
    
    predict=dnet(test_image_tensor,"test_img_label")
    print(predict.shape)
    
    
def test_classify_net(args):
    image_path=args.image_path
    test_image=Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet-18 需要 224x224 的输入
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_image_tensor = transform(test_image)
    print(test_image_tensor.shape)
    cnet=classify_net(10,args.classify_model_dir)
    result=cnet(test_image_tensor)
    print("cnet: ",result)
    print("cnet argmax: ",torch.argmax(result))
    
def test_gradient_attack(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=2, type=int, choices=[0,1,2])
    parser.add_argument("--gen_model_dir", default=None, type=str)
    parser.add_argument("--image_path", default=None, type=str)
    parser.add_argument("--result_dir", default=None, type=str)
    parser.add_argument("--classify_model_dir", default=None, type=str)
    args = parser.parse_args()
    if args.mode==0:
        test_gen_net(args)
    elif args.mode==1:
        test_classify_net(args)
    elif args.mode==2:
        test_gradient_attack(args)
    else:
        print("mode is not correct")
    