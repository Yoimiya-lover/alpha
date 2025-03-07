import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

class AlphaBlendNet(nn.Module):
    def __init__(self,kernel_size=5):
        super(AlphaBlendNet,self).__init__()#继承父类，是之可以成为一个PyTorch可训练模型
        #固定均值滤波器
        #创建二维卷积层，表示输出和输入都是1，kernerl_size为5，表示卷积核大小
        #padding=kernel_size//2：保证卷积后图像大小不变（填充 kernel_size//2 像素）。
        self.conv = nn.Conv2d(1,1,kernel_size,padding=kernel_size//2,bias = False)
        #(1, 1, 5, 5) 代表 1 个输入通道、1 个输出通道、5×5 卷积核。，所有元素的值除以kernerl_size的平方，滤波器总和为一，标准均值滤波
        weight =torch.ones(1,1,kernel_size,kernel_size) / (kernel_size ** 2)
        #手动设置滤波器权重，require_grad=False,表示不进行训练
        self.conv.weight = nn.Parameter(weight,requires_grad = False)

    def forward(self,fg,bg,alpha):
        #归一化alpha到[0,1]
        alpha = alpha.float() / 255.0
        #alpha只有三个维度，unsqueeze（1）增加一个维度，且全为1，才能输入conv2d
        #self.conv是均值滤波
        alpha = self.conv(alpha.unsqueeze(1))
        #去掉一个维度
        alpha = alpha.squeeze(1)

        #计算加权融合
        blended = fg * alpha[:,None,:,:] + bg * (1 - alpha[:,None,:,:])
        return blended
    

#读取图片
def read_img(fg_path,bg_path,alpha_path):
    fg = cv2.imread(fg_path).astype(np.float32) / 255.0  #归一化
    bg = cv2.imread(bg_path).astype(np.float32) / 255.0  #归一化
    alpha = cv2.imread(alpha_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #调整大小
    fg = cv2.resize(fg,(1024,1024))
    bg = cv2.resize(bg,(1024,1024))
    alpha = cv2.resize(alpha, (1024, 1024))  # 确保 alpha 与 fg/bg 大
    print("图像大小:", fg.shape, bg.shape, alpha.shape)

    #转化为PyTorch Tensor
    fg = torch.from_numpy(fg).permute(2,0,1).unsqueeze(0)
    bg = torch.from_numpy(bg).permute(2,0,1).unsqueeze(0)
    alpha = torch.from_numpy(alpha).unsqueeze(0)
    return fg,bg,alpha

#创建一个alpha
def create_alpha(size=1024):
    """ 生成一个有渐变效果的 alpha 蒙版 """
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # 计算径向距离
    r = np.sqrt(x**2 + y**2)
    
    # 归一化，中心区域 alpha=255，边缘变为 0
    alpha = np.clip(1 - r, 0, 1) * 255

    # 叠加不规则形状（椭圆）
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size//2, size//2), (size//3, size//4), 30, 0, 360, 255, -1)
    
    # 让 `alpha` 受 `mask` 影响
    alpha = np.maximum(alpha, mask)

    # 转换成 uint8
    return alpha.astype(np.uint8)

if __name__ == "__main__":
    # 生成并保存新的 alpha 蒙版
    alpha = create_alpha(1024)
    cv2.imwrite("alpha.jpg", alpha)

    #测试
    fg,bg,alpha = read_img("fg.jpg","bg.jpg","alpha.jpg")
    model = AlphaBlendNet()
    #等价于output = model.forward(fg, bg, alpha)
    #PyTorch 中，推荐使用 model(fg, bg, alpha)，因为 nn.Module 内部会自动处理 forward()，
    # 同时可以结合 hooks 等高级功能。
    output = model(fg,bg,alpha)
    #squeeze(0)去掉batch维度
    #permute(1, 2, 0)交换维度
    # detach() 让 output 脱离计算图，变成普通的 Tensor，不再追踪梯度。
    #在推理阶段 (inference) 必须使用，否则 .numpy() 可能会报错。
    #output 变成 NumPy 数组，方便后续处理：
    output = (output.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)

    #写出
    cv2.imwrite("output.jpg", output)

    # 显示图像
    # plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    # plt.title("Blended Image")
    # plt.axis("off")
    # plt.show()
