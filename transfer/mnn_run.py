import MNN
import numpy as np
import cv2
import torch

def read_img_mnn(fg_path, bg_path, alpha_path):
    """ 读取图像并转换为 MNN 需要的格式 """
    fg = cv2.imread(fg_path).astype(np.float32) / 255.0
    bg = cv2.imread(bg_path).astype(np.float32) / 255.0
    alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # 调整大小
    fg = cv2.resize(fg, (1024, 1024))
    bg = cv2.resize(bg, (1024, 1024))
    alpha = cv2.resize(alpha, (1024, 1024))

    # 变换维度 (H, W, C) -> (1, C, H, W)
    fg = np.transpose(fg, (2, 0, 1))[None, :, :, :]
    bg = np.transpose(bg, (2, 0, 1))[None, :, :, :]
    alpha = alpha[None, None, :, :]  # 增加批次和通道维度

    return fg, bg, alpha

# 加载 MNN 模型
interpreter = MNN.Interpreter("alpha_blend.mnn")
session = interpreter.createSession()
input_fg = interpreter.getSessionInput(session, "fg")
input_bg = interpreter.getSessionInput(session, "bg")
input_alpha = interpreter.getSessionInput(session, "alpha")

# 读取图片
fg, bg, alpha = read_img_mnn("fg.jpg", "bg.jpg", "alpha.jpg")
print("读取成功，开始推理")
print("把 numpy 转换成 MNN Tensor")
# 把 numpy 转换成 MNN Tensor
fg_tensor = MNN.Tensor((1, 3, 1024, 1024), MNN.Halide_Type_Float, fg, MNN.Tensor_DimensionType_Caffe)
bg_tensor = MNN.Tensor((1, 3, 1024, 1024), MNN.Halide_Type_Float, bg, MNN.Tensor_DimensionType_Caffe)
alpha_tensor = MNN.Tensor((1, 1, 1024, 1024), MNN.Halide_Type_Float, alpha, MNN.Tensor_DimensionType_Caffe)
print("转换成功")

print("开始把数据传入 MNN")

print("fg input shape:", fg_tensor.getShape())  # 打印 MNN 输入形状
print("bg input shape:", bg_tensor.getShape())
print("alpha input shape:", alpha_tensor.getShape())
print(fg.dtype, bg.dtype, alpha.dtype)  # 确保它们都是 float32


# 把数据传入 MNN
input_fg.copyFrom(fg_tensor)
input_bg.copyFrom(bg_tensor)
input_alpha.copyFrom(alpha_tensor)
print("数据传入成功")
# 运行推理
interpreter.runSession(session)

# 获取输出
output_tensor = interpreter.getSessionOutput(session, "output")
output_np = np.array(output_tensor.getData(), dtype=np.float32).reshape(1, 3, 1024, 1024)

# 变换维度回 (H, W, C)
output_np = np.transpose(output_np[0], (1, 2, 0)) * 255
output_np = np.clip(output_np, 0, 255).astype(np.uint8)

# 保存 MNN 推理结果
cv2.imwrite("mnn_output.jpg", output_np)
print("MNN 推理完成，结果已保存到 mnn_output.jpg")
