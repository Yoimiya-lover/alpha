import numpy as np
import MNN
import cv2

# 1️⃣ **加载 MNN 模型**
model_path = "alpha_blend.mnn"  # 你的 MNN 模型文件
interpreter = MNN.Interpreter(model_path)
session = interpreter.createSession()
input_tensor = interpreter.getSessionInput(session)

print("✅ MNN 模型加载成功")

# 2️⃣ **读取输入图片**
fg = cv2.imread("fg.jpg", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0  # 归一化
bg = cv2.imread("bg.jpg", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
alpha = cv2.imread("alpha.jpg", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

# 确保大小一致
H, W = 1024, 1024
fg = cv2.resize(fg, (W, H))
bg = cv2.resize(bg, (W, H))
alpha = cv2.resize(alpha, (W, H))

# 归一化 alpha 并调整 shape
if len(alpha.shape) == 3:
    alpha = alpha[:, :, 0]  # 只取 alpha 通道
alpha = np.expand_dims(alpha, axis=2)  # (H, W) -> (H, W, 1)

# 3️⃣ **准备输入数据**
input_data = np.concatenate([fg, bg, alpha], axis=2)  # (1024, 1024, 7) -> (H, W, 7)
input_data = np.transpose(input_data, (2, 0, 1))  # 转换为 (C, H, W)
input_data = input_data.astype(np.float32)

# 4️⃣ **将数据传入 MNN**
input_tensor.copyFrom(input_data)  # 设置输入数据
interpreter.runSession(session)    # 运行推理

# 5️⃣ **获取输出**
output_tensor = interpreter.getSessionOutput(session)
output_data = np.array(output_tensor.getData())  # 获取数据

# 6️⃣ **调整输出形状**
print("原始输出数据 shape:", output_data.shape)
output_data = output_data.reshape((3, 1024, 1024))  # 变成 (3, H, W)
output_data = np.transpose(output_data, (1, 2, 0))  # 变回 (H, W, 3)

# 7️⃣ **保存输出**
output = (output_data * 255).astype(np.uint8)  # 反归一化
cv2.imwrite("output.jpg", output)
print("✅ 推理完成，已保存 output.jpg")
