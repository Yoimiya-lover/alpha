import cv2
import numpy as np
import MNN

def load_and_preprocess(image_path, target_size):
    """ 读取、调整大小、归一化并转换为 NCHW 格式 """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 读取图像
    img = cv2.resize(img, (target_size, target_size))  # 调整大小
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # 归一化

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)  # 增加 Batch 维度
    
    print(f"Loaded image {image_path} with shape: {img.shape}")
    return img

def numpy_to_mnn_tensor(numpy_data, tensor):
    """ 修正后的 NumPy -> MNN Tensor 转换 """
    numpy_data = numpy_data.astype(np.float32)  # 确保是 float32
    tensor_shape = tensor.getShape()  # ✅ 使用 getShape()
    
    print(f"Expecting shape: {tensor_shape}, but got: {numpy_data.shape}")  # Debug
    
    if list(numpy_data.shape) != tensor_shape:
        numpy_data = numpy_data.reshape(tensor_shape)  # 调整形状
    
    mnn_tensor = MNN.Tensor(tensor_shape, MNN.Halide_Type_Float, numpy_data, MNN.Tensor_DimensionType_Caffe)
    tensor.copyFrom(mnn_tensor)

def run_inference(model_path, fg_path, bg_path, alpha_path, input_size):
    """ 运行模型推理 """
    # 加载 MNN 模型
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()

    # 读取图像
    fg = load_and_preprocess(fg_path, input_size)
    bg = load_and_preprocess(bg_path, input_size)
    alpha = load_and_preprocess(alpha_path, input_size)

    # 处理 Alpha（检查是否单通道）
    print(f"Alpha shape before: {alpha.shape}")
    if alpha.shape[1] != 1:  
        alpha = np.mean(alpha, axis=1, keepdims=True)  # 转换为单通道
    print(f"Alpha shape after: {alpha.shape}")

    # 获取输入张量（检查输入名称）
    inputs = interpreter.getSessionInputAll(session)
    print("模型输入名称:", inputs.keys())  # 打印输入名称
    
    if "fg" not in inputs or "bg" not in inputs or "alpha" not in inputs:
        raise ValueError("模型输入名称不匹配，请检查 MNN 模型的输入名称！")
    
    input_fg = inputs["fg"]
    input_bg = inputs["bg"]
    input_alpha = inputs["alpha"]

    print("开始复制数据")
    # 复制数据
    numpy_to_mnn_tensor(fg, input_fg)
    numpy_to_mnn_tensor(bg, input_bg)
    numpy_to_mnn_tensor(alpha, input_alpha)

    # 运行模型
    interpreter.runSession(session)

    # 获取输出
    output_tensor = interpreter.getSessionOutput(session)
    output_data = np.array(output_tensor.getData())

    print("推理结果:", output_data.shape)
    return output_data

if __name__ == "__main__":
    # 设定输入尺寸
    INPUT_SIZE = 1024

    # 运行推理
    output_data = run_inference("alpha_blend.mnn", "fg.jpg", "bg.jpg", "alpha.jpg", INPUT_SIZE)

    # 进一步处理输出结果，例如保存图像
    output_image = (output_data.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imwrite("output.jpg", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
