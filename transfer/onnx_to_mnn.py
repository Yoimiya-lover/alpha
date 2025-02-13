import subprocess
import os,sys
import argparse


def convert_onnx_to_mnn(onnx_path, mnn_path):
    if len(sys.argv) != 3:
        print("Usage: python onnx_to_mnn.py <onnx_path> <mnn_path>")
        sys.exit(1)

    MNN_CONVERT_PATH = "/home/pytorch_practice/homework2/MNN/build/MNNConvert"  # 指定 MNNConvert 绝对路径
    onnx_path = sys.argv[1] #第一个参数是onnx文件路径
    mnn_path = sys.argv[2]  #第二个参数是mnn文件路径
    command = [
        MNN_CONVERT_PATH,  # MNNConvert 可执行文件路径
        "-f", "ONNX",  # 指定输入格式为 ONNX
        "--modelFile", onnx_path,  # ONNX 文件路径
        "--MNNModel", mnn_path,  # 目标 MNN 模型路径
        "--bizCode", "MNN"  # 业务代码，通常填 "MNN"
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)  # 打印转换输出日志
    print(result.stderr)  # 打印错误信息（如果有）


if __name__ == "__main__":
    # 调用转换
    convert_onnx_to_mnn("model.onnx", "model.mnn")
