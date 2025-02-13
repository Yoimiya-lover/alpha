import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from Alpha_blend_net import AlphaBlendNet

#加载PyTorch模型

model = AlphaBlendNet()
#切换到推理模式（去掉dropout，BN，训练模式
model.eval()

dummy_fg = torch.randn(1024,1024)
dummy_bg = torch.randn(1024,1024)
dummy_alpha = torch.randn(1,1024,1024)

#导出onnx
torch.onnx.export(model,
                  (dummy_fg,dummy_bg,dummy_alpha),
                  "alpha_blend.onnx",
                  input_names=["fg","bg","alpha"],
                  output_names=["output"],
                  export_params=True,
                  opset_version=11)

print("export onnx success!")

