import torch
import os
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
fused2 = load(
    "fused2",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused.cu"),
    ],
)
# CUDA実装のインポート
import fused  # コンパイルされたCUDA拡張

# サンプルデータ
input = torch.randn(4, 8, 16, 16, device='cuda')
bias = torch.randn(8, device='cuda')
refer = torch.randn(4, 8, 16, 16, device='cuda')
act = 3
grad = 0
alpha = 0.2
scale = 1.0

# Python実装
output_py = fused2.fused_bias_act(input, bias, refer, act, grad, alpha, scale)

# CUDA実装
output_cuda = fused.fused_bias_act(input, bias, refer, act, grad, alpha, scale)

# 比較
print(torch.allclose(output_py, output_cuda, atol=1e-6))


import torch
import numpy as np
from upfirdn2d_pytorch import upfirdn2d  # Python版の関数
import upfirdn2d_op  # CUDA版の関数

# テスト関数
def test_upfirdn2d():
    # ランダムな入力テンソルとカーネルを生成
    batch_size, height, width, channels = 2, 16, 16, 3
    kernel_size = 4
    
    input_tensor = torch.randn(batch_size, height, width, channels, device='cuda', dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    
    # パラメータ
    up_x, up_y = 2, 2
    down_x, down_y = 2, 2
    pad_x0, pad_x1, pad_y0, pad_y1 = 1, 1, 1, 1

    # CUDAカーネル実装を呼び出し
    output_cuda = upfirdn2d_op.upfirdn2d(input_tensor, kernel_tensor, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)

    # PyTorch実装を呼び出し
    output_pytorch = upfirdn2d(input_tensor, kernel_tensor, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
    

    # 結果の比較
    max_diff = torch.max(torch.abs(output_cuda - output_pytorch))
    #print(f"Max difference between CUDA and PyTorch implementations: {max_diff.item()}")
    
    print(torch.allclose(output_cuda, output_pytorch, atol=1e-6))

# テスト実行
test_upfirdn2d()
