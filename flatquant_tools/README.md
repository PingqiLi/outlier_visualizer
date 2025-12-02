# FlatQuant 可视化工具

本目录包含用于模拟和可视化 FlatQuant (Kronecker 分解) 对激活值离群点影响的工具。

## 脚本

*   `visualize_flatquant.py`: 主要的可视化脚本。
*   `test_flatquant_sim.py`: 用于验证模拟逻辑的测试脚本 (需要 `torch`)。

## 用法

### 前置要求

*   Python 3
*   PyTorch
*   Matplotlib
*   NumPy

### 1. 准备 Dump 数据

确保你已经使用 `msit_llm` Dump 了 `npu_kronecker_quant` 算子的数据。
Dump 目录应包含：
*   `input_0.pth`: 原始激活值 (BF16/FP16)
*   `input_1.pth`: 左变换矩阵 (Left Transformation Matrix)
*   `input_2.pth`: 右变换矩阵 (Right Transformation Matrix)

### 2. 运行可视化

你可以通过指定 Dump 目录来运行脚本：

```bash
python3 visualize_flatquant.py --dump_dir /path/to/dump/layer_X/npu_kronecker_quant/
```

或者指定单独的文件：

```bash
python3 visualize_flatquant.py \
    --input_act /path/to/input_0.pth \
    --left_mat /path/to/input_1.pth \
    --right_mat /path/to/input_2.pth
```

### 3. 输出

脚本将在 `./flatquant_plots` (默认) 或 `--output_dir` 指定的目录中生成图表。

*   **3D 对比图**: 并排显示的 3D 曲面图，展示变换前后的激活值幅度。
*   **直方图**: 对数刻度的直方图，对比数值分布，突显离群点的抑制效果。
