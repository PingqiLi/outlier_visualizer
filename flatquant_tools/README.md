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

只需指定 Dump 目录：

```bash
python3 visualize_flatquant.py --dump_dir /path/to/dump/layer_X/npu_kronecker_quant/
```

脚本会自动检测：
1.  `input_0.pth`: 必须存在。
2.  `output_0.pth`: 如果存在，直接加载并解包可视化。
3.  如果 `output_0.pth` 不存在，尝试加载 `input_1.pth` 和 `input_2.pth` 并调用 NPU 算子计算。

### 3. 输出

脚本将在 `./flatquant_plots` (默认) 或 `--output_dir` 指定的目录中生成图表。

*   **3D 对比图**: 并排显示的 3D 曲面图，展示变换前后的激活值幅度 (全分辨率，无下采样)。
    *   左图: 原始输入 (BF16)
    *   右图: FlatQuant 输出 (Int4 Levels)
