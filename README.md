# Qwen3-MoE Ascend NPU 激活值 Dump 指南

本文档详细说明如何在 Ascend NPU 上使用 vLLM 对 Qwen3-MoE 模型进行激活值 Dump。

## 0. 环境准备 (安装 msit_llm)

如果你的环境中没有安装 `msit_llm`，请按照以下步骤安装：

1.  **进入 msit 源码目录**:
    ```bash
    git clone https://gitcode.com/Ascend/msit.git
    cd msit/msit
    ```

2.  **安装 msit 基础包**:
    ```bash
    pip install .
    ```

3.  **安装 llm 组件**:
    ```bash
    msit install llm
    ```

4.  **验证安装**:
    ```bash
    msit check llm
    ```

## 1. 代码修改

你需要修改 `vllm_ascend/worker/worker_v1.py` 文件，在 `compile_or_warm_up_model` 方法的末尾注入 `msit_llm` 的 Dump 钩子。
**注意：** 之前建议修改 `model_runner_v1.py`，但发现 vLLM 启动时会进行 Warmup 推理，导致提前触发 Dump。修改 `worker_v1.py` 可以确保在 Warmup 之后才注入钩子。

**文件路径**: `/Users/patrick/Projects/vllm-ascend/vllm_ascend/worker/worker_v1.py`

**修改内容**:
在 `compile_or_warm_up_model` 方法的末尾（`NPUPlatform.seed_everything` 之后），添加以下代码：

```python
        # Inject msit_llm dump hook AFTER warmup to avoid capturing warmup data
        try:
            from msit_llm import DumpConfig, register_hook
            from vllm_ascend.common.log import logger
            # 配置 Dump 参数
            # token_range=list(range(2000)): dump 前 2000 个 step
            # layer_name: 指定 dump 的层。
            # 注意：msit_llm 仅支持 '*' 通配符，不支持复杂的正则（如 [0-4]）。
            # 'root' 是 msit_llm 硬编码的顶层名称，因此必须以 'root' 开头。
            # 配置为 'root.model.layers.*' 将 dump 所有层，你需要 dump 后手动筛选前 5 层的数据。
            dump_config = DumpConfig(
                dump_path='./dump_data',
                token_range=list(range(2000)),
                layer_name='root.model.layers.*' 
            )
            # self.model_runner.model is the actual model instance
            register_hook(self.model_runner.model, dump_config)
            logger.info("Injected msit_llm dump hook for Qwen3-MoE (after warmup).")
        except ImportError:
            logger.warning("msit_llm not found, skipping dump hook injection.")
        except Exception as e:
            logger.warning(f"Failed to inject dump hook: {e}")
```

## 2. 启动服务

使用你提供的命令启动 vLLM 服务：

```bash
vllm serve /workspace/weights/Qwen3-30B \
    --max-model-len 2000 \
    --port 8017 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enforce-eager
```

*注意：*
1.  请确保 `--tensor-parallel-size 4` 与你的 NPU 卡数一致。
2.  **强烈建议添加 `--enforce-eager` 参数**。`msit_llm` 基于 PyTorch Hook 实现，在 Eager 模式下工作最稳定。如果不加此参数，vLLM 可能会使用 Graph 模式 (CUDAGraph/ACLGraph)，导致部分中间层 Hook 失效或无法捕获。

## 3. 发送请求进行 dump_request.py

使用提供的 Python 脚本 `dump_request.py` 发送请求。需要准备一个文本文件，脚本会自动截断到 2000 tokens。

**运行脚本**:
```bash
python3 dump_request.py --text_file your_text.txt --model_path /path/to/model
```

### 3.1 原始 Dump 目录结构
脚本执行完成后，会在 `./dump_data` 生成如下结构（未拼接）：
```
dump_data/
└── msit_dump_{PID}/                # PID 为 vLLM 进程 ID
    └── torch_tensors/
        └── npu{ID}_{PID}/          # npu{ID} 为卡号，如 npu0, npu1
            ├── 0/                  # Token ID (Step ID)
            │   ├── root.model.layers.0.self_attn.qkv_proj/
            │   │   ├── input.pth   # 输入 Tensor (BF16)
            │   │   └── output.pth  # 输出 Tensor (BF16)
            │   ├── root.model.layers.0.self_attn.o_proj/
            │   │   ...
            │   └── ...
            ├── 1/
            └── ...
```

## 4. 激活值拼接 (Stitching)

### 4.1 为什么需要拼接？
vLLM 使用 Tensor Parallel (TP) 时，模型被切分到多张 NPY 卡上。`msit_llm` dump 下来的是单卡视角的数据。
以 **Qwen3-MoE** 为例，我们需要根据切分策略还原完整的 Tensor：

*   **Column Parallel (列切分)**:
    *   **例子**: `qkv_proj` (Attention 输入), `gate_up_proj` (MLP 输入)。
    *   **现象**: 每张卡只计算了 Hidden Size 的一部分（例如总共 2048 维，4张卡每张负责 512 维）。
    *   **处理**: 需要将 4 张卡的数据在 **最后一维 (Dim -1)** 进行拼接 (Concatenate)。

*   **Row Parallel (行切分)**:
    *   **例子**: `o_proj` (Attention 输出), `down_proj` (MLP 输出)。
    *   **现象**: 每张卡计算部分结果后，会在内部进行 `All-Reduce` (求和)。
    *   **处理**: 因此，Dump 下来的结果在所有卡上是**完全相同**的（理论上）。我们不需要拼接，**任取一张卡**的数据即可。

*   **Fused MoE (混合专家)**:
    *   **例子**: `mlp.experts`。
    *   **现象**: MoE 算子内部处理了 Expert 分发和聚合，最终输出通常是完整的或者已经 Reduce 过的。
    *   **处理**: 同 Row Parallel，**不需要拼接**，任取一份。

### 4.2 执行拼接脚本
使用 `stitch_dump.py` 自动处理上述逻辑。

```bash
python3 stitch_dump.py --base_dir ./dump_data/msit_dump_{PID}/torch_tensors --output_dir ./stitched_npy --workers 8
```

### 4.3 拼接后的目录结构
脚本执行完成后，会在 `--output_dir` 生成如下结构：
```
stitched_npy/
├── layers.0.self_attn.qkv_proj/    # Column Parallel (已拼接)
│   ├── input.npy                   # [Seq_Len, Hidden_Size]
│   └── output.npy                  # [Seq_Len, Hidden_Size * 3] (Q+K+V)
├── layers.0.self_attn.o_proj/      # Row Parallel (单卡副本)
│   ├── input.npy
│   └── output.npy
├── layers.0.mlp.gate_up_proj/      # Column Parallel (已拼接)
│   ├── input.npy
│   └── output.npy                  # [Seq_Len, Intermediate * 2] (Gate+Up)
├── layers.0.mlp.down_proj/         # Row Parallel (单卡副本)
│   └── ...
└── ...
```

## 5. 可视化 (Visualization)

使用 `visualize_outliers.py` 生成高分辨率 3D 激活图。

### 5.1 功能特点
*   **自动拆分 QKV**: 自动识别 `qkv_proj` 并将其拆分为 Q, K, V 三个独立的投影进行绘图。
*   **自动拆分 MLP**: 自动识别 `gate_up_proj` 并拆分为 Gate, Up。
*   **MoE 支持**: 支持 MoE 层的输入输出可视化。
*   **3D 曲面图**: 生成 `(Token, Channel, Value)` 的 3D 交互式/静态图，直观展示 Outlier。

### 5.2 运行命令
```bash
python3 visualize_outliers.py \
    --data_dir ./stitched_npy \
    --layer_pattern layers.10 \
    --io_type output \
    --model_path /path/to/Qwen3-30B \
    --workers 16
```
*   `--model_path`: **(推荐)** 指定模型路径，脚本会自动读取 `config.json` 来获取 Head Dim 等信息，用于正确的 QKV 拆分。
*   `--layer_pattern`: 过滤层名，支持正则。例如 `layers.10` 只看第 10 层。
*   `--io_type`: `input` 或 `output`。

### 5.3 进阶：Dump 算子级数据 (如 FlatQuant)

如果你需要分析 FlatQuant 的 `npu_kronecker_quant` 等Layer中更细粒度的算子，需要进行以下修改：

1.  **修改 `msit_llm` 源码 (`hook_ops.py`)**:
    由于 `npu_kronecker_quant` 接口默认不在 `msit_llm` 的 Hook 列表中，需要手动添加。
    
    **文件路径**: `.../msit/msit/components/llm/msit_llm/dump/torch_dump/hook_ops.py` (具体路径取决于你的安装位置)
    
    **修改内容**:
    找到 `add_torch_npu_ops` 函数中的 `torch_npu_hooks` 列表，添加 `"npu_kronecker_quant"`：
    ```python
    def add_torch_npu_ops():
        # ...
        torch_npu_hooks = [
            "fast_gelu",
            "npu_mish",
            # ...
            "npu_all_gather_base_mm",
            "npu_kronecker_quant",  # <--- 添加这一行
        ]
        # ...
    ```

2.  **修改 `worker_v1.py`**:
    配置 `DumpConfig` 开启 API 模式。
    ```python
    import torch_npu
    dump_config = DumpConfig(
        ...,
        mode=["api"], # 或 ["module", "api"]
        api_list=[torch_npu.npu_kronecker_quant], # 白名单
        layer_name='.*'
    )
    ```

3.  **结果**:
    数据会保存在 `.../root.model.layers.*.mlp.experts.npu_kronecker_quant/` 目录下。
    *   `input_0.pth`: 原始激活值 (BF16)
    *   `input_1.pth`: Left Transform Matrix
    *   `input_2.pth`: Right Transform Matrix
