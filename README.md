# Qwen3-MoE Ascend NPU 激活值 Dump 指南

本文档详细说明如何在 Ascend NPU 上使用 vLLM 对 Qwen3-MoE 模型进行激活值 Dump。

## 0. 环境准备 (安装 msit_llm)

如果你的环境中没有安装 `msit_llm`，请按照以下步骤安装：

1.  **进入 msit 源码目录**:
    ```bash
    cd /Users/patrick/Projects/vllm-ascend/msit/msit
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
    python3 -c "import msit_llm; print('msit_llm installed successfully')"
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

## 3. 发送请求进行 Dump

使用提供的 Python 脚本 `dump_request.py` 发送请求。该脚本配置了 `ignore_eos=True` 和 `min_tokens=2000`，并使用 `temperature=0.0` (Greedy Decoding)，确保生成长度固定为 2000。

**运行脚本**:
```bash
python3 dump_request.py
```

**脚本内容关键点**:
```python
    payload = {
        "model": "/workspace/weights/Qwen3-30B",
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "max_tokens": 2000,
        "min_tokens": 2000,     # 强制生成至少2000 token
        "ignore_eos": True,     # 忽略EOS
        "temperature": 0.0,     # Greedy decoding
    }
```

## 4. 关于 TP=4 激活值拼接 (Stitching)

你提到的问题：**"TP=4, 4张npu dump的话，我是不是要按照rank把激活按照channel维度拼接起来呢？"**

**答案：取决于具体的层类型。**

vLLM 使用 Tensor Parallel (TP) 时，不同类型的层有不同的切分策略。`msit_llm` dump 下来的是单卡上的数据。

### 需要拼接的层 (Column Parallel)
以下层的输出在 TP 模式下是被切分的，你需要将 4 张卡的 dump 数据在 **最后一维 (Channel/Hidden Size)** 进行拼接 (Concatenate)：

*   **`qkv_proj`**: Attention 的 QKV 投影。
*   **`gate_up_proj`**: MLP 的 Gate 和 Up 投影。

### 不需要拼接的层 (Row Parallel / Replicated)
以下层的输出在 TP 模式下通常是已经经过 All-Reduce (求和) 的，或者是复制的，因此 4 张卡的数据是**完全相同**的（理论上），你只需要取其中任意一张卡的数据即可，**不需要拼接**：

*   **`o_proj`**: Attention 的输出投影 (Row Parallel, 默认 `reduce_results=True`)。
*   **`down_proj`**: MLP 的输出投影 (Row Parallel, 默认 `reduce_results=True`)。
*   **`experts` (MoE 输出)**: MoE 模块的输出通常也是经过 Reduce 的，或者是完整的。

### 总结表格

| 模块 | 层名称 (示例) | TP 切分方式 | Dump 后处理 |
| :--- | :--- | :--- | :--- |
| Attention | `self_attn.qkv_proj` | Column Parallel | **需要拼接** (Dim -1) |
| Attention | `self_attn.o_proj` | Row Parallel (Reduced) | **不需要拼接** (任取一份) |
| MLP (MoE) | `mlp.gate_up_proj` | Column Parallel | **需要拼接** (Dim -1) |
| MLP (MoE) | `mlp.down_proj` | Row Parallel (Reduced) | **不需要拼接** (任取一份) |
| MoE Block | `mlp.experts` | Reduced | **不需要拼接** (任取一份) |

## 5. 结果检查

Dump 完成后，数据会保存在 `./dump_data` 目录下（根据 `DumpConfig` 配置）。

### 目录结构说明
目录结构如下：
```
dump_data/
└── msit_dump_{PID}/                # PID 为 vLLM 进程 ID
    └── torch_tensors/
        └── npu{ID}_{PID}/          # npu{ID} 为卡号，如 npu0, npu1
            ├── 0/                  # Token ID (Step ID)
            │   ├── root.model.layers.0.self_attn.qkv_proj/
            │   │   ├── input.pth   # 输入 Tensor
            │   │   └── output.pth  # 输出 Tensor
            │   ├── root.model.layers.0.self_attn.o_proj/
            │   │   ...
            │   └── ...
            ├── 1/
            └── ...
```

### 常见问题
**Q: 日志中出现 `Unrecognized data type <class 'NoneType'>` 警告？**
A: **这是正常的，请忽略。**
这是因为某些层（如 Attention 或 MLP）的 forward 函数中包含可选参数（如 `bias` 或 `residual`），当这些参数为 `None` 时，Hook 尝试捕获它们会触发此警告。这不会影响其他正常 Tensor 的 Dump。

**Q: TP=4 激活值拼接 (Stitching)**
A: 我为你提供了一个自动拼接脚本 `stitch_dump.py`。

**使用方法**:
```bash
python3 stitch_dump.py --base_dir ./dump_data/msit_dump_{PID}/torch_tensors --output_dir ./stitched_npy --workers 8
```

**功能说明**:
1.  **并行加速**: 支持多进程并行处理 (`--workers`)，大幅提升拼接速度。
2.  **自动合并**: 将所有 Token 的数据合并为一个 `.npy` 文件，方便整体分析。
3.  **智能拼接**: 自动处理 Prefill (Sequence) 和 Decode (Single Token) 的形状差异，将它们在时间维度上拼接。
    *   最终形状: `[Total_Tokens, Hidden_Dim]`。
    *   其中 `Total_Tokens` = `Prefill_Seq_Len` + `Decode_Steps`。
4.  **目录重构**: 只保留包含数据的层级目录，结构更清晰：
    ```
    stitched_npy/
    ├── layers.0.mlp.down_proj/
    │   ├── output.npy
    │   └── input.npy
    ├── layers.0.self_attn.qkv_proj/
    │   └── ...
    ```

**Q: 如何可视化 Outlier？**
A: 我为你提供了一个可视化脚本 `visualize_outliers.py`。

**功能**:
*   **层过滤**: 支持 `--layer_pattern` (如 `layers.10`)，会自动匹配该层下的所有子模块。
*   **QKV 拆分**: 支持 `--qkv_config` 参数，可以将 `qkv_proj` 自动拆分为 `q_proj`, `k_proj`, `v_proj`。
*   **MoE 支持**: 自动识别 MoE 聚合输出。
*   **读取合并数据**: 适配新的 `stitch_dump.py` 输出格式。

**使用方法**:
```bash
# 方法 1: 自动解析 Config (推荐)
# 指定模型权重目录，脚本会自动读取 config.json 获取 QKV 配置
python3 visualize_outliers.py \
    --data_dir ./stitched_npy \
    --layer_pattern layers.10 \
    --io_type output \
    --model_path /path/to/Qwen3-30B-A3B

# 方法 2: 手动指定 QKV 配置
# 假设配置: Heads=32, KV_Heads=4, Dim=128
python3 visualize_outliers.py \
    --data_dir ./stitched_npy \
    --layer_pattern layers.10 \
    --io_type output \
    --qkv_config 32,4,128
```

**关于 MoE MLP**:
Qwen3-MoE 的 MLP 层由 `gate` (Router) 和 `experts` (FusedMoE) 组成。
*   `mlp.experts`: 代表 MoE 模块的整体输出。由于使用了 Fused Kernel，我们无法直接 Dump 内部 128 个 Expert 的独立激活值，但通常分析 MoE 的整体输入/输出对于 Outlier 定位已经足够。
