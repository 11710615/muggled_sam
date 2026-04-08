# 导出脚本说明（ONNX / TensorRT）

本目录用于放置本地导出相关脚本与导出产物（例如：SAMv3 组件导出 ONNX、以及用 `trtexec` 构建 TensorRT engine），并记录**导出前的必要源码改动**，以便复现与排查导出差异。

## 导出模块
- **image_projection_v2**
- **mask_decocder**
- **image_SAMV3MemoryImageFusion**
- **memory_encoder**

## 目录内容

- `sam1_onnx_export.py`
  - 用于导出 ONNX。
- `trt_export.sh`
  - 调用 `trtexec` 将 `exported_model/onnx/*.onnx` 转为 TensorRT engine（目前示例输出到 `exported_model/trt/fp16/`）。
- `compare_onnx.py`
  - 用于对比两份 ONNX 是否一致（结构/权重/可选 runtime 数值一致性）。
- `exported_model/`
  - 导出产物目录（示例：`exported_model/onnx/`、`exported_model/trt/`）。

## 导出前源码改动（重要）

以下改动是为 **ONNX/TensorRT 导出兼容性** 做的本地调整。若你希望“再次导出得到的文件”和“原始导出文件”保持一致，务必保证这些改动一致（或至少明确记录差异）。

### 1) 禁用复数路径（避免部分后端不兼容）

- **文件**：`muggled_sam/v3_sam/components/memory_image_fusion_attention.py`
- **改动**：将构造参数默认值 `use_complex_numbers` 改为 `False`（约第 51 行附近）
- **目的**：避免走复数（complex number）相关实现路径，部分导出/推理后端对复数支持不完整时可能失败或产生不可预期差异。

### 2) 用 `torch.where` 替代张量分支（更利于 TRT 导出）

- **文件**：`muggled_sam/v3_sam/components/memory_encoder_components.py`
- **改动位置**：约第 111-117 行
- **改动**：将基于 Python `if/else` 的张量分支逻辑改为 `torch.where`（并对 `is_prompt_encoding` 做 broadcast 以匹配 `hires_mask` 形状）
- **目的**：TensorRT/ONNX 导出时对 Python 控制流与布尔分支的支持有限，使用 `torch.where` 能让图更“静态化”，更容易被导出与解析。


## 导出 ONNX

以脚本默认输出目录为例（详见 `sam1_onnx_export.py` 内的 `onnx_output_dir`）：

```bash
python3 export_scripts/sam1_onnx_export.py
```

导出后的 ONNX（示例）会在：
 
- `export_scripts/exported_model/onnx/`

## 构建 TensorRT engine

脚本会读取 ONNX 并输出 engine 到（默认）：

- `export_scripts/exported_model/trt/fp16/`

执行：

```bash
bash export_scripts/trt_export.sh
```

## 对比两份 ONNX 是否一致

### 结构 + 权重一致性（推荐）

```bash
python3 export_scripts/compare_onnx.py a.onnx b.onnx
```

### 额外做 runtime 数值一致性（可选）

需要 `onnxruntime`：

```bash
pip install onnxruntime
python3 export_scripts/compare_onnx.py --runtime a.onnx b.onnx
```

