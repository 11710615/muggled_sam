#!/bin/bash
set -euo pipefail
trap 'echo "❌ 导出失败，停止执行。失败位置: line ${LINENO}, command: ${BASH_COMMAND}" >&2' ERR

# 定义输出目录
OUTPUT_DIR="export_scripts/exported_model/trt/fp16"
mkdir -p "${OUTPUT_DIR}"
echo "输出目录已确认/创建：${OUTPUT_DIR}"

# ==================== 1. image_projection_v2 ====================
# trtexec \
# --onnx=export_scripts/exported_model/onnx/image_projection_v2.onnx \
# --saveEngine="${OUTPUT_DIR}/image_projection_v2.engine" \
# --verbose \
# --noTF32 \
# --fp16

# echo "✅ image_projection_v2 engine 导出完成: ${OUTPUT_DIR}/image_projection_v2.engine"

# ==================== 2. mask_decoder ====================
trtexec \
--onnx=export_scripts/exported_model/onnx/mask_decocder.onnx \
--saveEngine="${OUTPUT_DIR}/mask_decoder.engine" \
--minShapes=encoded_prompts_bnc:1x0x256,lowres_tokens:1x256x72x72,hires_tokens_x2:1x64x144x144,hires_tokens_x4:1x32x288x288,grid_positional_encoding:1x256x72x72 \
--optShapes=encoded_prompts_bnc:1x32x256,lowres_tokens:1x256x72x72,hires_tokens_x2:1x64x144x144,hires_tokens_x4:1x32x288x288,grid_positional_encoding:1x256x72x72  \
--maxShapes=encoded_prompts_bnc:1x100x256,lowres_tokens:1x256x72x72,hires_tokens_x2:1x64x144x144,hires_tokens_x4:1x32x288x288,grid_positional_encoding:1x256x72x72  \
--verbose \
--noTF32 \
--fp16

echo "✅ mask_decoder engine 导出完成: ${OUTPUT_DIR}/mask_decoder.engine"

# ==================== 3. image_SAMV3MemoryImageFusion ====================
# trtexec \
# --onnx=export_scripts/exported_model/onnx/image_SAMV3MemoryImageFusion.onnx \
# --saveEngine="${OUTPUT_DIR}/image_SAMV3MemoryImageFusion.engine" \
# --minShapes=flat_imgtokens_bnc:1x5184x256,memory_tokens:1x5184x64,memory_posenc:1x5184x64,num_ptr_tokens_tensor:1x1 \
# --optShapes=flat_imgtokens_bnc:1x5184x256,memory_tokens:1x36348x64,memory_posenc:1x36348x64,num_ptr_tokens_tensor:1x61 \
# --maxShapes=flat_imgtokens_bnc:1x5184x256,memory_tokens:1x51940x64,memory_posenc:1x51940x64,num_ptr_tokens_tensor:1x101 \
# --verbose \
# --noTF32 \
# --fp16

# echo "✅ image_SAMV3MemoryImageFusion engine 导出完成: ${OUTPUT_DIR}/image_SAMV3MemoryImageFusion.engine"

# ==================== 4. memory_encoder ====================
# trtexec \
# --onnx=export_scripts/exported_model/onnx/memory_encoder.onnx \
# --saveEngine="${OUTPUT_DIR}/memory_encoder.engine" \
# --verbose \
# --noTF32 \
# --fp16

# echo "✅ memory_encoder engine 导出完成: ${OUTPUT_DIR}/memory_encoder.engine"

echo -e "\n🎉 所有模型 engine 导出全部成功完成！"