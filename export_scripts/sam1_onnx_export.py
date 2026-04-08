try:
    import muggled_sam  
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to muggled_sam folder!")
import torch
import torch.nn as nn
from muggled_sam.make_sam import make_sam_from_state_dict
from torch import Tensor

"""image projection v2 部分导出"""   
class SAMV3ImageProjectionV2_wrapper(nn.Module):
    def __init__(self, sammodel):
        super().__init__()
        self.image_projection_v2 = sammodel.image_projection.v2_projection
    def forward(self, vb_trunk_feat):
        v2_tokens_x1, v2_tokens_x2, v2_tokens_x4= self.image_projection_v2(vb_trunk_feat)
        return v2_tokens_x1, v2_tokens_x2, v2_tokens_x4
   
def export_SAMV3ImageProjection_v2(SAMV3ImageProjectionV2_wrapper, output_path: str, device="cpu"):
    # 将图像投影模块移动到指定设备并设置为评估模式
    SAMV3ImageProjectionV2 = SAMV3ImageProjectionV2_wrapper.to(device).eval()
    # 创建用于 ONNX 导出的随机输入张量
    vb_trunk_feat = torch.randn([1,1024,72,72])
    # 执行 ONNX 导出，配置参数包括算子版本、动态轴及输入输出名称
    torch.onnx.export(
        SAMV3ImageProjectionV2,
        (vb_trunk_feat),
        f"{output_path}/image_projection_v2.onnx",
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=['vb_trunk_feat'],
        output_names=['v2_tokens_x1', 'v2_tokens_x2', 'v2_tokens_x4'],
        verify=True,
    )
    # 打印保存确认信息
    print(f"saved to: {output_path}/image_projection_v2.onnx")

"""mask decoder 部分导出"""
class SAMV3MaskDecoder_wrapper(nn.Module):
    def __init__(self, sammodel):
        super().__init__()
        self.mask_decoder = sammodel.mask_decoder
        self.maskhint_encoder = self.mask_decoder.maskhint_encoder
        self.cls_obj_token = self.mask_decoder.cls_obj_token
        self.cls_mask_tokens = self.mask_decoder.cls_mask_tokens
        self.cls_iou_token = self.mask_decoder.cls_iou_token
        self.transformer = self.mask_decoder.transformer
        self.maskgen = self.mask_decoder.maskgen
        self.iou_token_mlp = self.mask_decoder.iou_token_mlp
        self.objptrgen = self.mask_decoder.objptrgen
    def forward(
        self,
        lowres_tokens: Tensor,  #[1,256,72,72]
        hires_tokens_x2: Tensor,  #[1,64,144,144]
        hires_tokens_x4: Tensor,  #[1,32,288,288]
        encoded_prompts_bnc: Tensor,  # [1,n_point,256]
        grid_positional_encoding: Tensor,  #[1,256,72,72]
    ):
        # Prepare mask & image data for fusion
        # For clarity
        batch_size_prompts = encoded_prompts_bnc.shape[0]

        # Concatenate learned 'cls' tokens to prompts
        cls_tokens = torch.cat([self.cls_obj_token, self.cls_iou_token, self.cls_mask_tokens], dim=0)
        cls_tokens = cls_tokens.unsqueeze(0).expand(batch_size_prompts, -1, -1)
        num_cls_tokens = cls_tokens.shape[1]

        # Expand per-image data in batch direction to be per-mask, as well as the position encoding
        """去除mask_hint"""
        mask_hint = None
        img_tokens_bchw = self.maskhint_encoder(lowres_tokens, mask_hint)
        img_tokens_bchw = torch.repeat_interleave(img_tokens_bchw, batch_size_prompts, dim=0)
        img_posenc_bchw = torch.repeat_interleave(grid_positional_encoding, batch_size_prompts, dim=0)

        # Cross-encode image tokens with prompt tokens
        prompt_tokens = torch.cat((cls_tokens, encoded_prompts_bnc), dim=1)
        encoded_prompt_tokens, encoded_img_tokens = self.transformer(prompt_tokens, img_tokens_bchw, img_posenc_bchw)

        # Extract the (now-encoded) 'cls' tokens by undoing the earlier cls concatenation step
        encoded_cls_tokens = encoded_prompt_tokens[:, :num_cls_tokens, :]
        obj_token_out = encoded_cls_tokens[:, 0, :]
        iou_token_out = encoded_cls_tokens[:, 1, :]
        mask_tokens_out = encoded_cls_tokens[:, 2:, :]

        # Produce final output mask & quality predictions
        mask_preds = self.maskgen(encoded_img_tokens, hires_tokens_x2, hires_tokens_x4, mask_tokens_out)  #[1,4,288,288]
        iou_preds = self.iou_token_mlp(iou_token_out)  #[1,4]

        # Generate 'object pointer' output
        obj_score, obj_ptrs = self.objptrgen(obj_token_out, mask_tokens_out)  # [1,4,256], [1,1]

        return mask_preds, iou_preds, obj_ptrs, obj_score

def export_SAMV3MaskDecoder(sammodel, output_path: str, device="cpu"):
    mask_decoder = SAMV3MaskDecoder_wrapper(sammodel).to(device).eval()
    lowres_tokens = torch.randn(1,256,72,72).to(device)
    hires_tokens_x2 = torch.randn(1,64,144,144).to(device)
    hires_tokens_x4 = torch.randn(1,32,288,288).to(device)
    encoded_prompts_bnc = torch.randn(1,3,256).to(device)
    grid_positional_encoding =torch.randn(1,256,72,72).to(device)
    mask_preds, iou_preds, obj_ptrs, obj_score = mask_decoder(lowres_tokens, hires_tokens_x2, hires_tokens_x4, encoded_prompts_bnc, grid_positional_encoding)
    print(mask_preds.shape, iou_preds.shape, obj_ptrs.shape, obj_score.shape, "**********")
    torch.onnx.export(mask_decoder, 
                      (lowres_tokens, hires_tokens_x2, hires_tokens_x4, encoded_prompts_bnc, grid_positional_encoding), 
                      f"{output_path}/mask_decocder.onnx",
                      export_params=True,
                      opset_version=19,
                      do_constant_folding=True,
                      input_names=["lowres_tokens", "hires_tokens_x2", "hires_tokens_x4", "encoded_prompts_bnc", "grid_positional_encoding"], 
                      output_names=["mask_preds", "iou_preds", "obj_ptrs","obj_score"], 
                      dynamic_axes=({"encoded_prompts_bnc":{1: "num_point"}}),
                      verify=True)
    # 打印保存确认信息
    print(f"saved to: {output_path}/mask_decocder.onnx")

"""video memory 部分导出"""
def export_SAMV3MemoryEncoder(sammodel, output_path: str, device="cpu"):
    SAMV3MemoryEncoder = sammodel.memory_encoder.to(device).eval()
    lowres_image_encoding = torch.randn([1,256,72,72]).to(device)
    mask_prediction = torch.randn([1,1,288,288]).to(device)
    object_score = torch.randn([1,1]).to(device)
    is_prompt_encoding = torch.tensor(True).to(device)
    memory_encoding = SAMV3MemoryEncoder(lowres_image_encoding, mask_prediction, object_score, is_prompt_encoding)
    print(memory_encoding.shape)
    torch.onnx.export(
        SAMV3MemoryEncoder,
        (lowres_image_encoding, mask_prediction, object_score, is_prompt_encoding),
        f"{output_path}/memory_encoder.onnx",
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=["lowres_image_encoding", "mask_prediction", "object_score", "is_prompt_encoding"],
        output_names=["memory_encoding"],
        verify=True
    )

"""memory image fusion 部分导出"""
class SAMV3MemoryImageFusion_wrapper(nn.Module):
    def __init__(self, sammodel):
        super().__init__()
        self.memory_image_fusion = sammodel.memory_image_fusion
        self.layers = self.memory_image_fusion.layers
        self.out_norm =  self.memory_image_fusion.out_norm
    def forward(self, flat_imgtokens_bnc, memory_tokens, memory_posenc, num_ptr_tokens_tensor):
        # Run transformer layers to fuse memory results with image tokens
        h = 72
        w = 72
        b = 1
        patch_hw = (h, w)
        num_ptr_tokens = num_ptr_tokens_tensor.shape[1] - 1
        for layer in self.layers:
            flat_imgtokens_bnc = layer(patch_hw, flat_imgtokens_bnc, memory_tokens, memory_posenc, num_ptr_tokens)

        # Convert back to image-like shape, from: BxNxC -> BxCxHxW
        flat_imgtokens_bnc = self.out_norm(flat_imgtokens_bnc)
        out = flat_imgtokens_bnc.permute(0, 2, 1).reshape(b, -1, h, w)
        return out  # [1, 256, 72, 72]
    
def export_SAMV3MemoryImageFusion(SAMV3MemoryImageFusion_wrapper, output_path: str, device="cpu"):
    SAMV3MemoryImageFusion = SAMV3MemoryImageFusion_wrapper.to(device).eval()
    flat_imgtokens_bnc = torch.randn(1, 5184, 256).to(device)
    memory_tokens = torch.randn(1, 36352, 64).to(device)
    memory_posenc = torch.randn(1, 36352, 64).to(device)
    num_ptr_tokens_tensor = torch.randn(1, 65).to(device)  # 防止傳入空，trt推理時不創建顯存地址
    out = SAMV3MemoryImageFusion(flat_imgtokens_bnc, memory_tokens, memory_posenc, num_ptr_tokens_tensor)
    print(out.shape)
    torch.onnx.export(
        SAMV3MemoryImageFusion,
        (flat_imgtokens_bnc, memory_tokens, memory_posenc, num_ptr_tokens_tensor),
        f"{output_path}/image_SAMV3MemoryImageFusion.onnx",
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=['flat_imgtokens_bnc', 'memory_tokens', 'memory_posenc', 'num_ptr_tokens_tensor'],
        output_names=['pix_feat_with_mem'],
        dynamic_axes={
            'memory_tokens': {1: 'num_feat'},  
            'memory_posenc': {1: 'num_feat'},  
            'num_ptr_tokens_tensor': {1: 'num_ptr_tokens'}, 
        },
        verify=True,
    )
    # 打印保存确认信息
    print(f"saved to: {output_path}/image_SAMV3MemoryImageFusion.onnx")


if __name__ == "__main__":
    # Define pathing & device usage
    model_path = "/home/kvein/sam3/sam3/modelweight/sam3.pt"
    device, dtype = "cpu", torch.float32

    # Define image processing config (shared for all video frames)
    imgenc_config_dict = {"max_side_length": 1008, "use_square_sizing": True}

    # load model
    model_config_dict, sammodel = make_sam_from_state_dict(model_path)
    assert sammodel.name in ("samv2", "samv3"), "Only SAMv2/v3 are supported for video segmentation"
    sammodel.to(device=device, dtype=dtype)

    """
    SAMv1 components except vision backbone part
    """
    onnx_output_dir = "export_scripts/exported_model/onnx"

    # 1: export image_projection_model
    # SAMV3ImageProjectionV2 = SAMV3ImageProjectionV2_wrapper(sammodel)
    # export_SAMV3ImageProjection_v2(SAMV3ImageProjectionV2, output_path=onnx_output_dir)

    # 2: export coordinate_encoder  & 3: export prompt_encoder_model
    # if os.path.exists(f"{onnx_output_dir}/model_weights.pt"):
    #     save_dict = torch.load(f"{onnx_output_dir}/model_weights.pt")
    # else:
    #     save_dict = {}
    # save_dict["coordinate_encoder"] = sammodel.coordinate_encoder.state_dict()
    # save_dict["prompt_encoder"] = sammodel.prompt_encoder.state_dict()
    # torch.save(save_dict, f"{onnx_output_dir}/model_weights.pt")

    # 4: export mask_decoder_model
    export_SAMV3MaskDecoder(sammodel, output_path=onnx_output_dir)

    """
    Video tracking components caried over from SAMv2
    """
    # 5: export memory_encoder_model
    # export_SAMV3MemoryEncoder(sammodel, output_path=onnx_output_dir)
    # 6: export memory_image_fusion_model
    # memory_image_fusion_model = SAMV3MemoryImageFusion_wrapper(sammodel)
    # export_SAMV3MemoryImageFusion(memory_image_fusion_model, output_path=onnx_output_dir)


