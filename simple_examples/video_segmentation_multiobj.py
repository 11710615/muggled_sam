#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import muggled_sam  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to muggled_sam folder!")
from collections import defaultdict
import cv2
import numpy as np
import torch
from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
from time import perf_counter

def normalize_point(point_list, width, height):
    """将像素坐标归一化到 [0, 1] 范围"""
    norm_points_list = []
    for point in point_list:
        x, y = point
        norm_points_list.append((x / width, y / height))
    return norm_points_list

def apply_non_overlapping_constraints(pred_masks):
    """
    Apply non-overlapping constraints to the object scores in pred_masks. Here we
    keep only the highest scoring object at each spatial location in pred_masks.
    """
    batch_size = pred_masks.size(0)
    if batch_size == 1:
        return pred_masks

    device = pred_masks.device
    # "max_obj_inds": object index of the object with the highest score at each location
    max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
    # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
    batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
    keep = max_obj_inds == batch_obj_inds
    # suppress overlapping regions' scores below -10.0 so that the foreground regions
    # don't overlap (here sigmoid(-10.0)=4.5398e-05)
    pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
    return pred_masks

def safe_destroy_window(wintitle):
    """安全销毁 OpenCV 窗口（窗口不存在时不报错）"""
    try:
        cv2.getWindowProperty(wintitle, cv2.WND_PROP_VISIBLE)
        cv2.destroyWindow(wintitle)
    except cv2.error:
        pass


def safe_destroy_all_windows():
    """安全销毁全部 OpenCV 窗口（headless 环境不报错）"""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def enhance_image_contrast(frame, alpha=1.15, beta=15):
    """提高原图对比度，让叠加的 mask 更醒目"""
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def overlay_color_masks(frame, obj_masks, obj_color_map, alpha=0.35):
    """在原图上叠加多对象颜色 mask，并保持原图可见"""
    overlay = frame.copy()
    for obj_key, mask in obj_masks.items():
        color = obj_color_map[obj_key]
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            continue
        overlay[mask_bool] = color

    return cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)


def draw_instance_labels(frame, obj_masks, obj_color_map):
    """在 overlay 图上为每个实例绘制 ID 标签"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    for obj_key, mask in obj_masks.items():
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            continue
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            continue
        x_min, y_min = int(xs.min()), int(ys.min())
        label = f"ID:{obj_key}"
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = max(x_min, 5)
        text_y = max(y_min - 5, text_size[1] + 5)
        cv2.rectangle(
            frame,
            (text_x - 2, text_y - text_size[1] - 2),
            (text_x + text_size[0] + 2, text_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, obj_color_map[obj_key], thickness)

# Define pathing & device usage
video_path = "/mnt/kvein/pank1/muggled_sam3_trt_infer_v1/test_image_video/videos/bedroom.mp4"
model_path = "/home/kvein/sam3/sam3/modelweight/sam3.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16

# Define image processing config (shared for all video frames)
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

# Read first frame to check that we can read from the video, then reset playback
vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
ok_frame, first_frame = vcap.read()
frame_height, frame_width = first_frame.shape[0], first_frame.shape[1]
assert ok_frame, f"Could not read frames from video: {video_path}"
vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)


prompts_per_frame_index = {
    0: {
        "obj1": {
            "box_tlbr_norm_list": [],
            "fg_xy_norm_list": normalize_point([[200, 300]], frame_width, frame_height),  # 前景点（绿色）
            "bg_xy_norm_list": [], #normalize_point([[275, 175]], frame_width, frame_height),  # 背景点（红色）
        },
        "obj2": {
            "box_tlbr_norm_list": [normalize_point([[300, 0], [500, 400]], frame_width, frame_height)],
            "fg_xy_norm_list": [],
            "bg_xy_norm_list": [],
        },
    },
    # 可在后续帧追加新对象或更新已有对象的 prompt：
    # 100: {
    #     "obj3": {
    #         "box_tlbr_norm_list": [...],
    #         "fg_xy_norm_list": [...],
    #         "bg_xy_norm_list": [...],
    #     },
    # },
}

enable_prompt_visualization = False
# *** These prompts are set up for a video of horses available from pexels.com ***
# https://www.pexels.com/video/horses-running-on-grassland-4215784/
# By: Adrian Hoparda

# Set up memory storage for tracked objects
# -> Assumes each object is represented by a unique dictionary key (e.g. 'obj1')
# -> This holds both the 'prompt' & 'recent' memory data needed for tracking!
memory_history_length = 7   # preframe 记忆保留帧数（滚动窗口）
memory_per_obj_dict = defaultdict(lambda: SAMVideoObjectResults.create(memory_history_length=memory_history_length))
non_overlap_masks_for_output = False

# 为每个对象分配稳定颜色，overlay 时使用
obj_color_palette = [
    (0, 0, 255),     # 红
    (0, 255, 0),     # 绿
    (255, 0, 0),     # 蓝
    (0, 255, 255),   # 黄
    (255, 0, 255),   # 紫
    (255, 165, 0),   # 橙
    (128, 0, 128),   # 深紫
    (0, 128, 255),   # 青
]
obj_color_map = {}


# Set up model
print("Loading model...")
stream = torch.cuda.Stream()
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
assert sammodel.name in ("samv2", "samv3"), "Only SAMv2/v3 are supported for video segmentation"
sammodel.to(device=device, dtype=dtype)

# Process video frames
stack_func = np.hstack if first_frame.shape[0] > first_frame.shape[1] else np.vstack
close_keycodes = {27, ord("q")}  # Esc or q to close
highgui_available = True
time_count_list = []
try:
    with torch.autocast(device_type=device, dtype=dtype), torch.cuda.stream(stream):
        total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx in range(total_frames):

            # Read frames
            ok_frame, frame = vcap.read()
            if not ok_frame:
                print("", "Done! No more frames...", sep="\n")
                break

            t1 = perf_counter()

            # Encode frame data (shared for all objects)
            encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)

            # Generate & store prompt memory encodings for each object as needed
            prompts_dict = prompts_per_frame_index.get(frame_idx, None)
            if prompts_dict is not None:
                # Loop over all sets of prompts for the current frame
                for obj_key_name, obj_prompts in prompts_dict.items():
                    print(f"Generating prompt for object: {obj_key_name} (frame {frame_idx})")
                    init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(encoded_imgs_list, **obj_prompts)
                    memory_per_obj_dict[obj_key_name].store_prompt_result(frame_idx, init_mem, init_ptr)

                    # Draw prompts for debugging
                    if enable_prompt_visualization:
                        prompt_vis_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
                        norm_to_px_factor = np.float32((prompt_vis_frame.shape[1] - 1, prompt_vis_frame.shape[0] - 1))
                        for xy_norm in obj_prompts.get("fg_xy_norm_list", []):
                            xy_px = np.int32(xy_norm * norm_to_px_factor)
                            cv2.circle(prompt_vis_frame, xy_px, 3, (0, 255, 0), -1)
                        for xy_norm in obj_prompts.get("bg_xy_norm_list", []):
                            xy_px = np.int32(xy_norm * norm_to_px_factor)
                            cv2.circle(prompt_vis_frame, xy_px, 3, (0, 0, 255), -1)
                        for xy1_norm, xy2_norm in obj_prompts.get("box_tlbr_norm_list", []):
                            xy1_px = np.int32(xy1_norm * norm_to_px_factor)
                            xy2_px = np.int32(xy2_norm * norm_to_px_factor)
                            cv2.rectangle(prompt_vis_frame, xy1_px, xy2_px, (0, 255, 255), 2)

                        # Show prompt in it's own window and close after viewing
                        wintitle = f"Prompt ({obj_key_name}) - Press key to continue"
                        cv2.imshow(wintitle, prompt_vis_frame)
                        cv2.waitKey(0)
                        cv2.destroyWindow(wintitle)

            # Update tracking using newest frame
            # ================================================================
            obj_masks_for_postprocess_dict = {}
            obj_masks = {}
            for obj_idx, (obj_key_name, obj_memory) in enumerate(memory_per_obj_dict.items()):
                if obj_key_name not in obj_color_map:
                    obj_color_map[obj_key_name] = obj_color_palette[obj_idx % len(obj_color_palette)]
                # 融合 prompt 长期记忆 + preframe 短期记忆，预测当前帧 mask
                obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                    encoded_imgs_list, **obj_memory.to_dict()
                )

                # 检查追踪质量：score < 0 表示模型认为对象丢失或遮挡
                obj_score = obj_score.item()
                if obj_score < 0:
                    print(f"Bad object score for {obj_key_name}! Skipping memory storage...")
                    continue  # 跳过记忆存储，避免将低质量结果污染 prevframe_buffer

                # Store 'recent' memory encodings from current frame (helps track objects with changing appearance)
                # -> This can be commented out and tracking may still work, if object doesn't change much
                obj_memory.store_frame_result(frame_idx, mem_enc, obj_ptr)

                # Add object mask prediction to 'combine' mask for display
                # -> This is just for visualization, not needed for tracking
                obj_mask = torch.nn.functional.interpolate(
                    mask_preds[:, best_mask_idx, :, :],
                    size=frame.shape[0:2],
                    mode="bilinear",
                    align_corners=False,
                )
                obj_masks_for_postprocess_dict[obj_key_name] = obj_mask

            """postprocess: 解决多对象分割时对象重叠的问题"""
            obj_masks_for_postprocess = torch.cat(list(obj_masks_for_postprocess_dict.values()), dim=0)
            obj_keys = list(obj_masks_for_postprocess_dict.keys())
            if non_overlap_masks_for_output:
                obj_masks_for_postprocess = apply_non_overlapping_constraints(obj_masks_for_postprocess)
            obj_masks_for_postprocess = obj_masks_for_postprocess.cpu().numpy()
            for i in range(len(obj_keys)):
                obj_masks[obj_keys[i]] = (obj_masks_for_postprocess[i] > 0.0).squeeze()

                # obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
                # # Keep key type consistent with obj_color_map (uses obj_key_name)
                # obj_masks[obj_key_name] = obj_mask_binary

            # ---- 推理计时 ----
            t2 = perf_counter()
            print(f"Took {round(1000 * (t2 - t1))} ms for {len(memory_per_obj_dict)} objects")
            time_count_list.append(1000 * (t2 - t1))

            # ---- 叠加 mask 到原图并增强对比度 ----
            # enhanced_frame = enhance_image_contrast(frame)
            overlay_frame = overlay_color_masks(frame, obj_masks, obj_color_map, alpha=0.35)
            draw_instance_labels(overlay_frame, obj_masks, obj_color_map)
            display_frame = overlay_frame


            if highgui_available:
                try:
                    cv2.imshow("Video Segmentation Result - q to quit", display_frame)
                    keypress = cv2.waitKey(1) & 0xFF
                    if keypress in close_keycodes:
                        break
                except cv2.error:
                    highgui_available = False
                    print("OpenCV highgui is unavailable; running in headless mode without imshow.")

        # 输出平均推理耗时（跳过前10帧预热）
        print("average inference time:", sum(time_count_list[10:]) / len(time_count_list[10:]), "ms")


except Exception as err:
    raise err

except KeyboardInterrupt:
    print("Closed by ctrl+c!")

finally:
    vcap.release()
    cv2.destroyAllWindows()
