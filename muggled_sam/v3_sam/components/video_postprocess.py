import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from torch import Tensor
import numpy.typing as npt
import torch.nn.functional as F


@dataclass
class RealizedAssociateDetTrkresult:
    new_det_fa_inds: np.array
    unmatched_trk_obj_ids: np.array
    det_to_matched_trk_obj_ids: Dict[int, np.array]
    trk_id_to_max_iou_high_conf_det: Dict[int, int]
    empty_trk_obj_ids: np.array
    new_det_obj_ids: Optional[np.array] = None
    new_det_gpu_ids: Optional[np.array] = None
    num_obj_dropped_due_to_limit: Optional[int] = None

    def get_new_det_gpu_ids(
        self, tracker_metadata_prev, is_image_only, det_scores, tracking_obj
    ):
        with torch.profiler.record_function("get_new_det_gpu_ids"):
            if self.new_det_obj_ids is None:
                det_scores_np: np.ndarray = det_scores.cpu().numpy()
                prev_obj_num = np.sum(tracker_metadata_prev["num_obj_per_gpu"])
                new_det_num = len(self.new_det_fa_inds)
                num_obj_dropped_due_to_limit = 0
                if (
                    not is_image_only
                    and prev_obj_num + new_det_num > tracking_obj.max_num_objects
                ):
                    new_det_num_to_keep = tracking_obj.max_num_objects - prev_obj_num
                    num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
                    self.new_det_fa_inds = tracking_obj._drop_new_det_with_obj_limit(
                        self.new_det_fa_inds, det_scores_np, new_det_num_to_keep
                    )
                    assert len(self.new_det_fa_inds) == new_det_num_to_keep
                    new_det_num = len(self.new_det_fa_inds)
                new_det_start_obj_id = tracker_metadata_prev["max_obj_id"] + 1
                new_det_obj_ids = new_det_start_obj_id + np.arange(new_det_num)
                if tracking_obj.is_multiplex:
                    prev_workload_per_gpu = tracker_metadata_prev["num_buc_per_gpu"]
                else:
                    prev_workload_per_gpu = tracker_metadata_prev["num_obj_per_gpu"]
                new_det_gpu_ids = tracking_obj._assign_new_det_to_gpus(
                    new_det_num=new_det_num,
                    prev_workload_per_gpu=prev_workload_per_gpu,
                )
                self.new_det_obj_ids = new_det_obj_ids
                self.new_det_gpu_ids = new_det_gpu_ids
                self.num_obj_dropped_due_to_limit = num_obj_dropped_due_to_limit
            return (
                self.new_det_obj_ids,
                self.new_det_gpu_ids,
                self.num_obj_dropped_due_to_limit,
            )


class LazyAssociateDetTrkResult:
    def __init__(
        self,
        trk_is_unmatched: Tensor,
        trk_is_nonempty: Tensor,
        is_new_det: Tensor,
        det_to_max_iou_trk_idx: Tensor,
        det_is_high_conf: Tensor,
        det_is_high_iou: Tensor,
        det_keep: Tensor,
        im_mask: Tensor,
    ):
        self.trk_is_unmatched = trk_is_unmatched
        self.trk_is_nonempty = trk_is_nonempty
        self.is_new_det = is_new_det
        self.det_to_max_iou_trk_idx = det_to_max_iou_trk_idx
        self.det_is_high_conf = det_is_high_conf
        self.det_is_high_iou = det_is_high_iou
        self.det_keep = det_keep
        self.im_mask = im_mask

    def _convert_to_numpy(self):
        with torch.profiler.record_function("Convert to numpy"):
            self.trk_is_unmatched = self.trk_is_unmatched.cpu().numpy()
            self.trk_is_nonempty = self.trk_is_nonempty.cpu().numpy()
            self.is_new_det = self.is_new_det.cpu().numpy()
            self.det_to_max_iou_trk_idx = self.det_to_max_iou_trk_idx.cpu().numpy()
            self.det_is_high_conf = self.det_is_high_conf.cpu().numpy()
            self.det_is_high_iou = self.det_is_high_iou.cpu().numpy()
            self.det_keep = self.det_keep.cpu().numpy().tolist()
            self.im_mask = self.im_mask.cpu().numpy()

    def _create_cpu_metadata(self, trk_obj_ids, det_masks):
        with torch.profiler.record_function("_create_cpu_metadata"):
            unmatched_trk_obj_ids = trk_obj_ids[self.trk_is_unmatched]
            empty_trk_obj_ids = trk_obj_ids[~self.trk_is_nonempty]
            new_det_fa_inds = np.nonzero(self.is_new_det)[0]
            det_is_high_conf_and_iou = set(
                np.nonzero(self.det_is_high_conf & self.det_is_high_iou)[0]
            )
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            for d in range(det_masks.size(0)):
                if self.det_keep[d]:
                    det_to_matched_trk_obj_ids[d] = trk_obj_ids[self.im_mask[d, :]]
                    if d in det_is_high_conf_and_iou:
                        trk_obj_id = trk_obj_ids[self.det_to_max_iou_trk_idx[d]].item()
                        trk_id_to_max_iou_high_conf_det[trk_obj_id] = d
        return RealizedAssociateDetTrkresult(
            new_det_fa_inds=new_det_fa_inds,
            unmatched_trk_obj_ids=unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det=trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids=empty_trk_obj_ids,
        )


def mask_intersection_vectorized(masks1, masks2):
    """
    Vectorized computation of mask intersection using Matrix Multiplication.

    Args:
        masks1: tensor of shape (N, H, W)
        masks2: tensor of shape (M, H, W)
    Returns:
        tensor of shape (N, M)
    """
    # Cast to float for Tensor Core acceleration via torch.mm
    m1_flat = masks1.flatten(1).float()
    m2_flat = masks2.flatten(1).float()
    intersection = torch.mm(m1_flat, m2_flat.t())
    return intersection.long()


def mask_iom(masks1, masks2):
    """
    Similar to IoU, except the denominator is the area of the smallest mask
    """
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool

    intersection = mask_intersection_vectorized(masks1, masks2)
    area1 = masks1.flatten(-2).sum(-1)
    area2 = masks2.flatten(-2).sum(-1)
    min_area = torch.min(area1[:, None], area2[None, :])
    return intersection / (min_area + 1e-8)


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU (Intersection over Union) between predicted masks and ground truth masks.
    Uses matmul-based vectorized intersection for Tensor Core acceleration.

    Args:
      - pred_masks: (N, H, W) bool Tensor, containing binary predicted segmentation masks
      - gt_masks: (M, H, W) bool Tensor, containing binary ground truth segmentation masks
    Returns:
      - ious: (N, M) float Tensor, containing IoUs for each pair of predicted and ground truth masks
    """
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    assert pred_masks.shape[1:] == gt_masks.shape[1:]

    # Matmul-based intersection (uses Tensor Cores via float mm)
    m1_flat = pred_masks.flatten(1).float()
    m2_flat = gt_masks.flatten(1).float()
    intersection = torch.mm(m1_flat, m2_flat.t())

    area1 = m1_flat.sum(dim=1)
    area2 = m2_flat.sum(dim=1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1)


def _associate_det_trk_compilable(
    det_masks,
    det_scores,
    det_keep,
    trk_masks,
    new_det_thresh,
    iou_threshold_trk,
    iou_threshold,
    HIGH_CONF_THRESH,
    use_iom_recondition,
    o2o_matching_masklets_enable,
    iom_thresh_recondition,
    iou_thresh_recondition,
):
    det_masks_binary = det_masks > 0
    det_masks_binary[~det_keep] = 0
    trk_masks_binary = trk_masks > 0
    intersection_metric = None
    if use_iom_recondition:
        intersection_metric = mask_iom(det_masks_binary, trk_masks_binary)  # (N, M)
    else:
        intersection_metric = mask_iou(det_masks_binary, trk_masks_binary)  # (N, M)

    assert not o2o_matching_masklets_enable, (
        "Temporarily disabled support for o2o_matching_masklets_enable, due to optimizations."
    )

    if o2o_matching_masklets_enable:
        intersection_metric_np = intersection_metric.cpu().numpy()
        from scipy.optimize import linear_sum_assignment

        cost_matrix = 1 - intersection_metric_np
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
        for d, t in zip(row_ind, col_ind):
            if intersection_metric_np[d, t] >= iou_threshold_trk:
                trk_is_matched[t] = True
        trk_is_matched = torch.from_numpy(trk_is_matched)
        trk_is_matched = trk_is_matched.to(device=intersection_metric.device)
    else:
        trk_is_matched = (intersection_metric >= iou_threshold_trk).any(dim=0)
    # Non-empty tracks not matched by Hungarian assignment above threshold are unmatched
    trk_is_nonempty = trk_masks_binary.any(dim=(1, 2))
    trk_is_unmatched = torch.logical_and(trk_is_nonempty, ~trk_is_matched)

    # For detections: allow many tracks to match to the same detection (many-to-one)
    # So, a detection is 'new' if it does not match any track above threshold
    is_new_det = torch.logical_and(
        torch.logical_and((det_scores >= new_det_thresh), (det_keep)),
        torch.logical_not(torch.any(intersection_metric >= iou_threshold, dim=1)),
    )

    intersection_thresh_recond = (
        iom_thresh_recondition if use_iom_recondition else iou_thresh_recondition
    )
    # if a detection matches to many tracks with high IoU or vice versa, we do not consider it for reconditioning as it might be ambiguous
    det_match_to_many_trk = (intersection_metric >= intersection_thresh_recond).sum(
        dim=1
    ) > 1
    trk_match_to_many_det = (intersection_metric >= intersection_thresh_recond).sum(
        dim=0
    ) > 1
    # # zero out these ambiguous matches
    # intersection_metric[:, trk_match_to_many_det] = (
    #     0.0  # only consider unique matches
    # )

    # intersection_metric[det_match_to_many_trk, :] = (
    #     0.0  # only consider unique matches
    # )

    intersection_metric = torch.where(
        trk_match_to_many_det.unsqueeze(0),
        torch.zeros_like(intersection_metric),
        intersection_metric,
    )

    intersection_metric = torch.where(
        det_match_to_many_trk.unsqueeze(1),
        torch.zeros_like(intersection_metric),
        intersection_metric,
    )

    det_to_max_iou_trk_idx = torch.argmax(intersection_metric, dim=1)
    det_is_high_conf = ((det_scores >= HIGH_CONF_THRESH) & det_keep) & ~is_new_det
    det_is_high_iou = (
        torch.amax(intersection_metric, dim=1) >= intersection_thresh_recond
    )
    im_mask = intersection_metric >= iou_threshold

    return (
        trk_is_unmatched,
        trk_is_nonempty,
        is_new_det,
        det_to_max_iou_trk_idx,
        det_is_high_conf,
        det_is_high_iou,
        det_keep,
        im_mask,
    )


def associate_det_trk(
    det_masks: Tensor,
    det_scores_np: npt.NDArray,
    trk_masks: Tensor,
    trk_obj_ids: npt.NDArray,
    assoc_iou_thresh=0.5,
    trk_assoc_iou_thresh=0.5,
    new_det_thresh=0.0,
    use_iom_recondition=False,
    o2o_matching_masklets_enable=False,
    iom_thresh_recondition=0.8,
    iou_thresh_recondition=0.8,
):
    """
    Match detections on the current frame with the existing masklets.

    Args:
        - det_masks: (N, H, W) tensor of predicted masks
        - det_scores_np: (N,) array of detection scores
        - trk_masks: (M, H, W) tensor of track masks
        - trk_obj_ids: (M,) array of object IDs corresponding to trk_masks

    Returns:
        - new_det_fa_inds: array of new object indices.
        - unmatched_trk_obj_ids: array of existing masklet object IDs that are not matched
        to any detections on this frame (for unmatched, we only count masklets with >0 area)
        - det_to_matched_trk_obj_ids: dict[int, npt.NDArray]: mapping from detector's detection indices
        to the list of matched tracklet object IDs
        - trk_id_to_max_iou_high_conf_det: dict mapping track obj_id to the highest-IoU high-conf detection idx
        - empty_trk_obj_ids: array of existing masklet object IDs with zero area in SAM2 prediction
    """
    iou_threshold = assoc_iou_thresh
    iou_threshold_trk = trk_assoc_iou_thresh
    new_det_thresh = new_det_thresh

    assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
    assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
    assert trk_masks.size(0) == len(trk_obj_ids), (
        f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
    )
    if trk_masks.size(0) == 0:
        # all detections are new
        new_det_fa_inds = np.arange(det_masks.size(0))
        unmatched_trk_obj_ids = np.array([], np.int64)
        empty_trk_obj_ids = np.array([], np.int64)
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}
        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )
    elif det_masks.size(0) == 0:
        # all previous tracklets are unmatched if they have a non-zero area
        new_det_fa_inds = np.array([], np.int64)
        trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
        empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}
        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    if det_masks.shape[-2:] != trk_masks.shape[-2:]:
        # resize to the smaller size to save GPU memory
        if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
            trk_masks = F.interpolate(
                trk_masks.unsqueeze(1),
                size=det_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        else:
            # resize detections to track size
            det_masks = F.interpolate(
                det_masks.unsqueeze(1),
                size=trk_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

    # Convert numpy scores to tensor for the compilable function
    det_scores = torch.from_numpy(det_scores_np).to(det_masks.device)
    det_keep = torch.ones(
        det_masks.size(0), dtype=torch.bool, device=det_masks.device
    )

    # Call the GPU-native compilable function
    adt_result_tensors = _associate_det_trk_compilable(
        det_masks=det_masks,
        det_scores=det_scores,
        det_keep=det_keep,
        trk_masks=trk_masks,
        new_det_thresh=new_det_thresh,
        iou_threshold_trk=iou_threshold_trk,
        iou_threshold=iou_threshold,
        HIGH_CONF_THRESH=0.8,
        use_iom_recondition=use_iom_recondition,
        o2o_matching_masklets_enable=o2o_matching_masklets_enable,
        iom_thresh_recondition=iom_thresh_recondition,
        iou_thresh_recondition=iou_thresh_recondition,
    )

    # Wrap in LazyAssociateDetTrkResult and immediately realize to numpy
    # for backward compatibility with existing callers
    lazy_result = LazyAssociateDetTrkResult(*adt_result_tensors)
    lazy_result._convert_to_numpy()
    realized = lazy_result._create_cpu_metadata(trk_obj_ids, det_masks)

    return (
        realized.new_det_fa_inds,
        realized.unmatched_trk_obj_ids,
        realized.det_to_matched_trk_obj_ids,
        realized.trk_id_to_max_iou_high_conf_det,
        realized.empty_trk_obj_ids,
    )