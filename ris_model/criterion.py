# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from .modeling.sam.mask2former_miscs import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule



def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))



def matrix_same_cococlass(x): # List -> tensor
    # テンソルをパディングして同じ長さに揃える
    device = x[0]["empty"].device
    label_tensor_lst = [_x['labels'] for _x in x]
    substitute_value = 200
    processed_tensors = []
    for t in label_tensor_lst:
        if t.numel() == 0:
            processed_tensors.append(torch.tensor([substitute_value], device=device))
            substitute_value += 1
        else:
            processed_tensors.append(t)
    max_length = max([_x.size(0) for _x in processed_tensors])
    # 無理やり_x[0]の要素でpaddingした。
    padded_tensor_lst = [torch.nn.functional.pad(_x, (0, max_length - _x.size(0)), value=_x[0]) for _x in processed_tensors]

    # テンソルをスタックして一つのテンソルにする
    tensors = torch.stack(padded_tensor_lst)

    # ブロードキャストを利用して要素ごとの一致を比較
    comparison_matrix = tensors.unsqueeze(1) == tensors.unsqueeze(0)

    # 比較結果を任意の次元に集約（anyを使用）
    result_matrix = comparison_matrix.any(dim=2)
    return result_matrix

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, learn_coco_bert_class, learn_coco_clipcls_class, use_weight_mask, use_bce2CLIP, num_queries, ce_v2l):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        additional_weight = torch.ones(1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.register_buffer("additional_weight", additional_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.learn_coco_bert_class = learn_coco_bert_class
        self.learn_coco_clipcls_class = learn_coco_clipcls_class
        self.use_classification_weight_mask = use_weight_mask
        self.ce_v2l = ce_v2l
        self.use_bce2CLIP = use_bce2CLIP

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        bz = src_logits.shape[0]
        idx = self._get_src_permutation_idx(indices)
        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if self.learn_coco_clipcls_class:
            target_classes_o = torch.zeros(src_logits.shape[0]).to(src_logits.device).long()
            empty_weight = torch.cat([self.additional_weight, self.empty_weight])
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes+1, dtype=torch.int64, device=src_logits.device
            ) # for wacv bgで埋めたいなら+bzな気がする。
        else:
            target_classes_o = idx[0].to(src_logits.device)#target_classes_o = torch.arange(src_logits.shape[0]).to(src_logits.device) # for wacv
            empty_weight = torch.cat([self.additional_weight.repeat(bz), self.empty_weight])
            #target_classes_o = torch.([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # NOTE サイズ１なのでcatできない？
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes+bz, dtype=torch.int64, device=src_logits.device
            ) # for wacv bgで埋めたいなら+bzな気がする。
        target_classes[idx] = target_classes_o
        # non-targertにlossが与えられる。
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        #loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        if self.ce_v2l:
            target_queries_o = idx[1].to(src_logits.device)
            target_queries = torch.full(
                src_logits.shape[:2], self.num_queries + 1, dtype=torch.int64, device=src_logits.device
            )
            target_queries[idx] = target_queries_o
            loss_ce2 =  F.cross_entropy(src_logits.transpose(1, 2), target_queries)
            loss_ce += loss_ce2
        losses = {"loss_ce": loss_ce}
        return losses


    def loss_labels_sigmoid(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        bz, nq, _ = src_logits.shape
        idx = self._get_src_permutation_idx(indices)
        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if self.learn_coco_clipcls_class:
            # NOTE ここ実装できてない。
            target_classes_ref = torch.zeros((bz, nq, 1)).to(src_logits.device).long() # 比較するrefは一つのみ
            empty_weight = torch.cat([self.additional_weight, self.empty_weight]) 
            bce_weight_mask = empty_weight.repeat(bz, nq, 1) # 下での合流に向け名前だけやっつけで変えた。
            classes_across_bz = torch.cat([x['labels'] + 1 for x in targets])# clstoken分の１
            target_classes_ref[idx] = 1.
            target_classes_coco = torch.zeros(
                (bz, nq, self.num_classes+1), dtype=torch.float32, device=src_logits.device
            ) # for wacv bgで埋めたいなら+bzな気がする。
            new_idx = (idx[0], idx[1], classes_across_bz)
            target_classes_coco[new_idx] = 1.

            target_classes = torch.cat([target_classes_ref, target_classes_coco], dim=-1)

        else:
            bce_weight_ref_mask = torch.ones(
                (bz, nq, bz), dtype=torch.float32, device=src_logits.device
            )
            if self.use_classification_weight_mask:
                # （誤差逆伝播を行いたくない要素の重みを0に
                # refcocoのバリエーション少ない問題に対処
                #target_refs =[x['sent'] for x in targets]
                # referring 要素向けの設計。 ここではref向けとcococlass識別向けと分ける。
                #comparison_matrix = classes_across_bz.unsqueeze(0) == classes_across_bz.unsqueeze(1)
                comparison_matrix = matrix_same_cococlass(targets)
                # 同一cocoクラスに対応するweightmaskの要素を0に ref に対してのmaskを作成するのが目的。　NOTE ここ上手くいってない
                bce_weight_ref_mask *= ~(comparison_matrix.unsqueeze(1).expand(-1, bce_weight_ref_mask.shape[1], -1))

                # 対角成分(自分自身)は1に書き換える
                diag_indices = torch.arange(comparison_matrix.size(0))
                bce_weight_ref_mask[diag_indices, :, diag_indices] = 1

            #target_classes_ref_o = idx[0].to(src_logits.device) # ref向け
            # idxはbzとN_qに対するidx
            target_classes_ref = torch.zeros(
                bce_weight_ref_mask.shape, dtype=torch.float32, device=src_logits.device
            )
            idx_bz, idx_nq = idx
            ref_idx = (idx_bz, idx_nq, idx_bz)
            target_classes_ref[ref_idx] = 1.#target_classes_ref_o      
            # cococlass識別向けの設計
            target_classes_coco = torch.zeros(
                (bz, nq, self.num_classes+1), dtype=torch.float32, device=src_logits.device
            ) # for wacv bgで埋めたいなら+bzな気がする。
            classes_across_bz = torch.cat([x['labels'] for x in targets])
            new_idx = (idx_bz, idx_nq, classes_across_bz)
            target_classes_coco[new_idx] = 1.

            # cococlass識別と合体させる
            bce_weight_ref_mask = bce_weight_ref_mask.to(src_logits.device).float()
            empty_weight = self.empty_weight.repeat(bz, nq, 1)
            bce_weight_mask = torch.cat([bce_weight_ref_mask, empty_weight], dim=-1).to(src_logits.device)
            target_classes = torch.cat([target_classes_ref, target_classes_coco], dim=-1).to(src_logits.device)
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes, bce_weight_mask)
        losses = {"loss_ce": loss_ce}
        return losses


    def loss_labels_bertcls(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits_bertcls" in outputs
        src_logits = outputs["pred_logits_bertcls"].float()
        
        idx = self._get_src_permutation_idx(indices)
        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        #target_classes_o = torch.arange(src_logits.shape[0]).to(src_logits.device) # for wacv
        #target_classes_o = torch.([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # NOTE サイズ１なのでcatできない？
        target_classes = torch.zeros(
            src_logits.shape, dtype=torch.float32, device=src_logits.device
        )#for wacv bgで埋めたいなら+bzな気がする。

        bz = src_logits.shape[0]
        if self.learn_coco_bert_class:
            bz_idx, nq_idx=idx
            # NOTE 0703:1900大き目の変更。今までのlearn_coco_bert_class間違ってた可能性
            new_idx = (bz_idx, nq_idx, torch.zeros(bz, device=src_logits.device, dtype=nq_idx.dtype)) # 必ず１要素目にclstokenを配置するため,3軸目は0という想定
            target_classes[new_idx] = 1.
            coco_class_idx = torch.cat([x['labels'] + 1 for x in targets]) # clstoken１つという想定なので+1
            coco_class_idx = (idx[0], idx[1], coco_class_idx) # 0703 18:03まで右のようだった。(idx[0], coco_class_idx)
            target_classes[coco_class_idx] = 1.
        else:
            # NOTE 0703 18:03　大き目な修正 idxについて３軸目を指定しないといけないかも
            #target_classes[idx] = 1.
            bz_idx, nq_idx=idx
            new_idx = (bz_idx, nq_idx, bz_idx)
            target_classes[new_idx] = 1.
            # non-targertにlossが与えられる。
            #loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            if self.use_classification_weight_mask:
                # （誤差逆伝播を行いたくない要素の重みを0に
                bce_weight_mask = torch.ones(
                    src_logits.shape, dtype=torch.float32, device=src_logits.device
                )
                # refcocoのバリエーション少ない問題に対処
                #target_refs =[x['sent'] for x in targets]
                #classes_across_bz = torch.cat([x['labels'] for x in targets])
                #comparison_matrix = classes_across_bz.unsqueeze(0) == classes_across_bz.unsqueeze(1)
                comparison_matrix = matrix_same_cococlass(targets)
                # 同一cocoクラスに対応するweightmaskの要素を0に
                bce_weight_mask *= ~(comparison_matrix.unsqueeze(1).expand(-1, bce_weight_mask.shape[1], -1))
                # 対角成分(自分自身)は1に書き換える
                diag_indices = torch.arange(comparison_matrix.size(0))
                bce_weight_mask[diag_indices, :, diag_indices] = 1
                bce_weight_mask = bce_weight_mask.to(src_logits.device).float()
                #print(comparison_matrix.int())
                #print(bce_weight_mask[0].int())
                #print(bce_weight_mask[1].int())
                loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes, bce_weight_mask)
                losses = {"loss_ce_bertcls": loss_ce}
                return losses
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        losses = {"loss_ce_bertcls": loss_ce}
        return losses


    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }
        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            #'labels': self.loss_labels_sigmoid if self.use_bce2CLIP else self.loss_labels,
            'masks': self.loss_masks,
        }
        if "loss_ce" in self.weight_dict.keys():
           loss_map.update({'labels': self.loss_labels_sigmoid if self.use_bce2CLIP else self.loss_labels})
        if "loss_ce_bertcls" in self.weight_dict.keys():
           loss_map.update({"labels_bertcls":self.loss_labels_bertcls})
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
