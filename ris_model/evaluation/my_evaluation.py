# from xdecoder
import torch.distributed as dist
import torch
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import logging
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator

class GroundingEvaluator(DatasetEvaluator):
    """
    Evaluate grounding segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        save_imgs=False,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._save_imgs = save_imgs

        self._cpu_device = torch.device("cpu")
        self._available_sources = ["refcoco", "grefcoco"]

        self._num_classes = 2

    def reset(self):
        self.cum_I = 0
        self.cum_U = 0
        self.mIoU = 0
        self.gIoU = 0
        self.eval_seg_iou_list = [.5, .6, .7, .8, .9]
        self.seg_correct = torch.zeros(len(self.eval_seg_iou_list), device=self._cpu_device)
        self.seg_total = 0
        self.gres_acc = 0
        self.seg_total_sub = 0
       
    @staticmethod
    def computeIoU(pred_seg, gd_seg):
        I = (pred_seg & gd_seg)
        U = (pred_seg | gd_seg)
        return I, U


    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred = (output["ref_seg"] > 0.5).to(self._cpu_device)
            if "nt_label" in output:
                output_nt = output["nt_label"].argmax(dim=0).bool().to(self._cpu_device)
                pred_nt = bool(output_nt) # Trueの場合non-taget
                if pred_nt:
                    pred = (torch.zeros(pred.shape) > 0.5).to(self._cpu_device)
            gt = input['gt_mask_merged'].to(self._cpu_device)
            bsi = len(pred)
            I, U = self.computeIoU(pred, gt)
            if input['source'] == 'refzom':
                if input['empty']:
                    incorrect_num= torch.sum(pred)
                    if incorrect_num == 0:
                        acc = 1.
                    else:
                        acc = 0.
                    self.gres_acc += acc
                    self.seg_total_sub += bsi
                else:
                    if torch.sum(U) != 0:
                        IoU = I.reshape(bsi,-1).sum(-1)*1.0 / (U.reshape(bsi,-1).sum(-1) + 1e-6)
                        self.mIoU += IoU.sum().cpu()
                    else:
                        self.mIoU += 0.
                    self.cum_I += I.sum().cpu()
                    self.cum_U += U.sum().cpu()
                    self.seg_total += bsi
                self.gIoU += torch.tensor(0.)
            else:
                self.cum_I += I.sum().cpu()
                self.cum_U += U.sum().cpu()
                IoU = I.reshape(bsi,-1).sum(-1)*1.0 / (U.reshape(bsi,-1).sum(-1) + 1e-6)
                self.mIoU += IoU.sum().cpu()
                acc_iou = I.reshape(bsi,-1).sum(-1)*1.0 / (U.reshape(bsi,-1).sum(-1) + 1e-5) # LISAでは1e-5だった
                if torch.sum(U) == 0:
                    acc_iou = 1.  # no-object target
                self.gIoU += acc_iou
                for idx in range(len(self.eval_seg_iou_list)):
                    eval_seg_iou = self.eval_seg_iou_list[idx]
                    self.seg_correct[idx] += (IoU >= eval_seg_iou).sum().cpu()
                    #if self._compute_box:
                    #    self.seg_correct_box[idx] += (IoU_box >= eval_seg_iou).sum().cpu()
                self.seg_total += bsi
            
    def evaluate(self):
        if self._distributed:
            synchronize()
            self.cum_I = torch.stack(all_gather(self.cum_I)).sum()
            self.cum_U = torch.stack(all_gather(self.cum_U)).sum()
            self.mIoU = torch.stack(all_gather(self.mIoU)).sum()
            self.gIoU = torch.stack(all_gather(self.gIoU)).sum()
            self.seg_correct = torch.stack(all_gather(self.seg_correct)).sum(0)
            self.seg_total = sum(all_gather(self.seg_total))
            self.gres_acc = sum(all_gather(self.gres_acc))
        #    if self._compute_box:
        #        self.mIoU_box = torch.stack(all_gather(self.mIoU_box)).sum()
        #        self.seg_correct_box = torch.stack(all_gather(self.seg_correct_box)).sum(0)
            if not is_main_process():
                return

        results = {}
        for idx in range(len(self.eval_seg_iou_list)):
            result_str = 'precision@{}'.format(self.eval_seg_iou_list[idx])
            results[result_str] = (self.seg_correct[idx]*100 / self.seg_total).item()
        results['cIoU'] = (self.cum_I*100./self.cum_U).item()
        results['mIoU'] = (self.mIoU*100./self.seg_total).item()
        results['gIoU'] = (self.gIoU*100./self.seg_total).item()
        results['gres_acc'] = self.gres_acc*100/(self.seg_total_sub + 1e-4)
        
        #self._logger.info(results)
        return results