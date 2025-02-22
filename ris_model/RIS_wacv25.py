from typing import Tuple

import torch,math
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.backbone.clip_wrapper import build_clip
from .modeling.sam.build_sam import sam_model_registry

from .modeling.backbone.backbone import SimpleFeaturePyramid, MMLayer, ViTDummy, MySimpleFeaturePyramid
from .modeling.module.mlp import MLP
from .modeling.module.pixel_decoder import build_pixel_decoder
from .modeling.decoder.txt2txt import build_txt2txt, Txt2txt_casa
from .modeling.decoder.dfam import DFAM

from .criterion import SetCriterion
from .matcher import HungarianMatcher
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import clip, einops
from transformers import BertTokenizer
from .my_misc import coco_classeslst_bert, load_coco_classes_bert_textlst
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
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
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


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
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss 
def ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor):

    loss = F.cross_entropy(inputs, targets, weight=weight)

    return loss


ce_loss_jit = torch.jit.script(
    ce_loss
)  # type: torch.jit.ScriptModule
def get_class_names(dataset_name: str):
    # COCO panoptic
    if dataset_name == "coco_2017_train_panoptic" or \
        dataset_name == "coco_2017_val_panoptic_with_sem_seg":
        class_names = [x['name'] for x in COCO_CATEGORIES]
    # ADE 150
    elif dataset_name == "ade20k_panoptic_val" or \
        dataset_name == "ade20k_panoptic_train":
        class_names = [x['name'] for x in ADE20K_150_CATEGORIES]
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")

    if 'train' in dataset_name:
        class_names.append('other')
    return class_names
@META_ARCH_REGISTRY.register()
class RIS_wacv25(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        sam,
        backbone_clip,
        backbone_sam,
        clip_pixel_decoder,
        tokenizer,
        max_tokens,
        clip_module: Backbone,
        criterion: nn.Module,
        lang_backbone: nn.Module,
        dim_projection:int,
        early_dim_projection:int,
        use_pickle:bool,
        clip_version:str,
        learn_coco_bert_class:bool, #in_sam_embed:bool,
        hungarian_weight:float,
        dice_loss_weight:float,
        bce_loss_weight:float,
        #accept_clip_embed:str,
        use_nomask_embed_weight:bool,
        img_size:int,
        #text_embed_in_MD :bool,
        use_clip_loss:bool,
        txt2txt:nn.Module,
        txt2txt_io:str,
        learn_coco_clipcls_class:bool,
        use_mlp_proj:bool,
        #mm_first:bool,
        #semseg_out_maskfeature:bool,
        use_instances_annos:bool,
        #use_sparse_emb:bool,
        clip_layer_idx:int,
        #type_sparse: str,
        gres_option:dict,
        objemb_proj:bool,
        dim_lang:int = 768,
    ):
        super().__init__()
        self.sam = sam
        self.criterion = criterion
        self.use_instances_annos = use_instances_annos
        self.is_pickle = use_pickle
        # language backbone
        self.text_encoder = lang_backbone
        #self.type_sparse = type_sparse
        
        # こっから書き直す。
        clip_t_dim = 768 if clip_version in ['ViT-L/14@336px',  "ViT-L/14"] else 512
        clip_v_dim = 1024 if clip_version in ['ViT-L/14@336px',  "ViT-L/14"] else 768

        if use_mlp_proj:
            self.lang_mlp = MLP(dim_lang, dim_lang, dim_projection) if dim_lang != dim_projection else nn.Sequential()
            self.clip_mlpl = MLP(clip_t_dim, dim_projection, dim_projection)
        else:
            self.lang_mlp = nn.Linear(dim_lang, early_dim_projection) if (dim_lang != early_dim_projection) else nn.Sequential()
            self.clip_mlpl =nn.Linear(dim_lang, early_dim_projection) # ここやばいかも
            
        if dim_projection != early_dim_projection:
            self.sparse_emb_proj = nn.Linear(early_dim_projection, dim_projection)
       
        self.clip_dense = clip_module

        
        self.txt2txt = txt2txt
        self.txt2txt_io = txt2txt_io
        if txt2txt_io == 'clip_cls_v':
            self.clip_cls_v_proj = nn.Linear(clip_v_dim, early_dim_projection)

    
        self.backbone_clip = backbone_clip
        self.backbone_sam = backbone_sam
        # ここでin 768 out 256
        self.clip_pixel_decoder = clip_pixel_decoder
        self.decoder = DFAM(in_channels=dim_projection, hidden_dim=dim_projection, num_queries=100, nheads=8, dim_feedforward=1024,
                dec_layers=3, pre_norm=False, mask_dim=dim_projection, enforce_input_project=False, logit_dim=clip_t_dim, sam_intermidiate_dim=dim_projection,
                gres_option=gres_option, 
                learn_coco_bert_class=learn_coco_bert_class, learn_coco_clipcls_class=learn_coco_clipcls_class,
                early_dim=early_dim_projection)


        
        self.adaptor_clip = MMLayer(clip_v_dim, early_dim_projection, dim_projection)
        self.gres_option = gres_option
        if gres_option['nt_label']:
            nt_weight = torch.FloatTensor([0.9, 1.1])
            self.register_buffer('nt_weight', nt_weight)
            self.nt_label_weight = gres_option['nt_label_weight']

        if objemb_proj:
            self.objemb_proj = nn.Linear(dim_projection, early_dim_projection)
        #self.embedding_mlp = MLP(dim_projection, dim_projection, dim_projection) if use_sparse_emb else nn.Sequential()
        #self.use_sparse_emb = use_sparse_emb
        self.hungarian_weight = hungarian_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight

        #self.accept_clip_embed = True
        self.use_clip_loss = use_clip_loss
        self.use_nomask_embed_weight = use_nomask_embed_weight 
        self.img_size = (img_size, img_size)
        #self.text_embed_in_MD = text_embed_in_MD
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.clip_layer_idx = clip_layer_idx
        empty_sparse_emb = torch.empty((1, 0, early_dim_projection))
        self.register_buffer('empty_sparse_emb', empty_sparse_emb)
        if hasattr(self, 'clip_dense'):
            clip_ = self.clip_dense.clip_model
            device = 'cuda' # ABCIでは一度GPUに乗せないと亀速度になるため、こんな実装になった
            with torch.cuda.device(device):
                text_embeddings = self.load_text_embedding("coco_2017", clip_.to(device), device)
            self.register_buffer('text_embeddings', text_embeddings)

        self.learn_coco_clipcls_class = learn_coco_clipcls_class
        self.learn_coco_bert_class = learn_coco_bert_class
        if learn_coco_bert_class:
            class_names= load_coco_classes_bert_textlst()
            bert_coco_classes = self.tokenizer(
                class_names,
                add_special_tokens=True,  # 特殊トークンを追加
                padding=True,             # パディングを自動で追加
                truncation=True,          # 長い文をトランケート
                return_tensors='pt'       # PyTorchのテンソルを返す
            )
            for k, v in bert_coco_classes.items():
                self.register_buffer('bert_coco_classes_inputs_'+k, v)
        self._freeze()
        
        
    @classmethod
    def from_config(cls, cfg):
        tokenizer = BertTokenizer.from_pretrained(cfg.REFERRING.BERT_TYPE)

        clip_module = build_clip(cfg)
        sam_pretrained = {
            'vit_b': "sam_vit_b_01ec64.pth",
            'vit_h': "sam_vit_h_4b8939.pth",
        }
        sam = sam_model_registry[cfg.MODEL.SAM.VIT_TYPE](sam_pretrained[cfg.MODEL.SAM.VIT_TYPE])
        

        text_encoder = BertModel.from_pretrained(cfg.REFERRING.BERT_TYPE) 
        text_encoder.pooler = None

        dim_projection = cfg.MODEL.MY_DECODER.DIM
        backbone_clip_dim = dim_projection
        backbone_clip = ViTDummy(backbone_clip_dim, patch_size=14)
        backbone_clip_outdim = 768
        backbone_clip = SimpleFeaturePyramid(
            net=backbone_clip,
            in_feature="last_feat",
            out_channels=backbone_clip_outdim,
            scale_factors=cfg.MODEL.MY_DECODER.CLIP_SCALE_FACTORS,
            top_block=None,
            norm="LN",
            square_pad=336,
        )
        clip_pixel_decoder = build_pixel_decoder(cfg, backbone_clip.output_shape())
        backbone_sam = ViTDummy(256, patch_size=16)
        backbone_sam = MySimpleFeaturePyramid(
            net=backbone_sam,
            in_feature="last_feat",
            out_channels=dim_projection,
            scale_factors=cfg.MODEL.MY_DECODER.SAM_SCALE_FACTORS,
            top_block=None,
            norm="LN",
            square_pad=1024,
        )

        txt2txt = build_txt2txt(cfg) if cfg.MODEL.TXT2TXT.IO is not None else None
        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        class_bertcls_weight = cfg.MODEL.MY_DECODER.CLASS_BERTCLS_WEIGHT

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_ce_bertcls": class_bertcls_weight}
        losses = ["labels", "masks", "labels_bertcls"]
        if not cfg.MODEL.USE_CLIP_TXT:
            weight_dict.pop("loss_ce")
            losses.remove("labels")
            class_weight = 0.

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_class_bertcls=class_bertcls_weight, # これの影響不明
            #is_sigmoid=cfg.MODEL.MY_DECODER.MATHER_SIGMOID,
            num_points=12544, #cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            learn_coco_bert_class=cfg.MODEL.BERT_COCO_CLASS,
            learn_coco_clipcls_class=cfg.MODEL.CLIPCLS_COCO_CLASS,
            use_bce2clip=cfg.MODEL.USE_BCE2CLIP
        )
        dec_layers = len(cfg.MODEL.MY_DECODER.CLIP_SCALE_FACTORS)
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        criterion = SetCriterion(
            len(COCO_CATEGORIES), #sem_seg_head.num_classes, # COCO + ref expression
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.MODEL.MY_DECODER.NO_OBJECT_WEIGHT,
            losses=losses,
            num_points=12544, #cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=3.0, #cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=0.75, #cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            learn_coco_bert_class=cfg.MODEL.BERT_COCO_CLASS,
            learn_coco_clipcls_class=cfg.MODEL.CLIPCLS_COCO_CLASS,
            use_weight_mask=cfg.MODEL.CRASSIFICATION_WEIGHT_MASK,
            use_bce2CLIP=cfg.MODEL.USE_BCE2CLIP,
            num_queries=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            ce_v2l=cfg.MODEL.CLIP_CE_V2L,
        )

        gres_option = {
            'nt_label':  True if ('grefcoco' in cfg.DATASETS.TRAIN[0] or 'refzom' in cfg.DATASETS.TRAIN[0]) else False, # Datasetはlistのため
            'reduction': cfg.GRES.REDUCTION,
            'use_minimap':  cfg.GRES.USE_MINIMAP,
            'nt_label_weight': cfg.GRES.NT_LABEL_WEIGHT,
        }
       
        return {
            #"semseg_out_maskfeature":cfg.MODEL.SEM_SEG_HEAD.OUT_FEATURES_IF_CLIP,
            #"sam_pixel_decoder": sam_pixel_decoder,
            'img_size':cfg.INPUT.IMAGE_SIZE,
            "hungarian_weight" : cfg.MODEL.MY_DECODER.MATCHER_WEIGHT,
            "dice_loss_weight" : cfg.MODEL.MY_DECODER.DICE_LOSS_WEIGHT,
            "bce_loss_weight" : cfg.MODEL.MY_DECODER.BCE_LOSS_WEIGHT,
            #"accept_clip_embed" : cfg.MODEL.MY_DECODER.ACCEPT_CLIPEMBED,
            "use_nomask_embed_weight" : cfg.MODEL.MY_DECODER.NOMASK_EMBED_WEIGHT,
            #"in_sam_embed": cfg.MODEL.MY_DECODER.IN_IMAGE_EMBEDDING,
            "learn_coco_bert_class": cfg.MODEL.BERT_COCO_CLASS,
            "learn_coco_clipcls_class": cfg.MODEL.CLIPCLS_COCO_CLASS,

            "sam":sam,
            "early_dim_projection": cfg.MODEL.MY_DECODER.EARLY_DIM,
            "max_tokens": cfg.REFERRING.MAX_TOKENS,
            "tokenizer": tokenizer,
            "backbone_sam": backbone_sam,
            "backbone_clip":backbone_clip,
            "clip_pixel_decoder": clip_pixel_decoder,
            #"use_dummy_mask": cfg.MODEL.MY_DECODER.DUMMY_LANG_MASK, 
            "clip_module":  clip_module,
            #'is_txt2txt_pe': cfg.MODEL.MY_DECODER.IS_TXT2TXT_PE,
            "clip_version": cfg.MODEL.CLIP.VERSION,
            "use_pickle": cfg.REFERRING.USE_PICKLE,
            "dim_projection": cfg.MODEL.MY_DECODER.DIM,
            "criterion": criterion,
            #"text_embed_in_MD": cfg.MODEL.MY_DECODER.TEXT_EMBED_IN_MD,
            "lang_backbone": text_encoder,
            "txt2txt": txt2txt,
            #"weight_target":cfg.MODEL.MY_DECODER.WEIGHT_TARGET,
            #"mm_first": cfg.MODEL.MY_DECODER.MM_FIRST,
            "txt2txt_io": cfg.MODEL.TXT2TXT.IO,
            #"adaptor_sam_lang": cfg.MODEL.MY_DECODER.PWAM_SAM,
            #"refering_token_num":cfg.REFERRING.MAX_TOKENS,
            #"seq_lang_type" : cfg.MODEL.MY_DECODER.SEQ_LANG_TYPE,
            #"clip_query_type_indec": cfg.MODEL.MY_DECODER.CLIP_QUERY_INDEC,
            #"decoder_normalize": cfg.MODEL.MY_DECODER.SIMILARITY_NORMALIZE,
            "use_mlp_proj" : cfg.MODEL.USE_MLPPROJ,
            #"norm_before_activation": cfg.MODEL.MY_DECODER.NORM_BEFORE_ACTIVATION,
            "use_instances_annos": cfg.REFERRING.USE_INSTANCES_ANNOS,
            #"use_RLA_layer": cfg.MODEL.MY_DECODER.USE_RLA_LAYER,
            #"bp_bert": cfg.MODEL.MY_DECODER.BP_BERT,
            #"use_bert": cfg.MODEL.USE_BERT,
            #"use_sparse_emb": cfg.MODEL.USE_SPARSE_EMB,
            "clip_layer_idx": cfg.MODEL.CLIP_LAYER_IDX,
            #"type_sparse": cfg.MODEL.MY_DECODER.TYPE_SPARSE,
            #"split_path": cfg.MODEL.MY_DECODER.SPLIT_PATH,
            "gres_option": gres_option,
            "objemb_proj" :cfg.MODEL.EARLY_PROJ_OBJEMB,
            "use_clip_loss": cfg.MODEL.USE_CLIP_TXT
        }

    def _freeze(self):
        from itertools import chain
        freeze_lst = chain(self.sam.prompt_encoder.mask_downscaling.parameters(),
                           self.sam.mask_decoder.iou_prediction_head.parameters(),
                           self.sam.mask_decoder.iou_prediction_head.parameters(), self.sam.prompt_encoder.no_mask_embed.parameters(),
                           self.sam.prompt_encoder.not_a_point_embed.parameters(), self.sam.prompt_encoder.point_embeddings.parameters())
        for param in freeze_lst:
            param.requires_grad = False
       
        for params in self.text_encoder.parameters():
            params.requires_grad = True

        if self.txt2txt_io in ["bert_coco_class", "bert_cls"]:
            for param in self.clip_mlpl.parameters():
                param.requires_grad = False

        if not self.use_clip_loss:
            for param in chain(self.decoder.to_logits.parameters()):
                param.requires_grad = False
            self.decoder.logit_scale_condsim.requires_grad = False


    @property
    def device(self):
        return next(self.parameters()).device

    def get_coco_classes_tokens(self):
        coco_tokens = []
        lang_masks = []
        for noun in coco_classeslst_bert:
            padded_input_ids = [0] * self.max_tokens
            attention_mask = [0] * self.max_tokens
            input_ids = self.tokenizer.encode(text=noun, add_special_tokens=True)
            input_ids = input_ids[:self.max_tokens]
    
            attention_mask[:len(input_ids)] = [1] * len(input_ids)
            padded_input_ids[:len(input_ids)] = input_ids
            input_token = torch.tensor(padded_input_ids).unsqueeze(0)
            lang_mask = torch.tensor(attention_mask).unsqueeze(0)
            coco_tokens.append(input_token)
            lang_masks.append(lang_mask)
        coco_tokens = torch.cat(coco_tokens, dim=0)
        lang_masks = torch.cat(lang_masks, dim=0)
        return coco_tokens, lang_masks
    def forward(self, batched_inputs):
        for x in batched_inputs:
            for key in ['image']:
                x[key] = x[key].to(self.device)

        # CLIP
        clip_cls_t_emb, clip_seq_t_emb, activations, _, _, clip_mask = self.clip_dense(batched_inputs, extract_layers=[self.clip_layer_idx])
        clip_mask = clip_mask.unsqueeze(-1)
        idx = [coco_classeslst_bert.index(x['coco_class']) for x in batched_inputs]
        coco_class_emb = self.text_embeddings[idx, :] #self.clip_mlpl(coco_class_emb)
        coco_class_emb = self.clip_mlpl(coco_class_emb)
        #if self.seq_lang_type in ["clip","bertclip", "clipbert", "cat_clipbert", "clipbert_coco"]:
        clip_cls_768 = clip_seq_t_emb[:, 0,:]
        clip_seq_t_emb = self.clip_mlpl(clip_seq_t_emb)
        clip_cls_t_emb = clip_seq_t_emb[:, 0,:].unsqueeze(1)#self.clip_mlpl(clip_cls_t_emb)
        clip_src = activations[-1][1:] # NOTE　clip cls使ってない
        clip_src = einops.rearrange(clip_src, 'l b f -> b l f')
     
        # clip_seq_t_emb: bz l, f
        # coco_emb: bz f
        lang_feat_dict = {
            'clip': clip_seq_t_emb,
            'clip_cls': clip_cls_t_emb,
            'clip_cls_768': clip_cls_768,
            'clip_coco_classes': self.text_embeddings
        }


        lang_mask_dict = {
            'clip': clip_mask,
            'clip_cls': None,
            #'bert_coco':torch.cat([additional_mask_coco, lang_mask],dim=1)
        }

        # BERT
        lang_emb = [x['lang_tokens'].to(self.device) for x in batched_inputs]
        lang_emb = torch.cat(lang_emb, dim=0)
        lang_mask = [x['lang_mask'].to(self.device) for x in batched_inputs]
        lang_mask = torch.cat(lang_mask, dim=0) # b len 1

        lang_feat = self.text_encoder(lang_emb, attention_mask=lang_mask)[0] # B, Nl, 768
        lang_feat = self.lang_mlp(lang_feat)# just for projection
        #lang_feat = lang_feat.permute(0, 2, 1)  # (B, 256, N_l) to make Conv1d happy
        lang_mask = lang_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        lang_feat_dict.update({'bert':lang_feat, 'bert_cls': lang_feat[:, 0].unsqueeze(1)})
        lang_mask_dict.update({'bert': lang_mask})
        if self.learn_coco_bert_class:
            bert_coco_classes_feat = self.text_encoder(input_ids=self.bert_coco_classes_inputs_input_ids,
                                                    token_type_ids=self.bert_coco_classes_inputs_token_type_ids, 
                                                    attention_mask=self.bert_coco_classes_inputs_attention_mask)[0]
            bert_coco_classes_feat = self.lang_mlp(bert_coco_classes_feat)
            lang_feat_dict.update({'bert_coco_classes': bert_coco_classes_feat})

        t_emb = lang_feat_dict["bert"]
        lang_mask = lang_mask_dict["bert"]


        clip_attn = self.adaptor_clip(clip_src, t_emb.permute(0, 2, 1), lang_mask)
        features = self.backbone_clip(clip_attn)
        mask_features, _, multi_scale_features = self.clip_pixel_decoder.forward_features(features)
       
            
       
        if self.is_pickle:
            image_embeddings = torch.cat([x['sam_embs']['img_emb'] for x in batched_inputs], dim=0).to(self.device)
        else: 
            input_images = torch.stack([self.sam.preprocess(x["image"]) for x in batched_inputs], dim=0)
            # vitは固めて運用するので特徴抽出したら勾配消す
            image_embeddings, _ = self.get_visual_embs(input_images)
        
        x_sam = self.backbone_sam(image_embeddings)
        #if self.sam_pixel_decoder:
        #    mask_features_sam, _, multi_scale_features_sam = self.sam_pixel_decoder.forward_features(x_sam)
        #    image_embeddings = mask_features_sam
        #else:
        multi_scale_features_sam = [_x for k, _x in x_sam.items()][::-1]
        mask_features_sam = image_embeddings

        
        predictions = self.decoder(multi_scale_features, multi_scale_features_sam, mask_features_sam, lang_feat_dict) # cond emb: l bz f

        lang_feat_dict.update({'obj_emb': predictions['sparse_emb']})
        cond_emb = lang_feat_dict["bert"]
        if self.txt2txt is not None:
            if self.txt2txt_io == 'bert_coco':
                txt2txt_in = coco_class_emb.unsqueeze(0)
            elif self.txt2txt_io == 'clip_cls': # GRES対応のため、coco_classから離れたかった。
                txt2txt_in = clip_cls_t_emb.permute(1, 0, 2)
            elif self.txt2txt_io == 'bert_cls':
                txt2txt_in = lang_feat_dict["bert_cls"].permute(1, 0, 2)
            else:
                NotImplementedError('self.txt2txt_io:', + self.txt2txt_io)
            #elif self.txt2txt_io == 'bert_coco_class':
            #    idx = [coco_classeslst_bert.index(x['coco_class']) for x in batched_inputs]
            #    txt2txt_in = self.coco_class_bert_feat[idx, :].unsqueeze(0)
            #elif self.txt2txt_io == 'clip_cls_v':
            #    clip_cls_v =  self.clip_cls_v_proj(activations[-1][0])
            #    txt2txt_in = clip_cls_v.unsqueeze(0)
            #if hasattr(self, "objemb_proj") and self.cond_lang == 'obj_emb':
            #   cond_emb = self.objemb_proj(cond_emb)
            cond_emb = self.txt2txt(txt2txt_in, cond_emb.permute(1, 0, 2))
        else:
            cond_emb = cond_emb.permute(1, 0, 2)
        
        sparse_emb = self.empty_sparse_emb.repeat(len(batched_inputs), 1, 1)

        sparse_emb = torch.cat([sparse_emb, cond_emb.permute(1, 0, 2)], dim=1)
       

        dense_clip = mask_features
        pred_masks = []    
        for i in range(len(batched_inputs)):  
            dense_sam = None
            dense_clip_in = dense_clip[i].unsqueeze(0) if dense_clip is not None else None

            sparse_emb_in = sparse_emb[i].unsqueeze(0)
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                pred_embeds=sparse_emb_in,
                text_embeds=None,
                additional_texts_embs=None,
                clip_visual_dense=dense_clip_in,
                sam_visual_dense=dense_sam,
                no_mask_embed=self.use_nomask_embed_weight
            )
            #sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            # NANはここで発生 dense sparse入力ともに問題なし。
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0), # NOTE imageembeddingが含まれるため。
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            _pred_mask = self.sam.postprocess_masks(
                low_res_masks,
                input_size=self.img_size,
                original_size=self.img_size,
            )
            pred_masks.append(_pred_mask[:, 0])
        
        if self.training:
            targets = self.prepare_targets(batched_inputs)
            losses = self.criterion(predictions, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k] * self.hungarian_weight
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            #loss_hungarian = sum(losses.values())
            gt_mask_merged = torch.stack([x["gt_mask_merged"] for x in targets]).to(self.device)
                
            if self.gres_option['nt_label']:
                nt_loss_dict = self.nt_loss(predictions, targets)
                losses.update(nt_loss_dict)
            sam_loss_dict = self.sam_mask_loss(pred_masks, gt_mask_merged, batched_inputs)
            losses.update(sam_loss_dict)
            return losses
        else:  
            mask_pred_results = torch.stack(pred_masks)#outputs["pred_masks"]
            # upsample masks

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            processed_results = []
            for cnt, (mask_pred_result, input_per_image) in enumerate(
                zip(mask_pred_results, batched_inputs)
            ):
                processed_results.append({})
                r = retry_if_cuda_oom(self.refer_inference)(mask_pred_result)
                processed_results[-1]["ref_seg"] = r
                #processed_results[-1]["nt_label"] = nt
                if 'nt_label' in predictions.keys():
                    processed_results[-1]["nt_label"] = predictions['nt_label'][cnt].sigmoid()
                 
                    # processed_results[-1]["nt_label"] = 1 - predictions['nt_label'][0, cnt].sigmoid()
            return processed_results
    
    def sam_mask_loss(self, pred_masks, gt_mask_merged, batched_inputs):
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_mask_merged[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            is_empty = batched_inputs[batch_idx]['empty']
            empty_scale = 0. if is_empty else 1. # nt target時のmasklossが幸るため。
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask.float(), num_masks=gt_mask.shape[0])
                * gt_mask.shape[0] * empty_scale
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask.float(), num_masks=gt_mask.shape[0])
                * gt_mask.shape[0] * empty_scale
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)


        loss_dict = {
            'main_mask_bce_loss': mask_bce_loss,
            'main_mask_dice_loss': mask_dice_loss,
        }
        return loss_dict

    def nt_loss(self, predictions, targets):
        losses_dict = dict()
        target_nts = torch.stack([x['empty'] for x in targets]).to(self.device)
        gt_mask_merged = [x["gt_mask_merged"].to(self.device) for x in targets]
        nt_loss = ce_loss_jit(predictions['nt_label'], target_nts, self.nt_weight) * self.nt_label_weight
       
        losses_dict.update({'nt_loss': nt_loss})
        # minimap loss cals
        if "mini_map" in predictions.keys():
            src_minimap = predictions["mini_map"].permute(0,2,1)
            target_minimap = torch.cat(gt_mask_merged).unsqueeze(1).float()
            target_minimap = F.interpolate(target_minimap, (10, 10), mode='bilinear', align_corners=False).flatten(start_dim=1)
            minimap_loss = ce_loss_jit(src_minimap, target_minimap.squeeze(1).long(), self.nt_weight) * self.nt_label_weight
            losses_dict.update({'minimap_loss':minimap_loss})
        return losses_dict

    def prepare_targets(self, batched_inputs):
        h_pad, w_pad = self.img_size
        new_targets = []

        for data_per_image in batched_inputs:
            # pad instances
            instances_key = 'instances_coco' if self.use_instances_annos else "instances"
            targets_per_image = data_per_image[instances_key].to(self.device)
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            is_empty = torch.tensor(data_per_image['empty'], dtype=targets_per_image.gt_classes.dtype
, device=targets_per_image.gt_classes.device)
            target_dict = {
                    "labels": targets_per_image.gt_classes,
                    "masks": gt_masks,
                    "empty": is_empty,
                    "sent": data_per_image['sentence']['raw'] # for debug
                }
            if data_per_image["gt_mask_merged"] is not None:
                target_dict["gt_mask_merged"] = data_per_image["gt_mask_merged"].to(self.device)

            new_targets.append(target_dict)
        return new_targets



    def get_visual_embs(self, pixel_values: torch.FloatTensor, extract_layers=[]):
        with torch.no_grad():
            image_embeddings_list = []
            intermidiate_lst = []
            for i in range(pixel_values.shape[0]):
                #torch.cuda.empty_cache()
                image_embeddings, intermidiate = self.sam.image_encoder.vit_forward(
                    pixel_values[i].unsqueeze(0), extract_layers
                    
                )
                image_embeddings_list.append(image_embeddings)
                intermidiate_lst.append(intermidiate)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
            if len(extract_layers) > 0:
                intermidiate = {key: torch.cat([x[key] for x in intermidiate_lst], dim=0) for key in intermidiate.keys()}
            else:
                intermidiate = None
            return image_embeddings, intermidiate
        

    def refer_inference(self, mask_pred):
        mask_pred = mask_pred.sigmoid()
        return mask_pred


    def load_text_embedding(self, dataset_name, model,  device):
        class_names = [x["name"] for x in COCO_CATEGORIES] + ["background"]
        return self.load_coco_text_embedding(class_names, model,  device).float().detach()
       

    def load_coco_text_embedding(self, classes, model, device):
        class_map = {
            "door-stuff": ["door"],
            "floor-wood": ["wood floor"],
            "mirror-stuff": ["mirror"],
            "wall-brick": ["brick wall"],
            "wall-stone": ["stone wall"],
            "wall-tile": ["wall tile"],
            "wall-wood": ["wood wall"],
            "water-other": ["water"],
            "window-blind": ["window blind"],
            "window-other": ["window"],
            "tree-merged": ["branch", "tree", "bush", "leaves"],
            "fence-merged": ["cage", "fence", "railing"],
            "ceiling-merged": ["ceiling tile", "ceiling"],
            "sky-other-merged": ["clouds", "sky", "fog"],
            "cabinet-merged": ["cupboard", "cabinet"],
            "table-merged": ["desk stuff", "table"],
            "floor-other-merged": ["marble floor", "floor", "floor tile"],
            "pavement-merged": ["stone floor", "pavement"],
            "mountain-merged": ["hill", "mountain"],
            "grass-merged": ["moss", "grass", "straw"],
            "dirt-merged": ["mud", "dirt"],
            "paper-merged": ["napkin", "paper"],
            "food-other-merged": ["salad", "vegetable", "food"],
            "building-other-merged": ["skyscraper", "building"],
            "rock-merged": ["stone", "rock"],
            "wall-other-merged": ["wall", "concrete wall", "panel wall"],
            "rug-merged": ["mat", "rug", "carpet"],
        }

        def get_class_embedding(classes, model, device):  # coco
            class_embedding = []
            for class_name in classes:
                if class_name in class_map.keys():
                    class_name = class_map[class_name]
                text_embedding = model.encode_text(clip.tokenize(class_name).to(device))
                text_embedding = text_embedding.mean(dim=0, keepdim=True)
                class_embedding.append(text_embedding)
            class_embedding = torch.concat(class_embedding, dim=0).float().detach()
            class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
            return class_embedding
        
        return get_class_embedding(classes, model, device)
        
    def load_ade20k_text_embedding(self, classes, model, device):
        class_embedding = []
        class_names = [c.split(", ") for c in classes]
        for names in class_names:
            text_embedding = model.encode_text(clip.tokenize(names).to(device))
            text_embedding = text_embedding.mean(dim=0, keepdim=True)
            class_embedding.append(text_embedding)
        class_embedding = torch.concat(class_embedding, dim=0).float().detach()
        class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
        return class_embedding
    
    def load_pascal_context_text_embedding(self, classes, model, device):
        tokens = clip.tokenize(classes).to(device)
        class_embedding = model.encode_text(tokens).float().detach()
        class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
        return class_embedding
    
    def load_coco_ins_text_embedding(self, classes, model, device):
        tokens = clip.tokenize(classes).to(device)
        class_embedding = model.encode_text(tokens).float().detach()
        class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
        return class_embedding