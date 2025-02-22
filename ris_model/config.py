# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.BASE_LR_END = 0.0000001

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.LANG_ATT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_refcoco_config(cfg):
    """
    Add config for RefCOCO.
    """
    cfg.DATASETS.REF_ROOT = "refer/data/"
    #cfg.DATASETS.DATASET_NAME = "refcoco"
    #cfg.DATASETS.SPLIT_BY = "unc"

    cfg.REFERRING = CN()
    cfg.REFERRING.BERT_TYPE = "bert-base-uncased"
    cfg.REFERRING.MAX_TOKENS = 35

def add_myris_config(cfg):
    """
    Add config for RefCOCO.
    """
    cfg.REFERRING.USE_PICKLE = False
    cfg.REFERRING.SAM_PICKLE_DIR = "/lustre/home/vap/k-ito/refcocog_pickle2" # "/fs2/groups2/gca50126/71347032/dataset/refcocog_pickle2"
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.MODEL.META_ARCHITECTURE = "RIS_wacv25"
    cfg.MODEL.SAM = CN()
    cfg.MODEL.SAM.VIT_TYPE = 'vit_b'
    cfg.MODEL.CLIP = CN()
    cfg.MODEL.CLIP.CLIP_MODULE_NAME = "CLIPWrapper"
    cfg.MODEL.CLIP.VERSION = "ViT-L/14@336px"
    cfg.MODEL.TXT2TXT = CN()
    cfg.MODEL.TXT2TXT.ATTN_ONLY = False
    cfg.MODEL.TXT2TXT.TXT2TXT_MODULE_NAME = 'Txt2txt' #cls2word
    cfg.MODEL.TXT2TXT.IO = 'bert_coco' #'clipcls_bert'
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["p2", "p3", "p4"]#["layer_3", "layer_7", "layer_11", "layer_15"]
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4"]#["layer_3", "layer_7", "layer_11", "layer_15"]
    #cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 1/2
    #!!cfg.MODEL.SEM_SEG_HEAD.OUT_FEATURES_IF_CLIP = True
    cfg.MODEL.MY_DECODER = CN()
    cfg.MODEL.MY_DECODER.DIM = 256
    cfg.MODEL.MY_DECODER.EARLY_DIM = 256
    #!!cfg.MODEL.MY_DECODER.DUMMY_LANG_MASK = False
    cfg.MODEL.MY_DECODER.CLIP_SCALE_FACTORS = (2.0, 1.0, 0.5)
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES2 = ["p4", "p5", "p6"]#["p3", "p4", "p5"] # 同じ名前のinstanceが立てないdetectron2のconfig仕様のため糞実装
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES2 = ["p4", "p5", "p6"]#["p3", "p4", "p5"]
    cfg.MODEL.MY_DECODER.SAM_SCALE_FACTORS = (1.0, 0.5, 0.25) #(2.0, 1.0, 0.5)
    #!!cfg.MODEL.MY_DECODER.SAM_PIX_DEC = False # RISにはそもそもない
    #!!cfg.MODEL.MY_DECODER.DEEP_SUPERVISION = True
    #!!cfg.MODEL.MY_DECODER.TYPE ='dfam'
    #cfg.MODEL.MY_DECODER.TEXT_COND ='bert_coco' # 機能してない
    #cfg.MODEL.MY_DECODER.IN_IMAGE_EMBEDDING =True# decoderにsamimage embedを入れるかPIXDEC出力にするか。
    #!!cfg.MODEL.MY_DECODER.IS_MATCHER =True# decoderにsamimage embedを入れるかPIXDEC出力にするか。
    # loss weight
    cfg.MODEL.MY_DECODER.MATCHER_WEIGHT = 1.0
    cfg.MODEL.MY_DECODER.CLASS_BERTCLS_WEIGHT = 1.0
    cfg.MODEL.MY_DECODER.DICE_LOSS_WEIGHT = 0.5
    cfg.MODEL.MY_DECODER.BCE_LOSS_WEIGHT = 2.0
    cfg.MODEL.MY_DECODER.NO_OBJECT_WEIGHT = 0.1
    #!!cfg.MODEL.MY_DECODER.TEXT_EMBED_IN_MD = False
    #!!cfg.MODEL.MY_DECODER.ACCEPT_CLIPEMBED = "dense" # dense or sparse
    cfg.MODEL.MY_DECODER.NOMASK_EMBED_WEIGHT = True
    #!!cfg.MODEL.MY_DECODER.IS_TXT2TXT_PE = False
    cfg.MODEL.USE_MLPPROJ = False
    #!!cfg.MODEL.MY_DECODER.PWAM_CLIP = "bert" # RISでは不使用な可能性
    #!!cfg.MODEL.MY_DECODER.PWAM_SAM = None
    #!!cfg.MODEL.MY_DECODER.SEQ_LANG_TYPE = "bert"
    #!!cfg.MODEL.MY_DECODER.COND_LANG_TYPE = "bert"
    #!!cfg.MODEL.MY_DECODER.CLIP_QUERY_INDEC = "same" # clip_cls, same diff
    #!!cfg.MODEL.MY_DECODER.MM_FIRST = True # MMしてからbackbone通す。
    #!!cfg.MODEL.MY_DECODER.NORM_BEFORE_ACTIVATION = False
    #!!cfg.MODEL.MY_DECODER.SIMILARITY_NORMALIZE = "multi_sigmoid"
    #!!cfg.MODEL.MY_DECODER.WEIGHT_TARGET = "output"
    
    #!!cfg.MODEL.MY_DECODER.USE_RLA_LAYER = False
    #!!cfg.MODEL.MY_DECODER.SPLIT_PATH = True # clip samのパスを分けるかどうか
    cfg.REFERRING.USE_INSTANCES_ANNOS = False
    #!!cfg.MODEL.MY_DECODER.BP_BERT = True
    #!!cfg.MODEL.USE_BERT = True
    #!!cfg.MODEL.USE_SPARSE_EMB = True
    cfg.MODEL.CLIP_LAYER_IDX = 23
    #!!cfg.MODEL.MY_DECODER.TYPE_SPARSE = "cond_emb"
    #!!cfg.MODEL.MY_DECODER.MATHER_SIGMOID = False
    cfg.MODEL.BERT_COCO_CLASS = False
    cfg.MODEL.CLIPCLS_COCO_CLASS = False
    cfg.MODEL.CRASSIFICATION_WEIGHT_MASK = False
    cfg.MODEL.USE_BCE2CLIP = False
    cfg.MODEL.EARLY_PROJ_OBJEMB = False
    cfg.MODEL.USE_CLIP_TXT = True
    cfg.GRES = CN()
    cfg.GRES.REDUCTION = "max"
    cfg.GRES.USE_MINIMAP = True
    cfg.GRES.NT_LABEL_WEIGHT = 0.1
    cfg.MODEL.CLIP_CE_V2L = False