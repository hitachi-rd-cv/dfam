from ..module.attention import CrossAttentionLayer, FFNLayer, SelfAttentionLayer, MyCrossAttentionLayer
from torch import nn
from detectron2.utils.registry import Registry
from detectron2.config import configurable
from ...my_misc import PositionalEncoding
TXT2TXT_REGISTRY = Registry("TXT2TXT")
def build_txt2txt(cfg):
    name = cfg.MODEL.TXT2TXT.TXT2TXT_MODULE_NAME
    return TXT2TXT_REGISTRY.get(name)(cfg)

@TXT2TXT_REGISTRY.register()
class Txt2txt(nn.Module):
    @configurable
    def __init__(self,*,
                hidden_dim: int,
                nheads: int,
                dim_feedforward: int,
                pre_norm: bool,
                attn_only:bool,
                 ):
        super().__init__()
        if attn_only:
            self.ca = MyCrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
        else:
            self.ca = CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
        self.fcs =  FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
    @classmethod
    def from_config(cls, cfg):
        return {
            'hidden_dim': cfg.MODEL.MY_DECODER.EARLY_DIM,
            'nheads': 8,
            'attn_only': cfg.MODEL.TXT2TXT.ATTN_ONLY,
            'dim_feedforward':  cfg.MODEL.MY_DECODER.DIM * 4,
            'pre_norm': False,
        }     
    def forward(self, coco_emb, seq_emb):
        emb = self.ca(coco_emb, seq_emb)
        emb = self.fcs(emb)
        return emb

@TXT2TXT_REGISTRY.register()
class cls2word(nn.Module):
    @configurable
    def __init__(self,
                hidden_dim: int,
                nheads: int,
                dim_feedforward: int,
                pre_norm: bool,
                 ):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=0.0)
        self.fcs =  FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
    @classmethod
    def from_config(cls, cfg):
        return {
            'hidden_dim': cfg.MODEL.MY_DECODER.DIM,
            'nheads': 8,
            'dim_feedforward':  cfg.MODEL.MY_DECODER.DIM * 4,
            'pre_norm': False,
        }         
    def forward(self, cls_token, seq_emb):
        
        emb = self.multihead_attn(cls_token, seq_emb, seq_emb)[0]
        emb = self.fcs(emb)
        return emb

class Txt2txt_casa(nn.Module):
    def __init__(self,
                hidden_dim: int,
                nheads: int,
                dim_feedforward: int,
                pre_norm: bool,
                pe:bool=False
                 ):
        super().__init__()
        if pe:
            self.pe = PositionalEncoding(hidden_dim, dropout=0.)
        self.ca = CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
        self.sa = SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
        )
        self.fcs =  FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
                
    def forward(self, seq1, seq2):
        if hasattr(self, 'pe'):
            seq1 = self.pe(seq1)
            seq2 = self.pe(seq2)
        emb = self.ca(seq1, seq2)
        emb = self.sa(emb)
        emb = self.fcs(emb)
        return emb

