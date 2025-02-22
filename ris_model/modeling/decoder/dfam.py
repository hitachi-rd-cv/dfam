from torch import nn
import torch, math
from ..module.attention import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from ..module.position_encoding import PositionEmbeddingSine
from detectron2.layers import Linear, Conv2d
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
class DFAM(nn.Module):
    def __init__(self,
                in_channels,
                hidden_dim: int,
                num_queries: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
                learn_coco_bert_class:bool=False,
                learn_coco_clipcls_class:bool=False,
                gres_option:dict=dict(),
                logit_dim:int=512,
                early_dim:int=256,
                sam_intermidiate_dim:int=768
                 ):
         # positional encoding
        super().__init__()
        self.learn_coco_bert_class = learn_coco_bert_class
        self.learn_coco_clipcls_class = learn_coco_clipcls_class
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_scale_condsim = nn.Parameter(torch.ones([]))
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.SA_layers = nn.ModuleList()
        self.SA_layers2 = nn.ModuleList()
        self.CA_layers = nn.ModuleList()
        self.CA_layers2 = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.transformer_ffn_layers2 = nn.ModuleList()

        for _ in range(self.num_layers):
            self.SA_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.SA_layers2.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.CA_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.CA_layers2.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers2.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        self.decoder_norm1 = nn.LayerNorm(hidden_dim)
        self.decoder_norm2 = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        
        self.num_feature_levels = dec_layers
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        self.input_proj_sam = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
            if sam_intermidiate_dim != hidden_dim:
                self.input_proj_sam.append(Conv2d(sam_intermidiate_dim, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj_sam[-1])
            else:
                self.input_proj_sam.append(nn.Sequential())
        # output FFNs
        self.mask_classification = True
        if self.mask_classification:
            self.v_embed = nn.Linear(hidden_dim, hidden_dim)
            self.to_logits = nn.Linear(hidden_dim, logit_dim)
            self.to_logits_bert = nn.Linear(hidden_dim, early_dim)
        #self.nt_embed = MLP(hidden_dim, hidden_dim, 2, 2) # nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.reduction_type = gres_option["reduction"]
        self.nt_label = gres_option['nt_label']
        if self.nt_label:
            #self.gres_minimap = nn.Linear(hidden_dim, 2)
            self.nt_embed = MLP(hidden_dim, hidden_dim, 2, 2)
    def forward(self, clip_src, x_sam, sam_embs, lang_feat_dict):
        #bert_feat = lang_feat_dict['bert_cls']
        clip_cls = lang_feat_dict['clip_cls_768']
        clip_coco_classes = lang_feat_dict['clip_coco_classes']
        # bert b f l
        #bert_feat = bert_feat.permute(2, 0, 1)
        # x is a list of multi-scale feature
        mask_features = sam_embs#clip_src[-1]
        x = clip_src
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        src_sam = []
        pos_sam = []
        size_list_sam = []
        if self.learn_coco_clipcls_class:
            bz = clip_cls.shape[0]
            clip_cls = clip_cls.unsqueeze(1)
            clip_coco_classes = clip_coco_classes.unsqueeze(0).repeat(bz, 1, 1)
            clip_text_emb = torch.cat([clip_cls, clip_coco_classes], dim=1)
        else:
            clip_text_emb = torch.cat([clip_cls, clip_coco_classes])

        bert_cls = lang_feat_dict['bert_cls']
        bz = bert_cls.shape[0]
        if self.learn_coco_bert_class:
            bert_coco_classes = lang_feat_dict["bert_coco_classes"][:, 0].unsqueeze(0) # nclass seq(7) 768 > 1 nclass 768
            bert_coco_classes = bert_coco_classes.repeat(bz, 1, 1)
            bert_cls = torch.cat([bert_cls, bert_coco_classes], dim=1) # bz, nclass+1(bertcls), f
        else:
            bert_cls = bert_cls # b 1 f


        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            size_list_sam.append(x_sam[i].shape[-2:])
            pos_sam.append(self.pe_layer(x_sam[i], None).flatten(2))
            src_sam.append(self.input_proj_sam[i](x_sam[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
            pos_sam[-1] = pos_sam[-1].permute(2, 0, 1)
            src_sam[-1] = src_sam[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape
        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        

        predictions_class = []
        predictions_class_bert = []
        predictions_mask = []
        predictions_weighted_emb = []
        # prediction heads on learnable query features
        all_mask = self.forward_prediction_heads_mask(
            output, mask_features)

        logit, all_mask, obj_emb, bertcls_logit = self.forward_prediction_heads_semantic(
            output, clip_text_emb, all_mask, bert_cls)
        attn_mask_clip = self.attn_mask_resize(all_mask, size_list[0])
        attn_mask_sam = self.attn_mask_resize(all_mask, size_list_sam[0])
        #weighted_emb = torch.einsum('qbc,bqn->bcn', output, sim_map)
        #weighted_emb = weighted_emb.permute(0, 2, 1)
        predictions_mask.append(all_mask)
        predictions_class.append(logit)
        predictions_class_bert.append(bertcls_logit)
        # ReLA is applied multiple times for perfromance
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            attn_mask_clip[torch.where(attn_mask_clip.sum(-1) == attn_mask_clip.shape[-1])] = False
            attn_mask_sam[torch.where(attn_mask_sam.sum(-1) == attn_mask_sam.shape[-1])] = False
            output = self.CA_layers[i](
                output, src_sam[level_index],
                memory_mask=attn_mask_sam,
                memory_key_padding_mask=None,
                pos=pos_sam[level_index], query_pos=query_embed
            )

            output = self.SA_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # Postprocessing
            output = self.transformer_ffn_layers[i](output)
            output_clip = output
        
            output_clip = self.CA_layers2[i](
                output_clip, src[level_index],
                memory_mask=attn_mask_clip,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=None
            )

            output_clip = self.SA_layers2[i](
                output_clip, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # Postprocessing
            output_clip = self.transformer_ffn_layers2[i](output_clip)

            all_mask = self.forward_prediction_heads_mask(
                output, mask_features)
            
            
            logit, all_mask, obj_emb, bertcls_logit = self.forward_prediction_heads_semantic(
                output_clip, clip_text_emb, all_mask, bert_cls)
            
            attn_mask_sam = self.attn_mask_resize(all_mask, size_list_sam[(i + 1) % self.num_feature_levels])
            attn_mask_clip = self.attn_mask_resize(all_mask, size_list[(i + 1) % self.num_feature_levels])
            # Predictions of all passes are recorded, but only the last output is used in this code
            predictions_weighted_emb.append(obj_emb)
            predictions_mask.append(all_mask)
            predictions_class.append(logit)
            predictions_class_bert.append(bertcls_logit)
        out = {
            "sparse_emb": predictions_weighted_emb[-1],
            'pred_masks': predictions_mask[-1],
            'pred_logits':predictions_class[-1],
            'pred_logits_bertcls':predictions_class_bert[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_class_bert if self.mask_classification else None
            )
        }
        if self.nt_label:
            #bertcls_logit_diag = torch.diagonal(bertcls_logit, dim1=0, dim2=2) # bz, 100
            #nt_label = torch.mean(bertcls_logit_diag, dim=0)
            nt_label = self.nt_embed(obj_emb)
            if self.reduction_type == 'max':
                nt_label, _ = torch.max(nt_label, dim=1)
            elif self.reduction_type == 'mean':
                nt_label = torch.mean(nt_label, dim=1)
            else:
                raise NotImplementedError
            out.update({'nt_label':nt_label})
        return out


    
    def forward_prediction_heads_mask(self, output, mask_features):
        region_features = self.decoder_norm1(output)
        region_features = region_features.transpose(0, 1)
        region_embed = self.mask_embed(region_features)
        all_mask = torch.einsum("bqc,bchw->bqhw", region_embed, mask_features)
        return all_mask
    
    def attn_mask_resize(self, all_mask, target_size):
        attn_mask = F.interpolate(all_mask, size=target_size, mode="bilinear", align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        return attn_mask
    
        # 比較向けの特徴射影
    def forward_prediction_heads_semantic(self, output, clip_text_emb, all_mask, bert_feat):
        region_features = self.decoder_norm2(output)
        region_features = region_features.transpose(0, 1)
        obj_embed = self.v_embed(region_features)
        v_emb = obj_embed
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
       
        # for clip text
        logits = self.to_logits(obj_embed)
        logits = logits / (logits.norm(dim=-1, keepdim=True)+1e-7)
        clip_text_emb = clip_text_emb / (clip_text_emb.norm(dim=-1, keepdim=True) + 1e-7)
        if self.learn_coco_clipcls_class:
            clip_logits = self.logit_scale_condsim.exp() * logits @ clip_text_emb.transpose(1, 2)
        else:
            clip_logits = self.logit_scale_condsim.exp() * logits @ clip_text_emb.transpose(0, 1)

        # for bert text
        logits_bert = self.to_logits_bert(obj_embed)
        logits_bert = logits_bert / (logits_bert.norm(dim=-1, keepdim=True)+1e-7)
        bert_feat = bert_feat / (bert_feat.norm(dim=-1, keepdim=True) + 1e-7)
        if self.learn_coco_bert_class:
            bertcls_logits = self.logit_scale.exp() * logits_bert @ bert_feat.transpose(1, 2)
        else:
            bertcls_logits = self.logit_scale.exp() * logits_bert @ bert_feat.squeeze(1).transpose(0, 1)
        return clip_logits, all_mask, obj_embed, bertcls_logits
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_bertcls):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_logits_bertcls": c}
                for a, b,c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_bertcls[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]