from typing import List
import torch, clip
from torch import nn
from torch.nn import functional as nnf
from torchvision.transforms.transforms import Normalize
from torch.nn import functional as F
from detectron2.utils.registry import Registry
from detectron2.config import configurable
CLIP_REGISTRY = Registry("CLIP")
def forward_multihead_attention(x, b, with_aff=False, attn_mask=None):
    """ 
    Simplified version of multihead attention (taken from torch source code but without tons of if clauses). 
    The mlp and layer norm come from CLIP.
    x: input.
    b: multihead attention module. 
    """

    x_ = b.ln_1(x)
    q, k, v = nnf.linear(x_, b.attn.in_proj_weight, b.attn.in_proj_bias).chunk(3, dim=-1)
    tgt_len, bsz, embed_dim = q.size()

    head_dim = embed_dim // b.attn.num_heads
    scaling = float(head_dim) ** -0.5

    q = q.contiguous().view(tgt_len, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)

    q = q * scaling

    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) #  n_heads * batch_size, tokens^2, tokens^2
    if attn_mask is not None:


        attn_mask_type, attn_mask = attn_mask
        n_heads = attn_output_weights.size(0) // attn_mask.size(0)
        attn_mask = attn_mask.repeat(n_heads, 1)
        
        if attn_mask_type == 'cls_token':
            # the mask only affects similarities compared to the readout-token.
            attn_output_weights[:, 0, 1:] = attn_output_weights[:, 0, 1:] * attn_mask[None,...]
            # attn_output_weights[:, 0, 0] = 0*attn_output_weights[:, 0, 0]

        if attn_mask_type == 'all':
            # print(attn_output_weights.shape, attn_mask[:, None].shape)
            attn_output_weights[:, 1:, 1:] = attn_output_weights[:, 1:, 1:] * attn_mask[:, None]
        
    
    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = b.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + b.mlp(b.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    else:
        return x


def build_clip(cfg):
    name = cfg.MODEL.CLIP.CLIP_MODULE_NAME
    return CLIP_REGISTRY.get(name)(cfg)

@CLIP_REGISTRY.register()
class CLIPWrapper(nn.Module):
    @configurable
    def __init__(self, *, version, input_resolution):
        super().__init__()
        # prec = torch.FloatTensor
        self.clip_model, _ = clip.load(version, device='cpu', jit=False)
        #self.model = self.clip_model.visual
        self.clip_prep_img = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.input_resolution = input_resolution
        self._freeze()
    @classmethod
    def from_config(cls, cfg):
        version = cfg.MODEL.CLIP.VERSION
        resolution = {
            "ViT-B/16": 224,
            "ViT-L/14": 224,
            "RN101": 224,
            "ViT-L/14@336px": 336
        }
        input_resolution = resolution[version]
        return {
            'input_resolution': input_resolution,
            'version': version
        }
    @property
    def device(self)-> torch.device:
        return next(self.parameters()).device
    def _freeze(self):
        #"SANを参考に"
        for name, param in self.named_parameters():
            param.requires_grad = False
    def forward(self, batched_input, extract_layers=(), skip=False, mask=None):
        with torch.no_grad():
            sents = [x['caption'] for x in batched_input]
            coco_class = [x["coco_class"] for x in batched_input]
            ref_emb, seq_emb, clip_seq_mask = self.get_clip_text_features(sents)
            coco_class_emb, _, _ = self.get_clip_text_features(coco_class)
            img_size = self.input_resolution
            x = [F.interpolate(x['image'].unsqueeze(0) / 255., (img_size, img_size), mode="bicubic") for x in batched_input]
            x = torch.cat(x, dim=0)
            x = self.clip_prep_img(x)
            x = self.clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND

            activations, affinities = [], []
            for i, res_block in enumerate(self.clip_model.visual.transformer.resblocks):
                x, aff_per_head = forward_multihead_attention(x, res_block, with_aff=True)

                if i in extract_layers:
                    affinities += [aff_per_head]

                    #if self.n_tokens is not None:
                    #    activations += [nnf.interpolate(x, inp_size, mode='bilinear', align_corners=True)]
                    #else:
                    activations += [x]

                if len(extract_layers) > 0 and i == max(extract_layers) and skip:
                    print('early skip')
                    break                
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.visual.ln_post(x[:, 0, :])
            if self.clip_model.visual.proj is not None:
                x = x @ self.clip_model.visual.proj
            return ref_emb, seq_emb, activations, affinities, coco_class_emb, clip_seq_mask

    
    def get_clip_text_features(self, caption_list: List[str]):
        text = clip.tokenize(caption_list).to(self.device)
        lang_mask = (text >  0).int()
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        seq_feat = x
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        x = x / x.norm(dim=1, keepdim=True)
        return x, seq_feat, lang_mask