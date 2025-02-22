from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from transformers import BertTokenizer
import torch, math
from torch import Tensor, nn
def load_coco_classes_bert_textlst():
    class_names = [x["name"] for x in COCO_CATEGORIES] + ["background"]
    return load_coco_textlst(class_names)

def load_coco_textlst(classes):
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
    coco_classes = []
    for class_name in classes:
        if class_name in class_map.keys():
            class_name = class_map[class_name]
            class_name = (' ').join(class_name)
        coco_classes.append(class_name)
    
    return coco_classes

coco_classeslst_bert = load_coco_classes_bert_textlst()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)