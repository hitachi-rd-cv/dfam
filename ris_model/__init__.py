from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_refcoco_config, add_myris_config

# dataset loading
from .data.dataset_mappers.refcoco_mapper import RefCOCOMapper

# models
#from .RIS import RIS
#from .RIS_temp import RIStemp
#from .RIS_temp2 import RIStemp2
#from .RISwoDec import RISwoDec
#from .vitdet import vitdet
from .RIS_wacv25 import RIS_wacv25

# evaluation
#from .evaluation.refer_evaluation import ReferEvaluator # GRESで使うかも
from .evaluation.my_evaluation import GroundingEvaluator

