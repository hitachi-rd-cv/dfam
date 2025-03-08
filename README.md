# Feature Design for Bridging SAM and CLIP toward Referring Image Segmentation (WACV25)

## Preparations
- environment
    * we utilized singularity but you need not to use it. refer to [dfam.def](dfam.def) for the environment. 
- dataset and model preparations
```sh
RIPOGITORY_DIR=<your_directory>
DATA_DIR=<yout_data_directory>
mkdir $DATA_DIR/coco && cd $DATA_DIR/coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip && rm annotations_trainval2014.zip
unzip train2014.zip && rm train2014.zip
# download refcoco
# reference: https://github.com/lichengunc/refer/issues/14#issuecomment-1258318183
mkdir $DATA_DIR/refcoco && cd $DATA_DIR/refcoco
wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
unzip refcoco.zip && rm refcoco.zip
unzip refcoco+.zip && rm refcoco+.zip
unzip refcocog.zip && rm refcocog.zip

cd $RIPOGITORY_DIR
ln -s $DATA_DIR/refcoco/refcoco datasets/refcoco
ln -s $DATA_DIR/refcoco/refcoco+ datasets/refcoco+
ln -s $DATA_DIR/refcoco/refcocog datasets/refcocog
ln -s $DATA_DIR/coco/annotations datasets/images/annotations
ln -s $DATA_DIR/coco/train2014 datasets/images/train2014

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# For grefcoco data, follow below 
# https://github.com/henghuiding/gRefCOCO

```

## Training
```sh
python train_net.py --config-file configs/mine.yaml --num-gpus 4 --dist-url auto OUTPUT_DIR output SOLVER.BASE_LR 5e-5 SOLVER.IMS_PER_BATCH 64 SOLVER.CHECKPOINT_PERIOD 140000 REFERRING.USE_PICKLE False
```

# Comment
- Sorry for the messy code. This is just an initial commit for now.

# Citation
```bib
@inproceedings{kito2025dfam,
      title={Feature Design for Bridging SAM and CLIP toward Referring Image Segmentation}, 
      author={Koichiro Ito},
      year={2025},
      pages={8357--8367},
      booktitle={Proc. of WACV}
}
```
