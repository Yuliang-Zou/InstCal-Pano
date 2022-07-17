# Learning Instance-Specific Adaptation for Cross-Domain Segmentation

Official PyTorch implementation of [Learning Instance-Specific Adaptation for Cross-Domain Segmentation](https://arxiv.org/pdf/2203.16530.pdf) for Panoptic Segmentation.

## Installation
1. Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html). Note that we use [this commit version](https://github.com/facebookresearch/detectron2/tree/ef2c3abbd36d4093a604f874243037691f634c2f). Later version might work but we did not test. It is recommended to use exactly the same version.
2. Using this codebase to replace the `projects/Panoptic-DeepLab` folder. Please also rename this folder so that it has the same name `Panoptic-DeepLab` for correct path loading.
3. Prepare cityscapes data follow the [tutorial](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html#expected-dataset-structure-for-cityscapes).
4. Download [foggy cityscapes](http://people.ee.ethz.ch/~csakarid/SFSU_synthetic/). Since this dataset shares the same ground truth with cityscapes, please create a soft link from `cityscapes/gtFine`.
5. Download off-the-shelf Panoptic-DeepLab checkpoints[1](https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl)[2](https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv/model_final_23d03a.pkl) and put them into `pretrained_model` folder.

You should have the following dataset structure if set correctly.
```bash
- datasets
	- cityscapes
		- leftImg8bit
		- gtFine
	- foggy_cityscapes
		- leftImg8bit_foggy
		- gtFine    # (Please create a soft link to the gtFine in cityscapes)
```

## Training

```bash
cd /path/to/detectron2/projects/Panoptic-DeepLab
python train_net_u.py --config-file configs/Foggy-Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml --num-gpus 1
```

**NOTE:** Replace `train_net_u.py` with `train_net_c.py` if you want to switch from InstCal-U to InstCal-C.

The best checkpoint is usually around 55k iter.

## Evaluation

```bash
cd /path/to/detectron2/projects/Panoptic-DeepLab
python train_net_u.py --config-file configs/Foggy-Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

**NOTE:** Replace `train_net_u.py` with `train_net_c.py` if you want to switch from InstCal-U to InstCal-C.

## Visualization
```bash
cd /path/to/detectron2/projects/Panoptic-DeepLab/demo
python demo_u.py --config-file configs/Foggy-Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml \
--input datasets/foggy_cityscapes/leftImg8bit_foggy/val/munster/*.png --output output/InstCalU/munster/ \
--opts MODEL.WEIGHTS /path/to/model_checkpoint
```

**NOTE:** Replace `demo_u.py` with `demo_c.py` if you want to switch from InstCal-U to InstCal-C.


## Citation

If you find this code useful for your research, please cite our paper.


```
@inproceedings{zou2022learning,
  title={Learning Instance-Specific Adaptation for Cross-Domain Segmentation},
  author={Zou, Yuliang and Zhang, Zizhao and Li, Chun-Liang and Zhang, Han and Pfister, Tomas and Huang, Jia-Bin},
  booktitle={ECCV},
  year={2022}
}
```


Please also cite the Panoptic-DeepLab paper.

```
@inproceedings{cheng2020panoptic,
  title={Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation},
  author={Cheng, Bowen and Collins, Maxwell D and Zhu, Yukun and Liu, Ting and Huang, Thomas S and Adam, Hartwig and Chen, Liang-Chieh},
  booktitle={CVPR},
  year={2020}
}
```

