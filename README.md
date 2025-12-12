# LoGA-Attack: Local Geometry-Aware Adversarial Attack on 3D Point Clouds

## Requirements

* tqdm >= 4.52.0
* numpy >= 1.19.2
* scipy >= 1.6.3
* open3d >= 0.13.0
* torchvision >= 0.7.0
* scikit-learn >= 1.0

## Datasets

Please download the aligned [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) dataset and [ScanObjectNN](https://hkust-vgd.ust.hk/scanobjectnn/) dataset in their point cloud format and unzip them into your own dataset path.

## Example Usage

### Generate adversarial examples by attacking PointNet:

```
python main.py --dataset ModelNet40 --attack_method LoGA --surrogate_model pointnet_cls --target_model pointnet_cls --step_size 0.007 --max_steps 50 --eps 0.16
```
