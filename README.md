# Real_R2R_denoising
Unsupervised R2R Denoising for Real Image Denosing

This repository is an PyTorch implementation of the paper [Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising](https://openaccess.thecvf.com/content/CVPR2021/html/Pang_Recorrupted-to-Recorrupted_Unsupervised_Deep_Learning_for_Image_Denoising_CVPR_2021_paper.html). The network we adopted is  [DnCNN](https://ieeexplore.ieee.org/document/7839189) and our implementation is based on [DnCNN-PyTorch](https://github.com/SaoYan/DnCNN-PyTorch). We give the author credit for the implementation of DnCNN. We give the author credit for the implementation of DnCNN. The Gaussian denoising version is available [R2R](https://github.com/PangTongyao/Recorrupted-to-Recorrupted-Unsupervised-Deep-Learning-for-Image-Denoising)

## 1.Dependencies

- [Matlab](https://www.mathworks.com/products/matlab.html) (For training patch generation)
- [PyTorch](http://pytorch.org/)
- OpenCV for Python
- [scikit-image](https://scikit-image.org/)


## 2. Download and generate SIDD training data

Here we adapt SIDD Medium data for training. The training data and validation data can be download in [SIDD website](https://www.eecs.yorku.ca/~kamel/sidd/). After downloading, move both the "Data" folder and "noise_level_functions.csv" of training data to "sidd_dataset" folder.

To generate training patch, please run the following commands.  

```
cd gen_data
bash gen_sidd.sh
```

## 3. Run Experiments on SIDD real noise removal 
Training

```
python train_sidd_dncnn.py --gpu 0 
```

The pretrained model is available on './experiments/pre_trained.pth'
Validation

```
python test_sidd_dncnn.py --gpu 0 --phase validation --model_path 'path of pretrained model' --val_path 'path of validation data'
```

Test

```
python test_sidd_dncnn.py --gpu 0 --phase test --model_path 'path of pretrained model' --val_path 'path of test data'
```

Any other NNs can be adapted here by changing the model architecture. 

## 4.Citation

```
@InProceedings{Pang_2021_CVPR,
    author    = {Pang, Tongyao and Zheng, Huan and Quan, Yuhui and Ji, Hui},
    title     = {Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2043-2052}
}
```