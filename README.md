# Learning Event Guided High Dynamic Range Video Reconstruction, CVPR 2023

#### Yixin Yang, Jin Han, Jinxiu Liang, Imari Sato, Boxin Shi.

## Dependency 

```shell
conda create -n HDRev python=3.7
conda activate HDRev
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
sudo apt install libopenexr-dev zlib1g-dev
conda install -c conda-forge openexr-python
pip install -r requirements.txt
```

## Testing

We provide the testing code for both stacked voxel event data and text event data (t, x, y, p).

### stacked voxel event data

one of the dataset example are shown in datasets/synthetic.

```shell
python test.py --dataroot datasets/synthetic
```


### text event data

one of the dataset example are shown in datasets/real.

```shell
python test.py --dataroot datasets/real/ --test_on_txt
```

## Contact
if you have any question, please send an email to yangyixin93@pku.edu.cn