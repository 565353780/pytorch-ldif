# Pytorch LDIF

## Install

```bash
./setup.sh

conda create -n ldif python=3.7
conda activate ldif
pip install torch torchvision torchaudio \
      --extra-index-url https://download.pytorch.org/whl/cu113
pip install wandb tqdm cython pytz python-dateutil \
      trimesh scipy scikit-image shapely jellyfish \
      vtk seaborn h5py opencv-python

./build.sh
```

## Dataset

### Pix3D

```bash
http://pix3d.csail.mit.edu/
```

and save it to

```bash
~/scan2cad/data/pix3d/metadata
```

then, run

```bash
python preprocess.py
```

## Train & Test

```bash
python demo.py
```

## Enjoy it~

