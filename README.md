# Video Label Refinement
The goal of this project to refine labels for temporal localization datasets.  The dataset that is currently support is ShakeFive2

## Components
This project depends on OpenMMLab https://openmmlab.com/.  It includes redristributed files from those libaries which are allowed under https://github.com/openmm/openmm/blob/master/libraries/sfmt/LICENSE.txt, provided the following copyright is included:
Copyright (c) 2006,2007 Mutsuo Saito, Makoto Matsumoto and Hiroshima University. All rights reserved.

- File downloader: download.py
- Generate signal data: inference.py
- View signal data: main.py
- Label Refinement: refinement.py
- Classification assessment: train_model.py

## How to run
This project depends on PyTorch and Torch Vision, which are not installed automatically.  

### Download the checkpoint files
The checkpoint files were too large to include in the repository
From the within the project director, run
```shell
python download.py
```

### Download the ShakeFive2 dataset
The ShakeFive2 dataset cannot be redistributed and has be downloaded separately.
Download the ShakeFive2 dataset from https://www.projects.science.uu.nl/shakefive/

### Run Signal Generation
The first step is to generate signal data.  Run the command: python inference.py --path=./PathToVideos
```shell
python inference.py --path=./ShakeFive2
```
Calculating signals takes a long time.  For that reason the signal data files are already included in the directory ShakeFive2.  
You may copy the .mp4 and .xml from the ShakeFive2 dataset that was previously downloaded to skip this step

### Label Refinement
To run label 
```shell
python refinement.py --path=./ShakeFive2
```

### Classification Assessment
Classification improvement is assessed by training models on two different sets of labels. 
This program requires a parameter to indicate whether to used the refined or human labels.  
To use the refined labels, set the parameter refined=True, to used the unmodified labels set refined=False.

```shell
python train_model.py --refined=False --train=True
```
The output is in model_performance.csv


### Visualizing Signal Data
Visualizing videos, pose and signal data is helpful for understanding the results.  
Run this command:
```shell
python main.py
```

