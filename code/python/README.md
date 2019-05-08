# Generic Decoding Demo/Python

This is Python code for Generic Decoding Demo.

## Requirements

All scripts are tested with Python 2.7.13.
The following packages are required.

- [bdpy](https://github.com/KamitaniLab/bdpy)
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib (mandatory for creating figures)
- caffe (mandatory if you calculate image and category features by yourself)
- PIL (mandatory if you calculate image and category features by yourself)

## Data organization

All data should be placed in `python/data`.
Data can be obrained from [figshare](https://figshare.com/articles/Generic_Object_Decoding/7387130).
The data directory should have the following files:

    data/ --+-- Subject1.h5 (fMRI data, subject 1)
            |
            +-- Subject2.h5 (fMRI data, subject 2)
            |
            +-- Subject3.h5 (fMRI data, subject 3)
            |
            +-- Subject4.h5 (fMRI data, subject 4)
            |
            +-- Subject5.h5 (fMRI data, subject 5)
            |
            +-- ImageFeatures.h5 (image features extracted with Matconvnet)

Download links:

- [Subject1.mat](https://ndownloader.figshare.com/files/13663487)
- [Subject2.mat](https://ndownloader.figshare.com/files/13663490)
- [Subject3.mat](https://ndownloader.figshare.com/files/13663493)
- [Subject4.mat](https://ndownloader.figshare.com/files/13663496)
- [Subject5.mat](https://ndownloader.figshare.com/files/13663499)
- [ImageFeatures.h5](https://ndownloader.figshare.com/files/15015971)

## Script files

- **analysis_FeaturePrediction.py**: Run image feature prediction for each subject, ROI, and layer (feature).
- **analysis_FeaturePredictionMergeResults.py**: Merge outputs of analysis_FeaturePrediction.py and calculate feature prediction accuracy.
- **analysis_CategoryIdentification.py**: Run category identification.
- **createfigure.py**: Create result figures.
- **god_config.py**: Define analysis parameters. This file is called in analysis_* scripts.

## Analysis

### Quick guide

    $ python analysis_FeaturePrediction.py
    $ python analysis_FeaturePredictionMergeResults.py
    $ python analysis_CategoryIdentification.py
    $ python createfigure.py
