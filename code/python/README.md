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
Data can be obrained from <http://brainliner.jp/data/brainliner/Generic_Object_Decoding>.
The data directory should have the following files:

    data/ --+-- Subject1.mat (fMRI data, subject 1)
            |
            +-- Subject2.mat (fMRI data, subject 2)
            |
            +-- Subject3.mat (fMRI data, subject 3)
            |
            +-- Subject4.mat (fMRI data, subject 4)
            |
            +-- Subject5.mat (fMRI data, subject 5)
            |
            +-- ImageFeatures.h5 (image features extracted with Matconvnet)
            |
            +-- ImageFeatures_caffe.pkl (image features extracted with Caffe)
            |
            +-- images/ (training and test images)
            |
            +-- cnn/ (Caffe CNN related data)

Either `ImageFeatures.h5` or `ImageFeatures_caffe.pkl` is necessary for the analysis.
`images` and `cnn` are required if you calculate image feature by yourself.

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
