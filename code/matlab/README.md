# Generic Decoding Demo/Matlab

This is MATLAB code for Generic Decoding Demo.

## Requirements

- [BrainDecoderToolbox2](https://github.com/KamitaniLab/BrainDecoderToolbox2)

## Data organization

All data should be placed in `matlab/data`.
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
            +-- ImageFeatures.mat (image features extracted with Matconvnet)
            |
            +-- images/ (training and test images)
            |
            +-- cnn/ (Caffe CNN related data)

`images` and `cnn` are required if you calculate image feature by yourself.

## Analysis

Run the following script on MATLAB.

```
>> analysis_FeaturePrediction
>> analysis_FeaturePredictionAccuracy
>> analysis_CategoryIdentification
```

The all results will be saved in `results` directory.

To visualize the results, run the following script.

```
>> createfigure
```

`god_makefigure` will create two figures: one shows the results of image feature and category-averaged feature prediction, and the other displays the results of category identification. The figures will be saved in `results` directory in PDF format (`FeaturePredictionAccuracy.pdf` and `IdentificationAccuracy.pdf`).
