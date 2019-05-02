# Generic Decoding Demo/Matlab

This is MATLAB code for Generic Decoding Demo.

## Requirements

- [BrainDecoderToolbox2](https://github.com/KamitaniLab/BrainDecoderToolbox2)

## Data organization

All data should be placed in `matlab/data`.
Data can be obrained from [figshare](https://figshare.com/articles/Generic_Object_Decoding/7387130).
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

Download links:

- [Subject1.mat](https://ndownloader.figshare.com/files/13663487)
- [Subject2.mat](https://ndownloader.figshare.com/files/13663490)
- [Subject3.mat](https://ndownloader.figshare.com/files/13663493)
- [Subject4.mat](https://ndownloader.figshare.com/files/13663496)
- [Subject5.mat](https://ndownloader.figshare.com/files/13663499)
- [ImageFeatures.mat](https://ndownloader.figshare.com/files/15015977)

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

`createfigure.m` will create two figures: one shows the results of image feature and category-averaged feature prediction, and the other displays the results of category identification. The figures will be saved in `results` directory in PDF format (`FeaturePredictionAccuracy.pdf` and `IdentificationAccuracy.pdf`).
