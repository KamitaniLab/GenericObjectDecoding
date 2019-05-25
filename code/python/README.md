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

- [Subject1.h5](https://ndownloader.figshare.com/files/15049646)
- [Subject2.h5](https://ndownloader.figshare.com/files/15049649)
- [Subject3.h5](https://ndownloader.figshare.com/files/15049652)
- [Subject4.h5](https://ndownloader.figshare.com/files/15049655)
- [Subject5.h5](https://ndownloader.figshare.com/files/15049658)
- [ImageFeatures.h5](https://ndownloader.figshare.com/files/15015971)

## Script files

- **analysis_FeaturePrediction.py**: Run image feature prediction for each subject, ROI, and layer (feature).
- **analysis_FeaturePredictionMergeResults.py**: Merge outputs of analysis_FeaturePrediction.py and calculate feature prediction accuracy.
- **analysis_CategoryIdentification.py**: Run category identification.
- **createfigure.py**: Create result figures.
- **god_config.py**: Define analysis parameters. This file is called in analysis_* scripts.

## Analysis

Run the following scripts.

    $ python analysis_FeaturePrediction.py
    $ python analysis_FeaturePredictionMergeResults.py
    $ python analysis_CategoryIdentification.py

The all results will be saved in `results` directory.

To visualize the results, run the following script.

    $ python createfigure.py

`createfigure.py` will create two figures: one shows the results of image feature and category-averaged feature prediction, and the other displays the results of category identification. The figures will be saved in `results` directory in PDF format (`createfigure_featureprediction.pdf` and `createfigure_categoryidentification.pdf`).
