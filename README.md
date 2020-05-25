# Generic Object Decoding

This repository contains the data and demo codes for replicating results in our paper: [Horikawa and Kamitani (2017) Generic decoding of seen and imagined objects using hierarchical visual features. Nature Communications 8:15037](https://www.nature.com/articles/ncomms15037).
The generic object decoding approach enabled decoding of arbitrary object categories including those not used in model training.

## Data (fMRI data and visual features)

The preprocessed fMRI data for five subjects (training, test_perception, and test_imagery) and visual features (CNN1-8, HMAX1-3, GIST, and SIFT) are available at [figshare](https://figshare.com/articles/Generic_Object_Decoding/7387130).
The fMRI data were saved as the [BrainDecoderToolbox2](https://github.com/KamitaniLab/BrainDecoderToolbox2)/[bdpy](https://github.com/KamitaniLab/bdpy) format.

The unpreprocessed fMRI data is available at [OpenNeuro](https://openneuro.org/datasets/ds001246).

## Visual images

For copyright reasons, we do not make the visual images used in our experiments publicly available.
Please contact us via email (<brainliner-admin@atr.jp>) for stimulus image request.

## Demo program

Demo programs for Matlab and Python are available in [code/matlab](code/matlab/) and [code/python](code/python), respectively.
See README.md in each directory for the details.
