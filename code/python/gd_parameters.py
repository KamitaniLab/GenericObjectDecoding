'''
Common parameters for Generic Decoding
'''

data_dir = './data'

## Brain data ----------------------------------------------------------

# All data
subject_list = ('Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5')
roi_list = ('V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC')
nvox_list = (500, 500, 500, 500, 500, 500, 500, 1000, 1000, 1000)

nvox_dict = dict(zip(roi_list, nvox_list))

## Image/category features ---------------------------------------------
feature_file = 'ImageFeatures.h5'
feature_type = 'matconvnet'

## Results files
results_featurepred = 'FeaturePrediction.pkl'
results_categoryident = 'CategoryIdentification.pkl'

## Output directory ----------------------------------------------------
result_dir = './results'

## Model training parameters -------------------------------------------

# Sparse linear regression settings
num_itr = 200
