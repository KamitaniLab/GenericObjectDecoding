'''Configureation of Generic Object Decoding'''


import os


analysis_name = 'GenericObjectDecoding'

# Data settings
subjects = {'Subject1' : ['data/Subject1.h5'],
            'Subject2' : ['data/Subject2.h5'],
            'Subject3' : ['data/Subject3.h5'],
            'Subject4' : ['data/Subject4.h5'],
            'Subject5' : ['data/Subject5.h5']}

rois = {'VC' : 'ROI_VC = 1',
        'LVC' : 'ROI_LVC = 1',
        'HVC' : 'ROI_HVC = 1',
        'V1' : 'ROI_V1 = 1',
        'V2' : 'ROI_V2 = 1',
        'V3' : 'ROI_V3 = 1',
        'V4' : 'ROI_V4 = 1',
        'LOC' : 'ROI_LOC = 1',
        'FFA' : 'ROI_FFA = 1',
        'PPA' : 'ROI_PPA = 1'}

num_voxel = {'VC' : 1000,
             'LVC' : 1000,
             'HVC' : 1000,
             'V1' : 500,
             'V2' : 500,
             'V3' : 500,
             'V4' : 500,
             'LOC' : 500,
             'FFA' : 500,
             'PPA' : 500}

image_feature_file = 'data/ImageFeatures.h5'
features = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']

# Results settings
results_dir = os.path.join('results', analysis_name)
results_file = os.path.join('results', analysis_name + '.pkl')

# Figure settings
roi_labels = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC']
