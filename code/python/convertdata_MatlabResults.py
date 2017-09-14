"""
Convert unit feature prediciton results from mat to pkl

This file is a part of GenericDecoding_demo.
"""


import sys
import os
import glob
import pickle
from itertools import product
from time import time

import numpy as np
import h5py


#------------------------------------------------------------------------------#
# Global settings                                                              #
#------------------------------------------------------------------------------#

analysis_name = __file__

subject_list = ('Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5')
roi_list = ('V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC')
nvox_list = (500, 500, 500, 500, 500, 500, 500, 1000, 1000, 1000)
nvox_dict = dict(zip(roi_list, nvox_list))

feature_file = 'ImageFeatures.mat'
feature_list = ('cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8',
                'hmax1', 'hmax2', 'hmax3', 'gist', 'sift')

## Directory to save results and temporally files
result_dir = './results_matlab'


#------------------------------------------------------------------------------#
# Main                                                                         #
#------------------------------------------------------------------------------#

if __name__ == "__main__":
    print "Running " + analysis_name

    for sbj, roi, feat in product(subject_list, roi_list, feature_list):
        print "Data %s-%s-%s" % (sbj, roi, feat)
        
        result_unit_file = os.path.join(result_dir, sbj, roi, feat + '.pkl')

        if os.path.isfile(result_unit_file):
            continue
        else:
            # Load unit result file
            print "Loading feature prediction results from mat files"

            flist = glob.glob(os.path.join(result_dir, sbj, roi, feat, "unit*.mat"))
            flist_unit = [f for f in flist if os.path.split(f)[1] != 'unit_all.mat']
            flist_unit.sort()

            results_unit = []
            for i, f in enumerate(flist_unit):
                print 'Loading %s' % f

                dat = h5py.File(f)
                results_unit.append({'subject' : sbj,
                                     'roi' : roi,
                                     'feature' : feat,
                                     'unit' : i + 1,
                                     'predict_percept' : np.array(dat['predict']['percept']).flatten(),
                                     'predict_imagery' : np.array(dat['predict']['imagery']).flatten(),
                                     'predict_percept_catave' : np.array(dat['predict']['perceptCatAve']).flatten(),
                                     'predict_imagery_catave' : np.array(dat['predict']['imageryCatAve']).flatten(),
                                     'category_test_percept' : np.array(dat['predict']['categoryImagery']).flatten(),
                                     'category_test_imagery' : np.array(dat['predict']['categoryPercept']).flatten()})

            # Save unit results
            print "Saving %s" % result_unit_file
            res_dir, res_file = os.path.split(result_unit_file)
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)
            with open(result_unit_file, 'wb') as f:
                pickle.dump(results_unit, f)

    print "Done"
