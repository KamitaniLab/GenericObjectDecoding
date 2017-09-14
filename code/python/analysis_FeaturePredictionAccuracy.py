'''
Feature prediction accuracy (image_seen, category_seen, category_imagined)

This file is a part of GenericDecoding_demo.
'''


import os
import pickle
import sys
from itertools import product
from time import time

import numpy as np
import pandas as pd

sys.path.append('lib/bdpy') # Path to bdpy
import bdpy
from bdpy.stats import corrcoef

import gd_parameters as gd
from gd_features import Features


#----------------------------------------------------------------------#
# Global settings                                                      #
#----------------------------------------------------------------------#

analysis_name = __file__

# Get global parametes from gd_parameters
data_dir = gd.data_dir
subject_list = gd.subject_list
roi_list = gd.roi_list
feature_file = gd.feature_file
feature_type = gd.feature_type
result_dir = gd.result_dir
feature_file = gd.feature_file
feature_type = gd.feature_type
result_dir = gd.result_dir

result_file = gd.results_featurepred


#----------------------------------------------------------------------#
# Main                                                                 #
#----------------------------------------------------------------------#

if __name__ == '__main__':
    print 'Running ' + analysis_name

    ## Load data (image features) --------------------------------------
    print 'Loading image features'
    features = Features(os.path.join(data_dir, feature_file), feature_type)

    ## Create result dir -----------------------------------------------
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    ## Run analysis for each subject, ROI, and feature -----------------
    results_columns = ['subject', 'roi', 'feature',
                       'category_test_percept', 'category_test_imagery',
                       'predict_percept', 'predict_imagery',
                       'predacc_image_percept',
                       'predacc_category_percept',
                       'predacc_category_imagery']

    results = pd.DataFrame([], columns=results_columns)

    for sbj, roi, feat in product(subject_list, roi_list, features.layers):
        start_time = time()

        analysis_id = '%s-%s-%s-%s' % (__file__, sbj, roi, feat)
        print 'Analysis %s' % analysis_id

        ## Load feature prediction results for each unit
        result_unit_file = os.path.join(result_dir, sbj, roi, feat + '.pkl')
        with open(result_unit_file, 'rb') as f:
            results_unit = pickle.load(f)

        ## Preparing data (image features)
        feature = features.get('value', feat)
        feature_type = features.get('type')
        feature_catlabel = features.get('category_label')

        ## Calculate image and category feature prediction accuracy

        ## Aggregate all units prediction (num_sample * num_unit)
        pred_percept = np.vstack(results_unit['predict_percept_catave']).T
        pred_imagery = np.vstack(results_unit['predict_imagery_catave']).T

        cat_percept = results_unit['category_test_percept'][0]
        cat_imagery = results_unit['category_test_imagery'][0]

        ## Get image features for image_seen, category_seen, and category_imagined
        ind_imgtest = feature_type == 2
        ind_cattest = feature_type == 3

        test_feat_img = bdpy.get_refdata(feature[ind_imgtest, :],
                                         feature_catlabel[ind_imgtest],
                                         cat_percept);
        test_feat_cat_percept = bdpy.get_refdata(feature[ind_cattest, :],
                                                 feature_catlabel[ind_cattest],
                                                 cat_percept)
        test_feat_cat_imagery = bdpy.get_refdata(feature[ind_cattest, :],
                                                 feature_catlabel[ind_cattest],
                                                 cat_imagery)

        ## Get image and category feature prediction accuracy
        predacc_image_percept = np.nanmean(corrcoef(pred_percept, test_feat_img, var='col'))
        predacc_category_percept = np.nanmean(corrcoef(pred_percept, test_feat_cat_percept, var='col'))
        predacc_category_imagery = np.nanmean(corrcoef(pred_imagery, test_feat_cat_imagery, var='col'))

        print 'Pred acc (image_percpet):    %f' % predacc_image_percept
        print 'Pred acc (category_percpet): %f' % predacc_category_percept
        print 'Pred acc (category_imagery): %f' % predacc_category_imagery
        
        df_tmp = pd.DataFrame({'subject' : [sbj],
                               'roi' : [roi],
                               'feature' : [feat],
                               'category_test_percept' : [cat_percept],
                               'category_test_imagery' : [cat_imagery],
                               'predict_percept' : [pred_percept],
                               'predict_imagery' : [pred_imagery],
                               'predacc_image_percept' : [predacc_image_percept],
                               'predacc_category_percept' : [predacc_category_percept],
                               'predacc_category_imagery' : [predacc_category_imagery]},
                              columns=results_columns)
        results = results.append(df_tmp, ignore_index=True)

        print 'Time: %.3f sec' % (time() - start_time)

    ## Save results ----------------------------------------------------
    print 'Saving %s' % os.path.join(result_dir, result_file)
    with open(os.path.join(result_dir, result_file), 'wb') as f:
        pickle.dump(results, f)
