'''
Feature prediction

This file is a part of GenericDecoding_demo.
'''


import os
import pickle
import sys
from itertools import product
from time import time

import numpy as np
from slir import SparseLinearRegression

import bdpy
from bdpy.preproc import select_top
from bdpy.ml import add_bias
from bdpy.stats import corrcoef
from bdpy.dataform import convert_dataframe

import gd_parameters as gd
from gd_features import Features


#----------------------------------------------------------------------#
# Global settings                                                      #
#----------------------------------------------------------------------#

analysis_name = __file__

data_dir = gd.data_dir
subject_list = gd.subject_list
roi_list = gd.roi_list
nvox_list = gd.nvox_list
feature_file = gd.feature_file
feature_type = gd.feature_type
result_dir = gd.result_dir
num_itr = gd.num_itr


#----------------------------------------------------------------------#
# Functions                                                            #
#----------------------------------------------------------------------#

def predict_feature_unit(analysis):
    '''Run feature prediction for each unit

    Parameters
    ----------
    analysis : dict
        Analysis parameters. This contains the following items:
            - subject
            - roi
            - num_voxel
            - feature
            - unit
            - num_test_category
            - ardreg_num_itr

    Returns
    -------
    dict
        Results dictionary. This contains the following items:
            - subject
            - roi
            - feature
            - unit
            - predict_percept
            - predict_imagery
            - predict_percept_ave
            - predict_imagery_ave
    '''

    ## Set analysis parameters -------------
    num_voxel = analysis['num_voxel']
    unit = analysis['unit']
    nitr = analysis['ardreg_num_itr']

    print 'Unit %d' % unit

    ## Get data ----------------------------

    ## Get brain data for training, test (perception), and test (imagery)
    ind_tr = data_type == 1
    ind_te_p = data_type == 2
    ind_te_i = data_type == 3

    data_train = data[ind_tr, :]
    data_test_percept = data[ind_te_p, :]
    data_test_imagery = data[ind_te_i, :]

    label_train = data_label[ind_tr]
    label_test_percept = data_label[ind_te_p]
    label_test_imagery = data_label[ind_te_i]

    ## Get image features for training
    ind_feat_tr = feature_type == 1
    feature_train = feature[ind_feat_tr, unit - 1]
    feature_label_train = feature_label[ind_feat_tr]

    ## Match training features to labels
    feature_train = bdpy.get_refdata(feature_train, feature_label_train, label_train)

    ## Preprocessing -----------------------

    ## Normalize data
    data_train_mean = np.mean(data_train, axis=0)
    data_train_std = np.std(data_train, axis=0, ddof=1)

    data_train_norm = (data_train - data_train_mean) / data_train_std
    data_test_percept_norm = (data_test_percept - data_train_mean) / data_train_std
    data_test_imagery_norm = (data_test_imagery - data_train_mean) / data_train_std

    feature_train_mean = np.mean(feature_train, axis=0)
    feature_train_std = np.std(feature_train, axis=0, ddof=1)

    if feature_train_std == 0:
        feature_train_norm = feature_train - feature_train_mean
    else:
        feature_train_norm = (feature_train - feature_train_mean) / feature_train_std

    ## Voxel selection based on correlation
    corr = corrcoef(feature_train_norm, data_train_norm, var='col')

    data_train_norm, select_ind = select_top(data_train_norm, np.abs(corr),
                                             num_voxel, axis=1, verbose=False)

    data_test_percept_norm = data_test_percept_norm[:, select_ind]
    data_test_imagery_norm = data_test_imagery_norm[:, select_ind]

    ## Add bias term
    data_train_norm = add_bias(data_train_norm, axis=1)
    data_test_percept_norm = add_bias(data_test_percept_norm, axis=1)
    data_test_imagery_norm = add_bias(data_test_imagery_norm, axis=1)

    ## Decoding ----------------------------

    if feature_train_std == 0:
        predict_percept_norm = np.zeros(data_test_percept.shape[0], feature_train_norm[1])
        predict_imagery_norm = np.zeros(data_test_imagery.shape[0], feature_train_norm[1])

    else:
        model = SparseLinearRegression(n_iter=nitr, prune_mode=1)

        ## Model training
        #import pdb; pdb.set_trace()
        model.fit(data_train_norm, feature_train_norm)

        ## Image feature preiction (percept & imagery)
        predict_percept_norm = model.predict(data_test_percept_norm)
        predict_imagery_norm = model.predict(data_test_imagery_norm)

    # De-normalize predicted features
    predict_percept = predict_percept_norm * feature_train_std + feature_train_mean
    predict_imagery = predict_imagery_norm * feature_train_std + feature_train_mean

    ## Average prediction results for each test category
    # [Note]
    # Image IDs (labels) is '<category_id>.<image_id>'
    # (e.g., '123456.7891011' is 'image 7891011 in category 123456').
    # Thus, the integer part of `label` represents the category ID.
    cat_te_p = np.floor(label_test_percept)
    cat_te_i = np.floor(label_test_imagery)

    category_test_percept = sorted(set(cat_te_p))
    category_test_imagery = sorted(set(cat_te_i))

    predict_percept_ave = [np.mean(predict_percept[cat_te_p == i]) for i in category_test_percept]
    predict_imagery_ave = [np.mean(predict_imagery[cat_te_i == i]) for i in category_test_imagery]

    #import pdb; pdb.set_trace()

    ## Return results
    return {'subject' : analysis['subject'],
            'roi' : analysis['roi'],
            'feature' : analysis['feature'],
            'unit' : unit,
            'predict_percept' : predict_percept,
            'predict_imagery' : predict_imagery,
            'predict_percept_catave' : predict_percept_ave,
            'predict_imagery_catave' : predict_imagery_ave,
            'category_test_percept' : category_test_percept,
            'category_test_imagery' : category_test_imagery}


#----------------------------------------------------------------------#
# Main                                                                 #
#----------------------------------------------------------------------#

if __name__ == '__main__':
    print 'Running ' + analysis_name

    ## Load data (brain data & image features) -------------------------
    print 'Loading brain data'
    brain_data = {sbj : bdpy.BData(os.path.join(data_dir, sbj + '.mat'))
                  for sbj in subject_list}

    print 'Loading image features'
    features = Features(os.path.join(data_dir, feature_file), feature_type)

    ## Setup directories -----------------------------------------------

    ## Result dir
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    ## Tmp dir
    if not os.path.isdir('./tmp'):
        os.makedirs('./tmp')

    ## Run analysis for each subject, ROI, and feature -----------------
    for sbj, roi, feat in product(subject_list, roi_list, features.layers):
        analysis_id = '%s-%s-%s-%s' % (__file__, sbj, roi, feat)
        result_unit_file = os.path.join(result_dir, sbj, roi, feat + '.pkl')
        lockfile = os.path.join('./tmp', analysis_id + '.lock')

        print 'Analysis %s-%s-%s' % (sbj, roi, feat)

        ## Check whether run the analysis or not
        if os.path.isfile(result_unit_file):
            print 'Already done. Skipped.'
            continue

        if os.path.isfile(lockfile):
            print 'Already running. Skipped.'
            continue

        with open(lockfile, 'w'):
            pass

        start_time = time()

        ## Generate analysis list
        nvox_dict = dict(zip(roi_list, nvox_list))
        analysis_list = [{'subject' : sbj,
                          'roi' : roi,
                          'num_voxel' : nvox_dict[roi],
                          'feature' : feat,
                          'unit' : i + 1,
                          'ardreg_num_itr' : num_itr}
                         for i in xrange(features.num_units[feat])]

        ## Preparing data ----------------------------------------------

        ## Brain data
        data = brain_data[sbj].get_dataset('ROI_' + roi)
        data_type = brain_data[sbj].get_dataset('DataType').flatten()
        data_label = brain_data[sbj].get_dataset('Label').flatten()

        ## Image features
        feature = features.get('value', feat)
        feature_type = features.get('type')
        feature_label = features.get('image_label')
        feature_catlabel = features.get('category_label')

        ## Feature prediction for each unit ----------------------------
        print 'Running feature prediction'

        ## Training models and get predictions
        results_unit = map(predict_feature_unit, analysis_list)

        ## Save unit results
        results_df = convert_dataframe(results_unit)
        
        res_dir, res_file = os.path.split(result_unit_file)
        if not os.path.isdir(res_dir):
            os.makedirs(res_dir)

        with open(result_unit_file, 'wb') as f:
            pickle.dump(results_df, f)

        print 'Time: %.3f sec' % (time() - start_time)

        os.remove(lockfile)
