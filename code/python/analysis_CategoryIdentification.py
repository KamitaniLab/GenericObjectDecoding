'''
Object category identification

This file is a part of GenericDecoding_demo.
'''


import sys
import os
import pickle
from itertools import product

import numpy

import bdpy
from bdpy.stats import corrmat
from bdpy.dataform import convert_dataframe

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
featpred_file = gd.results_featurepred
result_file = gd.results_categoryident


#----------------------------------------------------------------------#
# Functions                                                            #
#----------------------------------------------------------------------#

def category_identification(inputs):
    '''Runs category identification'''

    sbj, roi, feat = inputs

    res = results_featpred.query("subject == @sbj and roi == @roi and feature ==@feat")

    ## Preparing data
    feature = features.get('value', feat)
    feature_type = features.get('type')
    feature_label = features.get('image_label')
    feature_catlabel = features.get('category_label')

    ## Category identification -----------------------------------------

    # FIXME: is there better way to extract numpy array?
    pred_percept = res['predict_percept'].as_matrix()[0]
    pred_imagery = res['predict_imagery'].as_matrix()[0]

    cat_percept = res['category_test_percept'].as_matrix()[0]
    cat_imagery = res['category_test_percept'].as_matrix()[0]

    ## Get category features
    ind_cattest = feature_type == 3
    feat_cattest = feature[ind_cattest, :]
    featlb_cattest = feature_catlabel[ind_cattest]

    test_feat_cat_percept = bdpy.get_refdata(feat_cattest, featlb_cattest, cat_percept)
    test_feat_cat_imagery = bdpy.get_refdata(feat_cattest, featlb_cattest, cat_imagery)

    feat_cat_test = test_feat_cat_percept
    feat_cat_other = feature[feature_type == 4, :] # Unseen categories

    labels = range(feat_cat_test.shape[0])
    candidate = numpy.vstack([feat_cat_test, feat_cat_other])

    ## Seen categories
    simmat = corrmat(pred_percept, candidate)
    correct_rate_percept = get_pwident_correctrate(simmat, labels)

    ## Imagined categories
    simmat = corrmat(pred_imagery, candidate)
    correct_rate_imagery = get_pwident_correctrate(simmat, labels)

    ## Calculate average correct rate
    correct_rate_percept_ave = numpy.mean(correct_rate_percept)
    correct_rate_imagery_ave = numpy.mean(correct_rate_imagery)

    ## Print results
    res_str = "%s-%s-%s\n" % (sbj, roi, feat)
    res_str += "Correct rate (seen)\t: %.2f %%\n" % (correct_rate_percept_ave * 100)
    res_str += "Correct rate (imagined)\t: %.2f %%" % (correct_rate_imagery_ave * 100)
    print res_str

    return {'subject' : sbj,
            'roi' : roi,
            'feature' : feat,
            'correct_rate_percept' : correct_rate_percept,
            'correct_rate_imagery' : correct_rate_imagery,
            'correct_rate_percept_ave' : correct_rate_percept_ave,
            'correct_rate_imagery_ave' : correct_rate_imagery_ave}


def get_pwident_correctrate(simmat, labels):
    '''
    Returns correct rate in pairwise identification

    Parameters
    ----------
    simmat : numpy array [num_prediction * num_category]
        Similarity matrix
    labels : list or vector [num_prediction]
        List or vector of indexes of true labels

    Returns
    -------
    correct_rate : correct rate of pair-wise identification
    '''

    num_pred = simmat.shape[0]

    correct_rate = []
    for i in xrange(num_pred):
        pred_feat = simmat[i, :]
        correct_feat = pred_feat[labels[i]]
        pred_num = len(pred_feat) - 1
        correct_rate.append((pred_num - numpy.sum(pred_feat > correct_feat)) / float(pred_num))

    return correct_rate


#----------------------------------------------------------------------#
# Main                                                                 #
#----------------------------------------------------------------------#

if __name__ == "__main__":
    print "Running " + analysis_name

    ## Load data (image features)
    print "Loading image features"
    features = Features(os.path.join(data_dir, feature_file), feature_type)

    ## Load feature predition results
    print "Loading feature prediction results"
    res_file = os.path.join(result_dir, featpred_file)
    with open(res_file, 'rb') as f:
        results_featpred = pickle.load(f)

    ## Create result dir
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    ## Generate analysis list
    feature_list = features.layers
    input_list = [(s, r, f) for s, r, f in product(subject_list, roi_list, feature_list)]
    
    ## Category identification
    print "Running category identification"
    results = map(category_identification, input_list)

    ## Save results
    print "Saving results"
    df_results = convert_dataframe(results)

    with open(os.path.join(result_dir, result_file), 'wb') as f:
        pickle.dump(df_results, f)
