'''Merge results of analysis_FeaturePrediction.py'''


from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd

import bdpy
from bdpy.stats import corrcoef

import god_config as config


# Main #################################################################

def main():
    results_dir = config.results_dir
    output_file = config.results_file

    # Load results -----------------------------------------------------
    result_list = []
    for rf in os.listdir(results_dir):
        rf_full = os.path.join(results_dir, rf)
        print('Loading %s' % rf_full)
        with open(rf_full, 'rb') as f:
            res = pickle.load(f)
        result_list.append(res)

    # Merge result dataframes ------------------------------------------
    results = pd.concat(result_list, ignore_index=True)

    # Drop unnecessary columns
    results.drop('predicted_feature', axis=1, inplace=True)
    results.drop('true_feature', axis=1, inplace=True)
    results.drop('test_label', axis=1, inplace=True)

    # Calculated feature prediction accuracy ---------------------------
    res_pt = results.query('test_type == "perception"')
    res_im = results.query('test_type == "imagery"')

    # Profile correlation (image)
    res_pt['profile_correlation_image'] = [corrcoef(t, p, var='col')
                                           for t, p in zip(res_pt['true_feature_averaged'],
                                                           res_pt['predicted_feature_averaged'])]
    res_pt['mean_profile_correlation_image'] = res_pt.loc[:, 'profile_correlation_image'].apply(np.nanmean)

    # Profile correlation (category, seen)
    res_pt['profile_correlation_cat_percept'] = [corrcoef(t, p, var='col')
                                                 for t, p in zip(res_pt['category_feature_averaged'],
                                                                 res_pt['predicted_feature_averaged'])]
    res_pt['mean_profile_correlation_cat_percept'] = res_pt.loc[:, 'profile_correlation_cat_percept'].apply(np.nanmean)

    # Profile correlation (category, imagined)
    res_im['profile_correlation_cat_imagery'] = [corrcoef(t, p, var='col')
                                                 for t, p in zip(res_im['category_feature_averaged'],
                                                                 res_im['predicted_feature_averaged'])]
    res_im['mean_profile_correlation_cat_imagery'] = res_im.loc[:, 'profile_correlation_cat_imagery'].apply(np.nanmean)

    # Merge results
    results_merged = pd.merge(res_pt, res_im, on=['subject', 'roi', 'feature'])

    # Rename columns
    results_merged = results_merged.rename(columns={'test_label_set_x' : 'test_label_set_percept',
                                                    'test_label_set_y' : 'test_label_set_imagery',
                                                    'true_feature_averaged_x' : 'true_feature_averaged_percept',
                                                    'true_feature_averaged_y' : 'true_feature_averaged_imagery',
                                                    'predicted_feature_averaged_x' : 'predicted_feature_averaged_percept',
                                                    'predicted_feature_averaged_y' : 'predicted_feature_averaged_imagery',
                                                    'category_label_set_x' : 'category_label_set_percept',
                                                    'category_label_set_y' : 'category_label_set_imagery',
                                                    'category_feature_averaged_x' : 'category_feature_averaged_percept',
                                                    'category_feature_averaged_y' : 'category_feature_averaged_imagery'})

    # Drop unnecessary columns
    results_merged.drop('test_type_x', axis=1, inplace=True)
    results_merged.drop('test_type_y', axis=1, inplace=True)

    # Save the merged dataframe ----------------------------------------    
    with open(output_file, 'wb') as f:
        pickle.dump(results_merged, f)
    print('Saved %s' % output_file)


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
