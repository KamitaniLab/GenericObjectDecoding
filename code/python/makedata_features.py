'''
gd_makedata_features

This file is a part of GenericDecoding_demo.
'''


import csv
import os
import pickle
import re
import imghdr
from time import time

import caffe
import numpy as np
import pandas as pd

from gd_cnn import CnnModel


## Global settings #############################################################

# Feture selection settings
num_features = 1000

# CNN model settings
model_def = './data/cnn/bvlc_reference_caffenet/bvlc_reference_caffenet.prototxt'
model_param = './data/cnn/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
cnn_layers = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8')

mean_image_file = './data/images/ilsvrc_2012_mean.npy' # ImageNet Large Scale Visual Recognition Challenge 2012

# Stimulus image settings
exp_stimuli_dir = ('./data/images/image_training', './data/images/image_test')
catave_image_dir = ('./data/images/category_test', './data/images/category_candidate')

# Results file
data_dir = './data'
featuredir = os.path.join(data_dir, 'ImageFeatures_caffe_test/')
outputfile = os.path.join(data_dir, 'ImageFeatures_caffe_test.pkl')

# Misc settings
rand_seed = 2501


## Functions ###########################################################

def get_image_features(net, imagedir, save=False, savefile=None):
    '''Calculate image features in `imagedir` and save them in `saveile`'''

    if os.path.exists(savefile):
        print '%s already exists. Loading the results.' % savefile
        with open(savefile, 'rb') as f:
            feat = pickle.load(f)
        return feat

    ## Get image files
    imagefiles = []
    for root, dirs, files in os.walk(imagedir):
        imagefiles = [os.path.join(root, f)
                      for f in files
                      if imghdr.what(os.path.join(root, f))]

    print 'Image num: %d' % len(imagefiles)

    ## Get image features
    print 'Getting features'
    start_time = time()
    features = net.get_feature(imagefiles, cnn_layers, feature_num=num_features)
    end_time = time()

    print 'Time: %.3f sec' % (end_time - start_time)

    # Convert to pandas dataframe
    feat = pd.DataFrame([[lay for lay in img] for img in features],
                        index=[os.path.split(f)[1] for f in imagefiles],
                        columns=cnn_layers)

    # Save data in a pickle file
    if save:
        with open(savefile, 'wb') as f:
            pickle.dump(feat, f)
        print 'Saved %s' % savefile
            
    return feat


def get_category_averaged_features(net, imagedir, save=False, savefile=None):
    '''Calculate category averaged features'''

    feat_dir, feat_file = os.path.split(savefile)
    
    catlist = os.listdir(imagedir)

    dat = pd.DataFrame(index=catlist, columns=cnn_layers)

    for cat in catlist:
        catdir = os.path.join(imagedir, cat)
        catfile = os.path.join(feat_dir, feat_file + '-' + cat + '.pkl')

        if not os.path.isdir(catdir):
            continue

        print 'Category: %s' % catdir
        
        feat = get_image_features(net, catdir, save=True, savefile=catfile)

        # Calc mean
        print 'Calculating averaged features'
        layers = feat.columns
        for lay in layers:
            feat_mean = np.mean(np.vstack([f for f in feat[lay]]), axis=0)
            dat[lay][cat] = feat_mean

    if save:
        with open(savefile, 'wb') as f:
            pickle.dump(dat, f)
        print 'Saved %s' % savefile

    return dat


def add_features_df(df, features, featuretype=0):
    '''Add features to a dataframe'''

    imageids = []
    catids = []
    for i in features.index:

        # Get image and category IDs
        if featuretype == 1 or featuretype == 2:
            (cat_str, img_str) = i.split('_')
            cat_id = np.int(cat_str[1:])
            img_num = int(img_str[:-5])
            img_id = float('%d.%06d' % (cat_id, img_num))
        else:
            cat_id = np.int(i)
            img_id = np.nan

        imageids.append(img_id)
        catids.append(cat_id)

    features['ImageID'] = imageids
    features['CatID'] = catids
    features['FeatureType'] = [np.int(featuretype) for _ in xrange(features.shape[0])]

    return df.append(features)

    
## Main ################################################################

if __name__ == '__main__':

    caffe.set_mode_gpu()

    if not os.path.exists(featuredir):
        os.makedirs(featuredir)
    
    ## Load CNN model --------------------------------------------------
    model = CnnModel(model_def, model_param, mean_image_file, batch_size=128, rand_seed=rand_seed)

    ## Init Pandas dataframe
    df = pd.DataFrame(columns=['ImageID', 'CatID', 'FeatureType', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'])
    
    ## Image features --------------------------------------------------
    features = get_image_features(model, './data/images/image_training', save=True, savefile=os.path.join(featuredir, 'feature_training.pkl'))
    df = add_features_df(df, features, featuretype=1)

    features = get_image_features(model, './data/images/image_test', save=True, savefile=os.path.join(featuredir, 'feature_test.pkl'))
    df = add_features_df(df, features, featuretype=2)
    
    ## Category averaged features --------------------------------------
    features = get_category_averaged_features(model, './data/images/category_test', save=True, savefile=os.path.join(featuredir, 'feature_category_ave_test.pkl'))
    df = add_features_df(df, features, featuretype=3)

    features = get_category_averaged_features(model, './data/images/category_candidate', save=True, savefile=os.path.join(featuredir, 'feature_category_ave_candidate.pkl'))
    df = add_features_df(df, features, featuretype=4)

    ## Save merged features --------------------------------------------
    df['FeatureType'] = df['FeatureType'].astype('int')
    df['CatID'] = df['CatID'].astype('int')
    with open(outputfile, 'wb') as f:
        pickle.dump(df, f)
