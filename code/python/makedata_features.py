'''
makedata_features

Extarce CNN features from images and save them in a BData file (hdf5).

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
import PIL.Image
from scipy.misc import imresize

import bdpy


# Main #######################################################################

def main():
    # Settings ---------------------------------------------------------------

    # Feture selection settings
    num_features = 1000

    # CNN model settings
    model_def = './data/cnn/bvlc_reference_caffenet/bvlc_reference_caffenet.prototxt'
    model_param = './data/cnn/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    cnn_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

    mean_image_file = './data/images/ilsvrc_2012_mean.npy' # ImageNet Large Scale Visual Recognition Challenge 2012

    # Results file
    data_dir = './data'
    featuredir = os.path.join(data_dir, 'ImageFeatures_caffe/')
    outputfile = os.path.join(data_dir, 'ImageFeatures_caffe.h5')

    # Misc settings
    rand_seed = 2501

    # Preparation ------------------------------------------------------------

    # Caffe GPU setting
    caffe.set_mode_gpu()

    # Result directory
    if not os.path.exists(featuredir):
        os.makedirs(featuredir)

    # Load CNN model ---------------------------------------------------------
    model = CnnModel(model_def, model_param, mean_image_file, batch_size=128, rand_seed=rand_seed)

    # Get image features for traning images ----------------------------------
    features_train = get_image_features(model, './data/images/image_training',
                                        layers=cnn_layers, n_features=num_features,
                                        save=True, savefile=os.path.join(featuredir, 'features_training.pkl'))

    # Get image features for test images -------------------------------------
    features_test = get_image_features(model, './data/images/image_test',
                                       layers=cnn_layers, n_features=num_features,
                                       save=True, savefile=os.path.join(featuredir, 'features_test.pkl'))

    # Category averaged features ---------------------------------------------
    features_cat_test = get_category_averaged_features(model, './data/images/category_test',
                                                       layers=cnn_layers, n_features=num_features,
                                                       save=True, savefile=os.path.join(featuredir, 'feature_category_ave_test.pkl'))

    # features_cat_cand = get_category_averaged_features(model, './data/images/category_candidate',
    #                                                    layers=cnn_layers, n_features=num_features,
    #                                                    save=True, savefile=os.path.join(featuredir, 'feature_category_ave_candidate.pkl'))

    # Create features data in BData ------------------------------------------
    features = bdpy.BData()

    #features_list = [features_train, features_test, features_cat_test, features_cat_cand]
    features_list = [features_train, features_test, features_cat_test]

    featuretype_arrays = []
    categoryid_arrays = []
    imageid_arrays = []
    features_dict = {lay : [] for lay in cnn_layers}

    for i, feat in enumerate(features_list):
        n_sample = len(feat.index)
        feature_type = i + 1

        image_id = feat.index

        for img in image_id:
            cat_id = get_category_id(img)
            if i < 2:
                img_id = get_image_id(img)
            else:
                img_id = np.nan

            featuretype_arrays.append(feature_type)
            categoryid_arrays.append(cat_id)
            imageid_arrays.append(img_id)

            for lay in cnn_layers:
                features_dict[lay].append(feat[lay][img])

    # Add data
    features.add(np.vstack(featuretype_arrays), 'FeatureType')
    features.add(np.vstack(categoryid_arrays), 'CatID')
    features.add(np.vstack(imageid_arrays), 'ImageID')
    for ft in features_dict:
        features.add(np.vstack(features_dict[ft]), ft)

    # Save merged features ---------------------------------------------------
    features.save(outputfile)


def get_category_id(image_filename):
    return int(image_filename.split('_')[0][1:])


def get_image_id(image_filename):
    cat_id = get_category_id(image_filename)
    img_id = int(image_filename.split('_')[1][:-5])
    return float('%d.%06d' % (cat_id, img_id))


# Class ######################################################################

class CnnModel(object):
    '''
    CNN model class
    '''

    def __init__(self, model_def, model_param, mean_image, batch_size=8, rand_seed=None):
        '''
        Load a pre-trained Caffe CNN model

        Original version was developed by Guohua Shen
        '''

        # Init random seed
        if not rand_seed is None:
            np.random.seed(rand_seed)

        # Enable GPU mode
        caffe.set_mode_gpu()

        # Prepare a mean image
        img_mean = np.load(mean_image)
        img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

        # Init the model
        channel_swap = (2, 1, 0)
        self.net = caffe.Classifier(model_def, model_param, mean=img_mean, channel_swap=channel_swap)

        h, w = self.net.blobs['data'].data.shape[-2:]
        self.image_size = (h, w)
        self.batch_size = batch_size

        self.net.blobs['data'].reshape(self.batch_size, 3, h, w)

        ## Init select feature index
        # FIXME
        layers = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8')
        self.feat_index = {}
        self.feat_index = {lay: np.random.permutation(self.net.blobs[lay].data[0].flatten().size)
                           for lay in layers}


    def get_feature(self, images, layers, feature_num=0):
        '''
        Returns CNN features

        Original version was developed by Guohua Shen (compute_cnn_feat_mL)
        '''

        # Convert 'images' to a list
        if not isinstance(images, list):
            images = [images]

        num_images = len(images)
        num_loop = int(np.ceil(num_images / float(self.batch_size)))

        image_index = [[ind for ind in xrange(i * self.batch_size, (i + 1) * self.batch_size) if ind < num_images]
                       for i in xrange(num_loop)]

        (h, w) = self.image_size
        mean_image = self.net.transformer.mean['data']

        feature_all = []

        for i, imgind in enumerate(image_index):
            for j, k in enumerate(imgind):
                img = imresize(PIL.Image.open(images[k]).convert('RGB'), (h, w), interp='bicubic')
                img = np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(mean_image, (3, 1, 1))
                self.net.blobs['data'].data[j] = img

            self.net.forward(end=layers[-1])

            for j, k in enumerate(imgind):
                if not feature_num == 0:
                    feat_list = [self.net.blobs[lay].data[j].flatten()[self.feat_index[lay][:feature_num]] for lay in layers]
                else:
                    # Returns all features
                    feat_list = [self.net.blobs[lay].data[j].flatten() for lay in layers]
                feature_all.append(feat_list)

        feature_all = np.array(feature_all)
        return feature_all


# Functions ##################################################################

def get_image_features(net, imagedir, layers=[], n_features=1000, save=False, savefile=None):
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
    features = net.get_feature(imagefiles, layers, feature_num=n_features)
    end_time = time()

    print 'Time: %.3f sec' % (end_time - start_time)

    # Convert to pandas dataframe
    feat = pd.DataFrame([[lay for lay in img] for img in features],
                        index=[os.path.split(f)[1] for f in imagefiles],
                        columns=layers)

    # Save data in a pickle file
    if save:
        with open(savefile, 'wb') as f:
            pickle.dump(feat, f)
        print 'Saved %s' % savefile

    return feat


def get_category_averaged_features(net, imagedir, layers=[], n_features=1000, save=False, savefile=None):
    '''Calculate category averaged features'''

    feat_dir, feat_file = os.path.split(savefile)

    catlist = os.listdir(imagedir)

    dat = pd.DataFrame(index=catlist, columns=layers)

    for cat in catlist:
        catdir = os.path.join(imagedir, cat)
        catfile = os.path.join(feat_dir, feat_file + '-' + cat + '.pkl')

        if not os.path.isdir(catdir):
            continue

        print 'Category: %s' % catdir

        feat = get_image_features(net, catdir,
                                  layers=layers, n_features=n_features,
                                  save=True, savefile=catfile)

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


# Entry point ################################################################

if __name__ == '__main__':
    main()
