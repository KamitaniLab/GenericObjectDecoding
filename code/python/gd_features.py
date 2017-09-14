'''
Image feature class for Generic Decoding
'''

import sys
import pickle

import numpy as np

sys.path.append('lib/bdpy') # Path to bdpy
import bdpy


class Features(object):
    '''Image/category features class'''

    def __init__(self, datfile=None, dattype=None, feature_list=[]):
        if datfile == None:
            self.datatype = ''
            self.layers = []
            self.data_raw = None
            self.num_units = {}
        else:
            self.load_features(datfile, dattype, feature_list)

    def load_features(self, datfile, dattype=None, feature_list=[]):
        if dattype == None:
            # TODO: add data (file) type guessing
            pass

        if dattype == 'matconvnet':
            self.datatype = dattype
            self.layers = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']
            self.data_raw = bdpy.BData(datfile)
            self.num_units = {feat : self.data_raw.select_feature(feat + ' = 1').shape[1] for feat in self.layers}
        elif dattype == 'caffe':
            self.datatype = dattype
            self.layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
            with open(datfile, 'rb') as f:
                self.data_raw = pickle.load(f)
            self.num_units = {feat : self.data_raw[feat][0].shape[0]
                              for feat in self.layers}

    def get(self, key, layer=None):
        if self.datatype == 'matconvnet':
            if key == 'value':
                return self.data_raw.get_dataset(layer)
            elif key == 'type':
                return self.data_raw.get_dataset('FeatureType').flatten()
            elif key == 'image_label':
                return self.data_raw.get_dataset('ImageID').flatten()
            elif key == 'category_label':
                return self.data_raw.get_dataset('CatID').flatten()
        elif self.datatype == 'caffe':
            if key == 'value':
                return np.vstack(self.data_raw[layer])
            elif key == 'type':
                return np.array(self.data_raw['FeatureType'])
            elif key == 'image_label':
                return np.array(self.data_raw['ImageID'])
            elif key == 'category_label':
                return np.array(self.data_raw['CatID'])
