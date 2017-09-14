'''
Convolutional Neural Network class for Generic Decoding
'''


import caffe
import numpy as np
import PIL.Image
from scipy.misc import imresize


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
