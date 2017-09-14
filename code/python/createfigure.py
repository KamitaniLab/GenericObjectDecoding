'''
Create figures for Generic Decoding Demo

This file is a part of GenericDecoding_demo.
'''


import os
import pickle
from itertools import product

import numpy as np
import scipy.stats as scst
import matplotlib.pyplot as plt


## Data settings ###############################################################

result_dir = './results_matconvnet'
result_featpred = 'FeaturePrediction.pkl'
result_catident = 'CategoryIdentification.pkl'


## Figure settings #############################################################

fig_save_dir = './results_matconvnet'

figsize = (2 * 11.69, 2 * 8.27) # A3 landscape
figdpi = 100
fontsize = 9

barwidth = 0.8

# Matconvnet
subplot_pos_offset = {'cnn1' : 0, 'cnn2' : -6, 'cnn3' : -12, 'cnn4' : -18,
                      'cnn5' : -24, 'cnn6' : -30, 'cnn7' : -36, 'cnn8' : -42,
                      'hmax1' : 1, 'hmax2' : -5, 'hmax3' : -11, 'gist' : -17, 'sift' : -23}
# Caffe
# subplot_pos_offset = {'conv1' : 0, 'conv2' : -6, 'conv3' : -12, 'conv4' : -18,
#                       'conv5' : -24, 'fc6' : -30, 'fc7' : -36, 'fc8' : -42}

figure_list = ('featpred', 'catident')

figure_filename = {'featpred' : 'FeaturePredictionAccuracy.pdf',
                   'catident' : 'IdentificationAccuracy.pdf'}

plot_type_list = {'featpred' : ['image_seen', 'category_seen', 'category_imagined'],
                  'catident' : ['seen', 'imagined']}

plot_types = {'image_seen' : {'subplot_base_pos' : 43,
                              'subplot_pos_offset' : subplot_pos_offset,
                              'data' : 'predacc_image_percept',
                              'color' : '0.4',
                              'ylabel' : 'Corr. coeff.',
                              'ylim' : (-0.2, 0.6),
                              'yticks' : [-0.2, 0, 0.2, 0.4, 0.6],
                              'ylinecolor' : 'gray',
                              'baseline' : 'off',
                              'basevalue' : 0,
                              'basecolor' : 'black',
                              'caption_pos' : [0.5, -0.1]},
              'category_seen' : {'subplot_base_pos' : 45,
                                 'subplot_pos_offset' : subplot_pos_offset,
                                 'data' : 'predacc_category_percept',
                                 'color' : '0.4',
                                 'ylabel' : 'Corr. coeff.',
                                 'ylim' : (-0.2, 0.6),
                                 'yticks' : [-0.2, 0, 0.2, 0.4, 0.6],
                                 'ylinecolor' : 'gray',
                                 'baseline' : 'off',
                                 'basevalue' : 0,
                                 'basecolor' : 'black',
                                 'caption_pos' : [0.5, -0.1]},
              'category_imagined' : {'subplot_base_pos' : 47,
                                     'subplot_pos_offset' : subplot_pos_offset,
                                     'data' : 'predacc_category_imagery',
                                     'color' : '0.8',
                                     'ylabel' : 'Corr. coeff.',
                                     'ylim' : (-0.2, 0.4),
                                     'yticks' : [-0.2, 0, 0.2, 0.4],
                                     'ylinecolor' : 'gray',
                                     'baseline' : 'on',
                                     'basevalue' : 50,
                                     'basecolor' : 'black',
                                     'caption_pos' : [0.5, -0.1]},
              'seen' : {'subplot_base_pos' : 44,
                        'subplot_pos_offset' : subplot_pos_offset,
                        'data' : 'correct_rate_percept_ave',
                        'color' : '0.4',
                        'ylabel' : 'Accuracy (%)',
                        'ylim' : (40, 100),
                        'yticks' : [40, 60, 80, 100],
                        'ylinecolor' : 'gray',
                        'baseline' : 'on',
                        'basevalue' : 50,
                        'basecolor' : 'black',
                        'caption_pos' : [0.5, 95]},
              'imagined' : {'subplot_base_pos' : 46,
                            'subplot_pos_offset' : subplot_pos_offset,
                            'data' : 'correct_rate_imagery_ave',
                            'color' : '0.8',
                            'ylabel' : 'Accuracy (%)',
                            'ylim' : (40, 100),
                            'yticks' : [40, 60, 80, 100],
                            'ylinecolor' : 'gray',
                            'baseline' : 'on',
                            'basevalue' : 50,
                            'basecolor' : 'black',
                            'caption_pos' : [0.5, 95]}}


## Functions ###################################################################

def draw_plots(fig, plots, nrow=8, ncol=6):
    '''
    Draw plots on a figure
    '''

    for plot in plots:
        ax = fig.add_subplot(nrow, ncol, plot['subplot_position'])

        if plot['plot_type'] == 'bar':
            ax.bar(plot['xdata'], plot['ydata'], yerr=plot['yerr'],
                   width=plot['barwidth'], color=plot['color'], ecolor='black')
        else:
            raise ValueError('Unknown plot type')

        if 'ylinecolor' in plot:
            xlim = ax.get_xlim()
            [ax.plot(xlim, (y, y), '-', color=plot['ylinecolor']) for y in plot['yticks']]

        if plot['baseline'] == 'on':
            xlim = ax.get_xlim()
            baseval = plot['basevalue']
            ax.plot(xlim, (baseval, baseval), '-', color=plot['basecolor'])

        if 'caption' in plot:
            ax.text(plot['caption_position'][0], plot['caption_position'][1],
                    plot['caption'], va='top', ha='left', size=fontsize)

        box_off(ax)
        if 'xlabel' in plot:      ax.set_xlabel(plot['xlabel'], size=fontsize)
        if 'xlim' in plot:        ax.set_xlim(plot['xlim'])
        if 'xticks' in plot:      ax.set_xticks(plot['xticks'])
        if 'xticklabels' in plot: ax.set_xticklabels(plot['xticklabels'], size=fontsize)
        if 'ylabel' in plot:      ax.set_ylabel(plot['ylabel'], size=fontsize)
        if 'ylim' in plot:        ax.set_ylim(plot['ylim'])
        if 'yticks' in plot:      ax.set_yticks(plot['yticks'])
        if 'yticklabels' in plot: ax.set_yticklabels(plot['yticklabels'], size=fontsize)

    fig.tight_layout()
    fig.show()
    return fig


def box_off(ax):
    '''
    Remove top and right axes (`box off` in Matlab)
    '''

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


## Main ########################################################################

if __name__ == '__main__':

    ## Load results
    with open(os.path.join(result_dir, result_featpred), 'rb') as f:
        res_featpred = pickle.load(f)

    with open(os.path.join(result_dir, result_catident), 'rb') as f:
        res_catident = pickle.load(f)

    ## Get subjects, ROIs and features lists
    subject_list = res_featpred.subject.unique()
    roi_list = res_featpred.roi.unique()
    feature_list = res_featpred.feature.unique()

    num_sbj = len(subject_list)

    ## Figure loop
    for figtype in figure_list:
        pt_list = plot_type_list[figtype]

        ## Create plot information
        plots = []
        for pt_key, feat in product(pt_list, feature_list):
            pt = plot_types[pt_key]

            if figtype == 'featpred':
                ydata = [res_featpred.query('feature == @feat and roi == @roi')[pt['data']].mean() for roi in roi_list]
                ystd = [res_featpred.query('feature == @feat and roi == @roi')[pt['data']].std() for roi in roi_list]
            elif figtype == 'catident':
                ydata = [100 * res_catident.query('feature == @feat and roi == @roi')[pt['data']].mean() for roi in roi_list]
                ystd = [100 * res_catident.query('feature == @feat and roi == @roi')[pt['data']].std() for roi in roi_list]
            else:
                raise ValueError('Unknown plot type')

            yerr = [scst.t.ppf(1 - 0.05, num_sbj - 1) * i for i in ystd / np.sqrt(num_sbj)]

            # Draw x tick labels (ROI) only at bottom plots
            xticklabels = roi_list if feat == 'cnn1' or feat == 'hmax1' else ''

              # Make plot information
            plots.append({'plot_type' : 'bar',
                          'xdata' : np.arange(len(ydata)) + 0.5 - barwidth / 2,
                          'xticks' : np.arange(len(ydata)) + 0.5,
                          'xticklabels' : xticklabels,
                          'ydata' : ydata,
                          'yerr' : yerr,
                          'ylabel' : pt['ylabel'],
                          'ylim' : pt['ylim'],
                          'yticks' : pt['yticks'],
                          'yticklabels' : [str(i) for i in pt['yticks']],
                          'ylinecolor' : pt['ylinecolor'],
                          'baseline' : pt['baseline'],
                          'basevalue' : pt['basevalue'],
                          'basecolor' : pt['basecolor'],
                          'caption' : '%s; %s' % (pt_key, feat),
                          'caption_position' : pt['caption_pos'],
                          'subplot_position' : pt['subplot_base_pos'] + pt['subplot_pos_offset'][feat],
                          'barwidth' : barwidth,
                          'color' : pt['color']})

        ## Make a figure
        fig = plt.figure(figsize=figsize, dpi=figdpi)

        ## Draw plots
        draw_plots(fig, plots)

        ## Save a figure
        save_file = os.path.join(fig_save_dir, figure_filename[figtype])
        fig.savefig(save_file, format='pdf')
