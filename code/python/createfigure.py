'''Create figures for results of classification_pairwise'''


from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

import bdpy.fig as bfig

import god_config as config


# Main #################################################################

def main():
    results_file = config.results_file
    output_file_featpred = os.path.join('results', config.analysis_name + '_featureprediction.pdf')
    output_file_catident = os.path.join('results', config.analysis_name + '_categoryidentification.pdf')

    roi_label = config.roi_labels

    # Load results -----------------------------------------------------
    with open(results_file, 'rb') as f:
        print('Loading %s' % results_file)
        results = pickle.load(f)

    # Figure settings
    plt.rcParams['font.size'] = 8

    # Plot (feature prediction) ----------------------------------------
    fig = plt.figure(figsize=[2 * 11.69, 2 * 8.27], dpi=100)

    subplotpos_image = [43, 37, 31, 25, 19, 13, 7, 1, 44, 38, 32, 26, 20]
    subplotpos_catpt = [45, 39, 33, 27, 21, 15, 9, 3, 46, 40, 34, 28, 22]
    subplotpos_catim = [47, 41, 35, 29, 23, 17, 11, 5, 48, 42, 36, 30, 24]

    # Image
    plotresults(fig, results, value_key='mean_profile_correlation_image',
                roi_label=roi_label, subplot_index=subplotpos_image,
                caption='image_seen; ', ylabel='Corr. coef.', ylim=[-0.2, 0.6], ytick=[-0.2, 0, 0.2, 0.4])

    # Category, seen
    plotresults(fig, results, value_key='mean_profile_correlation_cat_percept',
                roi_label=roi_label, subplot_index=subplotpos_catpt,
                caption='category_seen; ', ylabel='Corr. coef.', ylim=[-0.2, 0.6], ytick=[-0.2, 0, 0.2, 0.4])

    # Category, imagined
    plotresults(fig, results, value_key='mean_profile_correlation_cat_imagery',
                roi_label=roi_label, subplot_index=subplotpos_catim,
                barcolor=[0.8, 0.8, 0.8],
                caption='category_imagined; ', ylabel='Corr. coef.', ylim=[-0.2, 0.6], ytick=[-0.2, 0, 0.2, 0.4])

    # Draw path to the script
    fpath = os.path.abspath(__file__)
    bfig.draw_footnote(fig, fpath)

    # Save the figure
    plt.savefig(output_file_featpred)
    print('Saved %s' % output_file_featpred)

    plt.show()

    # Plot (category identification) -----------------------------------
    fig = plt.figure(figsize=[2 * 11.69, 2 * 8.27], dpi=100)

    subplotpos_percept = [44, 38, 32, 26, 20, 14, 8, 2, 45, 39, 33, 27, 21]
    subplotpos_imagery = [46, 40, 34, 28, 22, 16, 10, 4, 47, 41, 35, 29, 23]

    # Image
    plotresults(fig, results, value_key='catident_correct_rate_percept',
                roi_label=roi_label, subplot_index=subplotpos_percept,
                caption='seen; ', ylabel='Accuracy', ylim=[0.4, 1.0], ytick=[0.4, 0.6, 0.8, 1.0], textpos=[0, 0.92])

    # Category, seen
    plotresults(fig, results, value_key='catident_correct_rate_imagery',
                roi_label=roi_label, subplot_index=subplotpos_imagery,
                barcolor=[0.8, 0.8, 0.8],
                caption='imagined; ', ylabel='Accuracy', ylim=[0.4, 1.0], ytick=[0.4, 0.6, 0.8, 1.0], textpos=[0, 0.92])

    # Draw path to the script
    fpath = os.path.abspath(__file__)
    bfig.draw_footnote(fig, fpath)

    # Save the figure
    plt.savefig(output_file_catident)
    print('Saved %s' % output_file_catident)

    plt.show()


# Functions ############################################################

def plotresults(fig, results, value_key='', roi_label=[], feature_label=[],
                subplot_index=[], caption='', barcolor=[0.4, 0.4, 0.4],
                ylabel='', ylim=[-1, 1], ytick=[], textpos=[0, -0.12]):
    '''Draw results of feature prediction'''

    # Get mean and confidence interval ---------------------------------
    tb_mean = pd.pivot_table(results, index=['roi'], columns=['feature'],
                             values=[value_key], aggfunc=np.mean)
    tb_sem = pd.pivot_table(results, index=['roi'], columns=['feature'],
                            values=[value_key], aggfunc=st.sem)
    tb_num = pd.pivot_table(results, index=['roi'], columns=['feature'],
                            values=[value_key], aggfunc=len)

    tb_mean = tb_mean.reindex(index=roi_label)
    tb_sem = tb_sem.reindex(index=roi_label)
    tb_num = tb_num.reindex(index=roi_label)

    ci = st.t.interval(1 - 0.05, tb_num, loc=tb_mean, scale=tb_sem)
    tb_yerr = (tb_mean - ci[0], ci[1] - tb_mean)

    # Plot -------------------------------------------------------------
    xpos = range(len(roi_label))

    for feat, si in zip(tb_mean, subplot_index):
        plt.subplot(8, 6, si)

        # Draw results
        y = tb_mean[feat]
        yerr = (tb_yerr[0][feat], tb_yerr[1][feat])
        plt.bar(xpos, y, align='center',
                color=barcolor, edgecolor=barcolor,
                linewidth=2,
                yerr=yerr, ecolor='k', capsize=0)

        # Draw 'feature' name
        feat_name = caption + feat[1]
        plt.text(textpos[0], textpos[1], feat_name, fontsize=12)

        # Modify x and y axes
        plt.xticks(xpos, roi_label)
        plt.xlim([-0.5, len(xpos) - 0.5])

        plt.ylabel(ylabel)
        plt.yticks(ytick)
        plt.ylim(ylim)

        bfig.box_off(plt.gca())

        # Remove ticks on x and y axes
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')

        # Horizontal grid lines
        plt.gca().yaxis.grid(True, linestyle='-', linewidth=1, color='gray')
        plt.gca().set_axisbelow(True)

    # Adjust subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.2)


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
