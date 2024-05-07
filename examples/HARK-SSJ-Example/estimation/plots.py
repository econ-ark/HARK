"""Simple plotting functions for time series"""
import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(data_dict, dims, xlabel='Quarters', **kwargs):
    plt.figure(**kwargs)
    for i, (name, data) in enumerate(data_dict.items()):
        plt.subplot(*dims, i+1)
        plt.plot(data)
        plt.title(name)
        plt.xlabel(xlabel)
        plt.axhline(y=0, color='#808080', linestyle=':')
    plt.tight_layout()


def plot_impulses(dict_of_impulsedicts, labels, series, dims, xlabel='Quarters', T=None, **kwargs):
    plt.figure(**kwargs)
    for i, name in enumerate(series):
        plt.subplot(*dims, i+1)
        for k, impulse_dict in dict_of_impulsedicts.items():
            plt.plot(impulse_dict.get(name)[:T], label=labels[k])
        plt.title(name)
        plt.xlabel(xlabel)
        plt.axhline(y=0, color='#808080', linestyle=':')
        if i == 0:
            plt.legend()
    plt.tight_layout()


def plot_decomp(Ds, data, shocks, series, xaxis, **kwargs):
    Tshow, len_se, len_sh = Ds.shape
    plt.figure(**kwargs)

    for io, o in enumerate(series):
        plt.subplot(1,3,1+io)
        y_offset_pos, y_offset_neg = 0, 0

        for ii, i in enumerate(shocks):
            D = Ds[:, io, ii] # current contribution
            y_offset = (D > 0) * y_offset_pos + (D < 0) * y_offset_neg
            y_offset_pos_ = y_offset_pos + np.maximum(D,0)
            y_offset_neg_ = y_offset_neg - np.maximum(-D,0)
            plt.fill_between(xaxis, y_offset_pos, y_offset_pos_, color=f'C{ii}', label=i)
            plt.fill_between(xaxis, y_offset_neg, y_offset_neg_, color=f'C{ii}')
            y_offset_pos = y_offset_pos_
            y_offset_neg = y_offset_neg_
        
        if data is not None:
            plt.plot(xaxis, data[:, io], color='black')
        if io == 0:
            plt.legend(framealpha=1)
        plt.title(o)

    plt.tight_layout()
    plt.show()
