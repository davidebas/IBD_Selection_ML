def import_lib():
    import time
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LogNorm
    from scipy.stats import gaussian_kde
    import sys
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    import argparse
    import configparser

    from sklearn.metrics import roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import auc

    import tensorflow as tf
    from tensorflow import keras
    from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
    from keras.callbacks import ReduceLROnPlateau

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import get_custom_objects
	
    plt.rcParams['figure.figsize'] = (10,6)
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    plt.rc('legend',fontsize=15)
    plt.rc('font',size=15)
    plt.rcParams["image.cmap"] = 'bwr_r'    # Altre impostazioni grafiche

    return time, np, os, plt, pd, LogNorm, gaussian_kde, sys, sns, ListedColormap, argparse, configparser, roc_curve, train_test_split, auc, tf, keras, SGD, Adam, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, ReduceLROnPlateau, Sequential, Dense, Activation, Dropout, K, get_custom_objects

def read_settings(file_path):
    import configparser
    config = configparser.ConfigParser()
    config.read(file_path)

    return dict(config.items(config.sections()[0]))

def create_folder_output(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)
