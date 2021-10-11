#---------------------------------------------------------------
#Import libraries
import logging

import pandas as pd
import os
import datetime
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.optimize import curve_fit
from lmoments3 import distr

#---------------------------------------------------------------

#---------------------------------------------------------------
# Function that allow to visualize the output of dropdowns menus
def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            print("Selected value: " + change['new'])
#---------------------------------------------------------------

#---------------------------------------------------------------
# Function for read Continuum section files
def read_section(section_path = None, column_labels = None, sep="\s+", column_names=["r_HMC","c_HMC","basin","section"], format='tabular'):
    if format == 'tabular':
        section_df = pd.read_csv(section_path, sep=sep, header=None)

        if len(section_df.columns) > len(column_names):
            print(' ---> WARNING! Section files has ' + str(section_df.columns.__len__) + ' columns!')
            print(' ---> First  ' + str(len(column_names)) + ' columns are interpreted as ' + ','.join(column_names) + ', others are ignored!')
            column_names.extend(['']*(len(section_df.columns)-len(column_names)))
        if len(section_df.columns) < len(column_names):
            print(' ---> ERROR! Section files has ' + str(section_df.columns.__len__) + ' columns, cannot interpret as standard HMC section file!')
            raise IOError("Verify your section file or provide a personal column setup!")

        section_df.columns=column_names

    return section_df
#---------------------------------------------------------------

#---------------------------------------------------------------
# Function for read Continuum outputs
def read_discharge_hmc(output_path='', output_name="hmc.hydrograph.txt", file_name= None, format='txt', start_time=None, end_time=None, col_names=None):
    if format=='txt':
        custom_date_parser = lambda x: datetime.datetime.strptime(x, "%Y%d%m%H%M%S")
        hmc_discharge_df = pd.read_csv(os.path.join(output_path,output_name), header=None, delimiter=r"\s+", parse_dates=[0], index_col=[0], date_parser=custom_date_parser)
        if len(col_names)==len(hmc_discharge_df.columns):
            hmc_discharge_df.columns=col_names
        else:
            print(' ---> ERROR! Number of hmc output columns is not consistent with the number of stations!')
            raise IOError("Verify your section file, your run setup or provide a personal column setup!")
        if not start_time is None:
            raise NotImplementedError("Time slice not implemented yet")

    return hmc_discharge_df
#---------------------------------------------------------------

#---------------------------------------------------------------
# Function for initialize the script
def read_data():
    section = read_section(section_path = "~/data/buzi/buzi.info_section.txt", column_names=["r_HMC","c_HMC","basin","section","station","area","th1","th2"], format='tabular')
    continuum_series = read_discharge_hmc(output_path='~/data/buzi/', output_name="hmc.hydrograph.txt", file_name= None, format='txt', col_names=section["section"].values)
    
    section_list = [i for i in section["section"].values]
    section_chooser = widgets.Dropdown(
    options=['Choose a section...'] + section_list,
    value='Choose a section...',
    description='Section:',
    disabled=False,
    )
    section_chooser.observe(on_change)
    display(section_chooser)
    
    return continuum_series,section_chooser
#---------------------------------------------------------------------
# Function to plot time series
def plot_series(continuum_series,series):
    plt.figure(figsize=[15,7])
    continuum_series[series].plot()
    plt.xlabel('time')
    plt.ylabel('discharge (m3/s)')
    plt.title('Series:' + series)

    plt.figure(figsize=[15,7])
    continuum_series[series].resample('Y').max().plot(marker='o', linewidth=0)
    plt.xlabel('time')
    plt.ylabel('discharge (m3/s)')
    plt.title('Annual maxima:' + series)

def calculate_empirical_cdf(continuum_series,series):
    # define the true objective function
    def objective(x, a, b):
        return a * x + b

    series_sort = np.sort(continuum_series[series].resample('Y').max())
    index = np.arange(1,len(series_sort)+1,1)
    ECDF = index/((max(index))+1)
    T_emp = 1/(1-ECDF)

    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale('log')
    ax.plot(T_emp,series_sort,'or')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.set_xlabel("Return period (Y)")
    ax.set_ylabel("Q (m3/s)")

    # Linear fit to the ECDF
    popt, _ = curve_fit(objective, series_sort, np.log(T_emp))
    a, b = popt
    x_line = np.arange(min(series_sort),max(series_sort),1)
    y_line = objective(x_line, a, b)
    ax.plot(np.exp(y_line),x_line,'-c', linewidth=2)
    ax.set_title("Empirical Return Period: " + series)
    ax.grid(which='minor', axis='both')
    
def extrapolate_cdf(continuum_series,series,Tmax=1000):
    # GEV L-moments fit
    paras = distr.gev.lmom_fit(continuum_series[series].resample('Y').max())
    fitted_gev = distr.gev(**paras)
    T_gev = np.arange(1,Tmax,1)
    P_gev = (T_gev-1)/T_gev
    x_gev = fitted_gev.ppf(P_gev)
    
    calculate_empirical_cdf(continuum_series,series)
    ax=plt.gca()
    ax.plot(T_gev,x_gev,'-g', linewidth=2)
