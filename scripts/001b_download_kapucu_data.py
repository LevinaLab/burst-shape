import os
import subprocess

import numpy as np

from src.folders import get_data_kapucu_folder

na = np.array

# % Load rats cultures
days = [7, 10, 14, 17, 21, 24, 28, 31, 35]

url = "https://gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset/raw/master/Data/Rat_MEA1/Rat_MEA1_spikes_noise_explogs/Rat_190617_MEA1_DIV%s_spikes.csv"  # %day

path = get_data_kapucu_folder()
os.makedirs(path, exist_ok=True)
load = True

if load:
    for day in days:
        bashCommand = "wget %s -P %s" % (url % day, path)
        subprocess.call(bashCommand, shell=True)  # stdout=subprocess.PIPE)

# % Load iPSC cultures


ipsc_days = np.arange(7, 66)  # [7,10,11,14,17,18,21,24,25,28]
url = "https://gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset/raw/master/Data/hPSC_MEA1/hPSC_MEA1_spikes_noise_explogs/hPSC_20517_MEA1_DIV%s_spikes.csv"

# path = '../../../data/iPSCcsCTX/'
load = True

if load:
    for day in ipsc_days:
        bashCommand = "wget %s -P %s" % (url % day, path)
        subprocess.call(bashCommand, shell=True)  # stdout=subprocess.PIPE)

# %
ipsc_days = np.arange(7, 66)  # [7,10,11,14,17,18,21,24,25,28]
url = "https://gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset/raw/master/Data/hPSC_MEA2/hPSC_MEA2_spikes_noise_explogs/hPSC_20517_MEA2_DIV%s_spikes.csv"

# path = '../../data/iPSCcsCTX/'
load = True

if load:
    for day in ipsc_days:
        bashCommand = "wget %s -P %s" % (url % day, path)
        subprocess.call(bashCommand, shell=True)  # stdout=subprocess.PIPE)
