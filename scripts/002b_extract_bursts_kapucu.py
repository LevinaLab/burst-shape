import os
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.folders import get_data_kapucu_folder

na = np.array
path = get_data_kapucu_folder()

res = list(os.walk(path, topdown=True))
files = res[0][2]  # all file names
div_days = [f.split('_')[3] for f in files if 'DIV' in f]
types = [f.split('_')[0] for f in files if 'DIV' in f]
mea_n = [f.split('_')[2] for f in files if 'DIV' in f]
import re

div_days = [re.findall(r'\d+', div) for div in div_days]
div_days = na(div_days, dtype=int).flatten()
indis = np.argsort(div_days)
div_days = div_days[indis]
types = na(types)[indis]
mea_n = na(mea_n)[indis]
files = na(files)[indis]


# %%
def gid_to_numbers(gid):
    for i, u_id in enumerate(np.unique(gid)):
        gid[gid == u_id] = i
    return gid


divs = []
# summaries = []
well_id = []
culture_type = []
mea_number = []
spks = []
for i, file_ in tqdm(enumerate(files), desc='Loading files'):
    div = div_days[i]
    type_ = types[i]
    mea_ = mea_n[i]
    spikes = pd.read_csv(os.path.join(path, file_))
    channels = spikes['Channel']
    wells = [ch.split('_')[0] for ch in channels]
    ch_n = [ch.split('_')[1] for ch in channels]
    spikes['well'] = wells
    spikes['ch_n'] = ch_n
    # Extract spikes for different wells
    # well_spikes= []
    for well in np.unique(wells):
        st = na(spikes['Time'][spikes['well'] == well])
        gid = na(spikes['ch_n'][spikes['well'] == well])
        spks.append([st, gid])
        # summaries.append(get_summary([st,gid],type_))
        divs.append(div)
        well_id.append(well)
        culture_type.append(type_)
        mea_number.append(mea_)

# Cut the noise at the beginning of a recording
mask = spks[247][0] > 125
spks[247][0] = spks[247][0][mask]
spks[247][1] = spks[247][1][mask]
# %%
data_stacked = pd.DataFrame({
    'spikes':spks,'DIV':divs,'well_id':well_id, 'culture_type':culture_type,'mea_number':mea_number,
})
data_stacked["times"] = data_stacked["spikes"].apply(lambda x: x[0])
# %%
plt.plot(spks[30][0],spks[30][1],'|')
plt.show()
# %%
plt.figure()

sc,bin_= np.histogram(spks[30][0],np.arange(0,600,0.01))
plt.plot(bin_[1:],sc,'-')
plt.show()