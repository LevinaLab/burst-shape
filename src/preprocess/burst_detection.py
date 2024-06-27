# Updated population burst detection
import numpy as np
import scipy.stats
from numba import jit

na = np.array


def pop_burst_detection(
    st,
    gid,
    isi_thr,
    minBdur=50,  # 100,
    minIBI=800,  # 1000,
    minSburst=200,
    onset_threshold=3,
    peak_threshold=5,
):
    """Detect network burst based on the detection of bursts in
    each electrode and further thresholding

    Args:
        st (list): spike times (is ms )
        gid (list): neuron ids
    Returns:
        pop_bursts_final (list): list of tuples with the start and end of each burst
    """
    st = na(st)
    gid = na(gid)
    # gid_ids = np.unique(gid)
    # main vector to store the number of bursts per electrode in each time bin
    burst_vectors = np.ones(shape=(int(np.max(st)), 1))
    # add the population detection

    # isi_thr =20
    # isi_thr = 100#min(max(np.median(np.diff(st)),100),500)
    # min(max(np.median(np.diff(st)),1),100)
    bursts = MI_bursts(
        st,  # np.sort(st),
        maxISIstart=isi_thr,  # 100.0,
        maxISIb=isi_thr,  # isi_thr,
        minBdur=minBdur,  # 100,
        minIBI=minIBI,  # 1000,
        minSburst=minSburst,
    )

    # print(bursts)
    for burst in bursts:
        burst_vectors[int(burst[0]) : int(burst[1]), 0] += 10
        # plt.plot([int(burst[0])/1000,int(burst[1])/1000],[0,0],'-',color='g',alpha=1,linewidth=2)

    # onset_threshold = 3# numer of sim active electrodes to consider a burst
    onset_threshold = 1  # np.percentile(na(burst_vectors[:,0]),50)#max(np.percentile(na(burst_vectors[:,0]),50),3)#min(np.median(na(burst_vectors[:,0])),3)
    peak_threshold = 1  # max(np.percentile(na(burst_vectors[:,0]),50),3)#max(np.percentile(na(burst_vectors[:,0]),99),5)
    # plt.plot(np.linspace(int(np.min(st))/1000,int(np.max(st))/1000,len(burst_vectors)),burst_vectors)
    # print(onset_threshold,peak_threshold)
    pop_bursts_final = detect_onests(
        burst_vectors,
        na(burst_vectors[:, 0] > onset_threshold, dtype=int),  # >onset_threshold
        peak_threshold=peak_threshold,
    )
    return pop_bursts_final


@jit(nopython=True)
def detect_onests(burst_vectors, thr_burst_vector, peak_threshold=5):
    """Detect the start and end of the network burst"""
    # detect the start and end of the network burst
    # peak_threshold = 5 # number of max simult active electrodes to consider a peak
    pop_bursts = []
    df_vec = np.diff(thr_burst_vector)
    for i, k in enumerate(df_vec):
        if k > 0:
            start = i
        elif k < 0:
            end = i
            pop_bursts.append([start, end])
    pop_bursts_final = []
    for burst in pop_bursts:
        if np.max(burst_vectors[burst[0] : burst[1], 0]) >= peak_threshold:
            pop_bursts_final.append(burst)
    return pop_bursts_final


def _bimodality_index(x):
    """computes the sample bomodality index based on
    https://en.wikipedia.org/wiki/Multimodal_distribution"""
    gamma = scipy.stats.skew(x)
    kappa = scipy.stats.kurtosis(x)
    n = len(x)
    return ((gamma**2) + 1) / (kappa + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))


def fixed_bursted_detection(
    st,
    gid,
    sc,
    bi_thr=0.5,
    isi_thr=20,
    minBdur=50,  # 100,
    minIBI=800,  # 1000,
    minSburst=200,
):
    bi = _bimodality_index(sc)

    if bi > bi_thr:  # np.median(sc)<=10:# and
        # isi_thr =min(max(np.percentile(np.diff(np.sort(st)),75),20),200)
        # isi_thr = np.percentile(np.diff(np.sort(st)),95)
        # plt.figure()
        # plt.hist(np.diff(np.sort(st)),200)

        # print(isi_thr)
        # print(isi_thr)
        # isi_thr = 100#min(max(np.mean(np.diff(st)),1),500)
        # isi_thr = min(np.mean(np.diff(st)),100)
        # print(isi_thr)
        bursts = pop_burst_detection(
            na(st),
            na(gid),
            isi_thr,
            minBdur=minBdur,  # 100,
            minIBI=minIBI,  # 1000,
            minSburst=minSburst,
            onset_threshold=3,
            peak_threshold=10,
        )

        bursts = na(bursts) / 1000
        # print(bursts.shape)
        # print(np.vstack([np.diff(bursts)<50]))
        if len(bursts) > 1:
            max_length = 50
            bursts = bursts[np.vstack([np.diff(bursts) < max_length])[:, 0], :]

    else:
        bursts = []

    return na(bursts)


def MI_bursts(st, maxISIstart=4.5, maxISIb=4.5, minBdur=40, minIBI=40, minSburst=50):
    """Min Interval method [1,2] for burst detections
    Optimized version from 03.03.20
    OV

    Args:
            [1] spike times (list) (single neuron or population) (ms)
            [2] (float) max ISI at start of burst (ms)
            [3] (float) max ISI in burst (ms)
            [4] (float) min burst duration (ms)
            [5] (float) min inter-burst interval (ms)
            [6] (float) min number of spike in a burst
    Returns:
            burst (list of tuples): burst start, burst end
    [1] Nex Technologies.NeuroExplorer Manual.  Nex Technologies,2014
    [2] Cotterill, E., and Eglen, S.J. (2018). Burst detection methods. ArXiv:1802.01287.
    """
    spikes = np.sort(st)
    r_spikes = np.round(spikes, -1)
    isi_pop = np.diff(spikes)
    burst_ = find_burstlets(spikes, r_spikes, isi_pop, maxISIstart, maxISIb, minSburst)
    #     print(burst_)
    bursts = []
    if burst_:
        bursts.append(burst_[0])
        for i, b in enumerate(burst_[1:]):
            if b[1] - b[0] >= minBdur and b[0] - bursts[-1][1] >= minIBI:
                bursts.append(b)
            elif b[0] - bursts[-1][1] <= minIBI:
                bursts[-1] = (bursts[-1][0], b[1])

    return bursts


@jit(nopython=True)
def find_burstlets(
    spikes, r_spikes, isi_pop, maxISIstart=4.5, maxISIb=4.5, minSburst=100
):
    """
    Helper to find burstlets
    Args:
        spikes (arr): spike times
        r_spikes(arr): rounded spike times
        isi_pop(arr):isi
    Returns:
            burst_ (list of tuples): Burst start, burst end
    """
    b_spikes = None
    burst_ = []
    sync_b = False
    b_start = 0
    b_size = 0
    for i, s in enumerate(spikes[:-1]):
        if isi_pop[i] < maxISIstart and b_start == 0:
            b_size = 0
            b_start += 1
            b_spikes = s

        elif isi_pop[i] <= maxISIb and b_start > 0:  # start if two conseq init isi
            b_start += 1
            b_size += 1

        elif b_start >= minSburst:
            burst_.append((b_spikes, s))
            b_spikes = None
            b_size = 0
            sync_b = False
            b_start = 0

        else:
            b_spike = None
            b_size = 0
            sync_b = False
            b_start = 0
    return burst_
