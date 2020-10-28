#!/usr/bin/env python

# RFI Hunter
# This program cleans RFI affected frequency channels
# *.py PSRCHIVE*.ar gatefile
# gatefile has comma separated start and end bins of the main pulse
# made by Mohsen Shamohammadi msh.ph.ir@gmail.com
from __future__ import print_function
import numpy as np
import psrchive as psr
import scipy.stats
import sys
from timeit import default_timer as timer
from matplotlib import pyplot as plt
def load_data(fn, gate_file):
    """ Load data from a .ar file 
    Open file, remove baseline, dedisperse, pscrunch, 
    remove the main pulse, apply gates.
    
    Args:
        fn (str): Filename to open
        gate_file (file): phase bin intervals to be removed (look at gate_reader function)
        0,15,500,512 means phase bins in the ranges 
        from 0 to 15 and from 500 to 512 will be removed
    
    Returns the archive and the amplitudes in the form of a numpy array 
    with a shape (time=1, pol=1, freq, phase_bin)
    """
    ar = psr.Archive_load(fn)
    print("Cleaning is started for {}".format(ar.get_filename()))
    patient = ar.clone()
    patient.remove_baseline()
    patient.pscrunch()
#    patient.tscrunch()
#    patient.fscrunch_to_nchan(patient.get_nchan()/scr_factor)
#    patient.dedisperse()
    data = patient.get_data()
    data_offpulse = main_pulse_wash(data, gate_file)
    return ar, data_offpulse


def apply_weights(data, weights):
    """ Apply weights from a .ar file """
    nsubs, npol, nchans, nbins = data.shape
    for isub in range(nsubs):
        data[isub] = data[isub]*weights[isub,...,np.newaxis]
    return data


def gate_reader(gate_list):
    """ Load a gate file
    Open a gate file -- it is a text file which contains 
    the phase bin intervals to be removed (gates).
    For instance: 0,15 means the gates are located in phase bins 0 and 15 
    the main pulse are located between the gates and will be removed.

    In the case of having multiple phase bin intervals to be removed, the gates should be placed
    next to each other and "," separated. 
    For instance: 0,15,500,512 means the phase bins from 0 to 15 and also from 500 to 512 
    are gated and will be removed.

    Return:
    s is the start bins to be gated
    e is the end bins to be gated
    """
    g = np.genfromtxt(gate_list, delimiter=',')

    # create two lists fro start and end bins
    g = g.reshape(len(g)/2, 2)
    g =np.flip(g,0)
    s = g[:, 0]
    e = g[:, 1]
    return s, e


def main_pulse_wash(data, bin_gate):
    """ Load amplitudes of the archive and gate file
    Wash the main pulse from the data
    
    Return:
    the off-pulse region of data (ready to be searched for RFI)
    """
    df = data.copy()
    if bin_gate == None:
        print("No gate file has been given")
    gate_start, gate_end = gate_reader(bin_gate)
    for i,_ in enumerate(gate_start):
        df = np.concatenate((df[:, :, :, :int(gate_start[i])], df[:, :, :, int(gate_end[i]):]), 3)
    return df



def fft_peak(data):
    """Load amplitufes of a single frequency channel
    This function was inspired from CoastGuard:
    https://github.com/plazar/coast_guard/blob/master/clean_utils.py

    Return:
    maximum of the absolute value of the FFT of the difference between the amplitudes and the mean value
    """
    data = np.asarray(data)
    m = np.mean(data)
    d = data - m
    fft = np.fft.rfft(d)
    ab = np.abs(fft)
    mx = np.max(ab)
    return mx



def get_std(l, xr1, xr2):
    """find a standard deviation coefficient for the simulation in max_simulate_find"""
    np.random.seed(0)
    o = np.random.normal(0,  1 , l)
    o_sort = np.sort(o)
#    xr1, xr2 = np.int(16*l/100) , np.int(50*l/100)
    part_std = np.std(o_sort[xr1 : xr2])
    std_coef = 1.0/part_std
#    print("std coeficient: {}".format(std_coef))
    return std_coef



def max_simulate_find(data):
    """
    Input:
    - a list of file containing the standard deviation or any statistics of the frequency channels

    Operations:
    - calculate the length of the data (number of channels)
    - sort the data
    - mask the zero values in the sorted data
    - find the 34% of the lower part of the data from the median, it is equal to the 1st sigma deviation from the median
    - based on simulation, std of good data should be 3.6938 times the std of the data in the range mentioned above (random generator starts from 0, i.e. np.random.seed(0))
    - take the difference between median of the data and median of simulated data
    - add the difference to the simulated data

    Return:
    - find the maximum value of the simulated data as the best value for the cut
    """
    l = len(data)
    data_sort = np.sort(data)
    xr1, xr2 = np.int(45*l/100) , np.int(65*l/100)
    np.random.seed(0)
    sim = np.random.normal(0, get_std(l, xr1, xr2) * np.std(data_sort[xr1 : xr2]) , l)
    shift_ind = np.int((xr1 + xr2)/2.0)
    sim_sort = np.sort(sim)
    diff = data_sort[shift_ind] - sim_sort[shift_ind]
    sim = sim + diff
    max_sim = np.max(sim)
    min_sim = np.min(sim)
    return min_sim, max_sim



def max_simulate_find_sigma(data, sigma):
    l = len(data)
    data_sort = np.sort(data)
    xr1, xr2 = np.int(50*l/100) , np.int(75*l/100)
    np.random.seed(0)
    sim = np.random.normal(0, get_std(l, xr1, xr2) * sigma * np.std(data_sort[xr1 : xr2]) , l)
    shift_ind = np.int((xr1 + xr2)/2.0)
    sim_sort = np.sort(sim)
    diff = data_sort[shift_ind] - sim_sort[shift_ind]
    sim = sim + diff
    max_sim = np.max(sim)
    min_sim = np.min(sim)
    return min_sim, max_sim

def simulate_baseline_tester(data):
    l = len(data)
    x = xrange(l)
    data_sort = np.sort(data)
#    data_sort = np.ma.masked_equal(data_sort, 0.)
#    b = peakutils.baseline(data_sort, deg=1)
    xr1, xr2 = np.int(50*l/100) , np.int(75*l/100)
#    data_norm, min_data_norm, max_data_norm = find_sigma_range(data_sort, 1)
#    xr1, xr2 = np.where(data_sort == min_data_norm)[0][0], np.where(data_sort == max_data_norm)[0][0]
#    print("normal indecies from {} to {}\nnormal range from {} to {}".format(xr1, xr2, min_data_norm, max_data_norm))
    np.random.seed(0)
#    sim = np.random.normal(0, get_std(l) * np.std(data_sort[xr1 : xr2]) , l)
    sim = np.random.normal(0, get_std(l, xr1, xr2) * np.std(data_sort[xr1 : xr2]) , l)
    sim_sort = np.sort(sim)
#    diff = np.median(data_sort) - np.median(sim)
    shift_ind = np.int((xr1 + xr2)/2)
    diff = data_sort[shift_ind] - sim_sort[shift_ind]
    sim = sim + diff
#    mu, std = scipy.stats.norm.fit(data_sort)
#    xmin, xmax = 0, l-1
#    xf = np.linspace(xmin, xmax, l)
#    p = scipy.stats.norm.pdf(x, mu, std)
    max_sim = np.max(sim)
    sim_sort = np.sort(sim)
#    res = data_sort - b
#    _, _, sigma_res = find_sigma_range(res, 4)
####    hline = sigma_cut_residual(data, 5)
 #    for i, (item1, item2) in enumerate(zip(data_sort, b)):
#        if (item1 - item2) <= sigma_res:
#            max_res.append(item1)
    fig, ax = plt.subplots(figsize=(12.80,8.20))
    ax.plot(x, data, label="data", color='orange')
    ax.plot(x, data_sort, label="standard deviation", color="r")
    ax.plot(x, sim_sort, label="Simulated data", color="k")
#    ax.plot(x, b, label="Baseline")
####    ax.hlines(hline, 0, len(data), label="5-sigma cut of deviations from the besaline")
#    ax.plot(xf, p, 'k', linewidth=2)
    ax.set_xlabel("Frequency channel")
    ax.set_ylabel("Standard deviation")
    ax.legend()
#    ax.set_ylim(0,0.03)
#    fig.savefig('{}.png'.format("J1909-3744.data.sim"))
    plt.show()
    plt.close("all")  


def find_sigma_range(data, sigma):
    data_sorted = np.sort(data)
    data_sorted = np.ma.masked_equal(data_sorted, 0.)
    pval = scipy.stats.mstats.normaltest(data_sorted)[1]
    skew = scipy.stats.mstats.skewtest(data_sorted)[0]
    p_sigma = scipy.stats.norm.sf(abs(sigma))
    while pval < p_sigma:
        if skew > 0:
            data_sorted = np.delete(data_sorted, np.argmax(data_sorted))
            pval = scipy.stats.mstats.normaltest(data_sorted)[1]
            skew = scipy.stats.mstats.skewtest(data_sorted)[0]
        if skew < 0:
            data_sorted = np.delete(data_sorted, np.argmin(data_sorted))
            pval = scipy.stats.mstats.normaltest(data_sorted)[1]
            skew = scipy.stats.mstats.skewtest(data_sorted)[0]
    max_data_norm = np.max(data_sorted)
    min_data_norm = np.min(data_sorted)
    return min_data_norm, max_data_norm



def chunk_data_bin(data):
    """divide the whole phase bins into parts with n phase bins"""
    n = 1024
    for i in xrange(nbin // n + 1):
        yield (i, data[:, :, :, (i * n) : ((i + 1) * n)])


def chunk_data_chan(data, k):
    """divide the whole phase bins into parts with n phase bins"""
    n = nchan//k
    for j in xrange(k):
        yield (j, data[:, :, (j * n) : ((j + 1) * n)])


def chunk_data_baseline(data, k):
    """divide the whole phase bins into parts with n phase bins"""
    n = nchan//k
    for j in xrange(k):
        yield (j, data[(j * n) : ((j + 1) * n)])

def clean_baseline(df):
        chunk_num = 4
        for s in xrange(nsub):
            stats = df.get_Integration(s).baseline_stats()
            stats = np.asarray(stats)
            stats = np.log(stats)
            chan_remove = []
            for c, chunk in chunk_data_baseline(stats[0, 0], chunk_num):
                min_select_bl, max_select_bl = max_simulate_find_sigma(chunk, 1.0)
#                print("std, ptp, fft = ", max_select_std, max_select_ptp, max_select_fft)
                #simulate_baseline_tester(chunk)


                for i, item in enumerate(chunk):
                    if item > max_select_bl or item < min_select_bl:
                        chan_remove.append(i + c*(nchan//chunk_num))
                        ar.get_Integration(s).set_weight(i + c*(nchan//chunk_num), 0.0)
            chan_list = np.sort(list(set(chan_remove)))
            print("{} channels are deleted for subint {}".format(len(chan_list), s))
        ar.unload("{}.{}".format(filename, "mohsen"))
        print("{}.{} is unloaded".format(filename, "mohsen"))



def intersection(lst1, lst2):
    lst1 = set(lst1)
    lst2 = set(lst2) 
    lst3 = [value for value in lst1 if value in lst2]  
    return lst3 


def apply_statistics(data, s, chan_divide_by):
    chunk_num = nchan//chan_divide_by
    data = data[:, :, :(nchan - nchan%chunk_num)]
    chan_remove = []
    xchan = data.shape[2]
    for c, chunk in chunk_data_chan(data, chunk_num):
        std_list = []
        ptp_list = []
        fft_list = []
        for j in xrange(xchan//chunk_num):
            std_list.append(np.std(chunk[s,: , j]))
            ptp_list.append(np.ptp(chunk[s, :, j]))
            fft_list.append(fft_peak(chunk[s, :, j]))
        min_select_std, max_select_std = max_simulate_find(std_list)
        min_select_ptp, max_select_ptp = max_simulate_find(ptp_list)
        min_select_fft, max_select_fft = max_simulate_find(fft_list)
        # min_select_std, max_select_std = find_sigma_range(std_list, 2.5)
        # min_select_ptp, max_select_ptp = find_sigma_range(ptp_list, 2.5)
        # min_select_fft, max_select_fft = find_sigma_range(fft_list, 2.5)
        for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
            if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft:
                chan_remove.append(i + c*(xchan//chunk_num))
    chan_remove = np.sort(list(set(chan_remove)))
    return chan_remove


def clean(data, filename):
    """Clean data based on the number of phase bins """

    for s in xrange(nsub):
        chan_remove_1 = apply_statistics(data, s, nchan/5)
        chan_remove_2 = apply_statistics(data, s, nchan/4)
        chan_remove = intersection(chan_remove_1, chan_remove_2)
        for item in chan_remove:
            ar.get_Integration(s).set_weight(item, 0.0)
        chan_list = np.sort(list(set(chan_remove)))
        print("{} channels are deleted at subint {}".format(len(chan_list), s))


    chunk_num = 1
    ffact = 8
    w3 = ar.get_weights()
    d3 = apply_weights(data, w3)
    data = np.sum(data, axis=0)[np.newaxis,...]
    data = np.add.reduceat(data, xrange(0,nchan,ffact), axis=2)
    print(data.shape)
    for s in xrange(data.shape[0]):
        chan_remove = []
        for c, chunk in chunk_data_chan(data, chunk_num):
            std_list = []
            ptp_list = []
            fft_list = []
            for j in xrange(nchan//ffact):
                std_list.append(np.std(chunk[s,: , j]))
                ptp_list.append(np.ptp(chunk[s, :, j]))
                fft_list.append(fft_peak(chunk[s, :, j]))
            # min_select_std, max_select_std = max_simulate_find(std_list)
            # min_select_ptp, max_select_ptp = max_simulate_find(ptp_list)
            # min_select_fft, max_select_fft = max_simulate_find(fft_list)

            min_select_std, max_select_std = find_sigma_range(std_list, 2)
            min_select_ptp, max_select_ptp = find_sigma_range(ptp_list, 2)
            min_select_fft, max_select_fft = find_sigma_range(fft_list, 2)




            for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
                if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft or \
                item_std < min_select_std or item_ptp < min_select_ptp or item_fft < min_select_fft:
                    #chan_remove.append(i + c*(nchan//chunk_num)
                    k = i + c*(data.shape[2]//chunk_num)
                    for p in xrange(k*ffact, (k+1)*ffact):
                         chan_remove.append(p)
    print("length of chan remove: ", len(chan_remove))
    for s in xrange(nsub):
        for i, item in enumerate(chan_remove):
            ar.get_Integration(s).set_weight(np.int(item), 0.0)
    chan_list = np.sort(list(set(chan_remove)))
    print("{} channels are deleted".format(len(chan_list)))

    ar.unload("{}.{}".format(filename, "mohsen"))
    print("{}.{} is unloaded".format(filename, "mohsen"))

#     # if nbin is less than or equal to 1024
#     if nbin//1024 <= 1:
#         chunk_num = 1
#         for s in xrange(nsub):
#             chan_remove = []
#             for c, chunk in chunk_data_chan(data, chunk_num):
#                 std_list = []
#                 ptp_list = []
#                 fft_list = []
#                 for j in xrange(nchan//chunk_num):
#                     std_list.append(np.std(chunk[s,: , j]))
#                     ptp_list.append(np.ptp(chunk[s, :, j]))
#                     fft_list.append(fft_peak(chunk[s, :, j]))
#                 min_select_std, max_select_std = max_simulate_find(std_list)
#                 min_select_ptp, max_select_ptp = max_simulate_find(ptp_list)
#                 min_select_fft, max_select_fft = max_simulate_find(fft_list)
# #                print("std, ptp, fft = ", max_select_std, max_select_ptp, max_select_fft)
# #                print("std, ptp, fft = ", min_select_std, min_select_ptp, min_select_fft)



#                 for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
#                     if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft or item_std < min_select_std or item_ptp < min_select_ptp or item_fft < min_select_fft:
#                         chan_remove.append(i + c*(nchan//chunk_num))
#                         ar.get_Integration(s).set_weight(i + c*(nchan//chunk_num), 0.0)
#             chan_list = np.sort(list(set(chan_remove)))
#             print("{} channels are deleted at subint {}".format(len(chan_list), s)) 
#         new_ar = ar.clone()
#         new_ar.tscrunch()
#         data = new_ar.get_data()
# #        weights = ar.get_weights()
# #        for isub in xrange(nsub):
# #            data[isub] = data[isub]*weights[isub,...,np.newaxis] #  Apply weights from a .ar file 
# #        data = np.sum(data, axis=0)
# #        data = data[np.newaxis,...]
#         print("data shape: ", data.shape)

#         chan_remove = []
#         for c, chunk in chunk_data_chan(data, chunk_num):
#             std_list = []
#             ptp_list = []
#             fft_list = []
#             for j in xrange(nchan//chunk_num):
#                 std_list.append(np.std(chunk[:,: , j]))
#                 ptp_list.append(np.ptp(chunk[:, :, j]))
#                 fft_list.append(fft_peak(chunk[:, :, j]))
#             min_select_std, max_select_std = max_simulate_find_sigma(std_list, 1)
#             min_select_ptp, max_select_ptp = max_simulate_find_sigma(ptp_list, 1)
#             min_select_fft, max_select_fft = max_simulate_find_sigma(fft_list, 1)
# #            print("std, ptp, fft = ", max_select_std, max_select_ptp, max_select_fft)
# #            print("std, ptp, fft = ", min_select_std, min_select_ptp, min_select_fft)



#             for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
#                 if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft or item_std < min_select_std or item_ptp < min_select_ptp or item_fft < min_select_fft:
#                     chan_remove.append(i + c*(nchan//chunk_num))
#         chan_list = np.sort(list(set(chan_remove)))
#         print("{} channels are deleted at tscrunched data".format(len(chan_list)))
#         for i in xrange(nsub):
#             for j, item in enumerate(chan_list):
#                 ar.get_Integration(i).set_weight(item, 0.0)
#         ar.unload("{}.{}".format(filename, "mohsen"))
#         print("{}.{} is unloaded".format(filename, "mohsen"))






#    # if nbin is more than 1025. cleaning is done by deviding the bins into several chunks
#    if nbin//1024 > 1:
#        print("Phase bins are chunked into {} parts".format(nbin//1024 + 1))
#        chan_remove = []
#        for s in xrange(nsub):
#            for c, chunk in chunk_data_bin(data):
#                std_list = []
#                ptp_list = []
#                fft_list = []
#                for j in xrange(nchan):
#                    std_list.append(np.std(chunk[s, :, j]))
#                    ptp_list.append(np.ptp(chunk[s, :, j]))
#                    fft_list.append(fft_peak(chunk[s, :, j]))
#       
#                max_select_std = max_simulate_find(std_list)
#                max_select_ptp = max_simulate_find(ptp_list)
#                max_select_fft = max_simulate_find(fft_list)
##               print("std, ptp, fft = ", max_select_std, max_select_ptp, max_select_fft)
#
#
#                for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
#                    if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft:
#                        chan_remove.append(i)
#                        ar.get_Integration(s).set_weight(i, 0.0)
#        ar.unload("{}.{}".format(filename, "mohsen"))
#        print("{}.{} is unloaded".format(filename, "mohsen"))
#        chan_list = np.sort(list(set(chan_remove))) 
#        print("{} channels are deleted".format(len(chan_list)))
        
if __name__ == '__main__':
    scr_factor = 10
    for filename in sys.argv[1:-1]:
        print("="*60) 
        start = timer()
        ar, data = load_data(filename, sys.argv[-1])
        nsub, npol, nchan, nbin = data.shape
        clean(data, filename)
#        clean_baseline(ar)
        print("Cleaning time: {} sec".format(timer()-start))









