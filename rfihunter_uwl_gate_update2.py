#!/usr/bin/env python

# RFI Hunter
# This program cleans RFI affected frequency channels
# *.py PSRCHIVE*.ar gatefile
# gatefile has comma separated start and end bins of the main pulse
# made by Mohsen Shamohammadi msh.ph.ir@gmail.com
from __future__ import print_function
import numpy as np
import psrchive as psr
import sys
from timeit import default_timer as timer

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
    print('nsub={} npol={} nchan={} nbin={}'.format(ar.get_nsubint(), ar.get_npol(), ar.get_nchan(), ar.get_nbin()))
    patient = ar.clone()
    patient.remove_baseline()
    patient.pscrunch()
    patient.dedisperse()
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
    data = data.squeeze()
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
    data_sort = np.ma.masked_equal(data, 0.0)
    xr1, xr2 = np.int(15*l/100) , np.int(50*l/100)
    np.random.seed(0)
    sim = np.random.normal(0, get_std(l, xr1, xr2) * np.std(data_sort[xr1 : xr2]) , l)
    shift_ind = np.int((xr1 + xr2)/2.0)
    sim_sort = np.sort(sim)
    diff = data_sort[shift_ind] - sim_sort[shift_ind]
    sim = sim + diff
    max_sim = np.max(sim)
    return max_sim

def intersection(lst1, lst2):
    lst1 = set(lst1)
    lst2 = set(lst2) 
    lst3 = [value for value in lst1 if value in lst2]  
    return lst3 



def chunk_data_bin(data):
    """divide the whole phase bins into parts with n phase bins"""
    xbin = data.shape[3]
    n = 1024
    for i in xrange(xbin // n + 1):
        yield (i, data[:, :, :, (i * n) : ((i + 1) * n)])


def chunk_data_chan(data, k):
    """divide the whole phase bins into parts with n phase bins"""
    xchan = data.shape[2]
    n = xchan//k
    for j in xrange(k):
        yield (j, data[:, :, (j * n) : ((j + 1) * n)])



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
        max_select_std = max_simulate_find(std_list)
        max_select_ptp = max_simulate_find(ptp_list)
        max_select_fft = max_simulate_find(fft_list)
        for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
            if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft:
                chan_remove.append(i + c*(xchan//chunk_num))
    chan_remove = np.sort(list(set(chan_remove)))
    return chan_remove


def clean(data, filename):
#    """Clean data based on the number of phase bins """

    # if nbin is less than or equal to 1024
#    if nbin//1024 <= 1:
        for s in xrange(nsub):
            chan_remove_1 = apply_statistics(data, s, 832)
            chan_remove_2 = apply_statistics(data, s, 665)
            chan_remove = intersection(chan_remove_1, chan_remove_2)
            for item in chan_remove:
                ar.get_Integration(s).set_weight(item, 0.0)
            chan_list = np.sort(list(set(chan_remove)))
            print("{} channels are deleted at subint {}".format(len(chan_list), s))

        chunk_num = 4
        chan_remove_3 = []
        w3 = ar.get_weights()
        d3 = apply_weights(data, w3)
        bfact = nchan/1664
        dbin = np.add.reduceat(d3, xrange(0,nchan,bfact), axis=2)
        dbin = np.sum(dbin, axis=0)[np.newaxis, ...]
        bchan = nchan/bfact

        for c, chunk in chunk_data_chan(dbin, chunk_num):
            std_list = []
            ptp_list = []
            fft_list = []
            for j in xrange(bchan//chunk_num):
                std_list.append(np.std(chunk[:,: , j]))
                ptp_list.append(np.ptp(chunk[:, :, j]))
                fft_list.append(fft_peak(chunk[:, :, j]))
            max_select_std = max_simulate_find(std_list)
            max_select_ptp = max_simulate_find(ptp_list)
            max_select_fft = max_simulate_find(fft_list)
            ilist = []
            for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
                if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft:
                    k = i + c*(bchan//chunk_num)
                    ilist.append(k)
                    for p in xrange(k*bfact, (k+1)*bfact):
                        chan_remove_3.append(p)
        chan_remove_3 = np.sort(list(set(chan_remove_3)))
        print("{} channels are deleted at tscrunched".format(len(chan_remove_3)))

        for s in xrange(nsub):
            for item in chan_remove_3:
                ar.get_Integration(s).set_weight(np.int(item), 0.0)
                


        ar.unload("{}.{}".format(filename, "mohsen"))
        print("{}.{} is unloaded".format(filename, "mohsen"))






    # if nbin is more than 1025. cleaning is done by deviding the bins into several chunks
    # if nbin//1024 > 1:
    #     print("Phase bins are chunked into {} parts".format(nbin//1024 + 1))
    #     chan_remove = []
    #     for s in xrange(nsub):
    #         for b, chunkb in chunk_data_bin(data):
    #             for c, chunkc in chunk_data_chan(chunkb, chunk_num):
    #                 std_list = []
    #                 ptp_list = []
    #                 fft_list = []
    #                 for j in xrange(nchan//chunk_num):
    #                     std_list.append(np.ma.std(chunkc[s, :, j]))
    #                     ptp_list.append(np.ma.ptp(chunkc[s, :, j]))
    #                     fft_list.append(fft_peak(chunkc[s, :, j]))
        
    #                 max_select_std = max_simulate_find(std_list)
    #                 max_select_ptp = max_simulate_find(ptp_list)
    #                 max_select_fft = max_simulate_find(fft_list)
    # #               print("std, ptp, fft = ", max_select_std, max_select_ptp, max_select_fft)


    #                 for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
    #                     if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft:
    #                         chan_remove.append(i)
    #                         ar.get_Integration(s).set_weight(i, 0.0)
    #         chan_list = np.sort(list(set(chan_remove)))
    #         print("{} channels are deleted at subint {}".format(len(chan_list), s))
    #     ar.unload("{}.{}".format(filename, "mohsen"))
    #     print("{}.{} is unloaded".format(filename, "mohsen"))
        
if __name__ == '__main__':
    for filename in sys.argv[1:-1]:
        print("="*60) 
        start = timer()
        ar, data = load_data(filename, sys.argv[-1])
        nsub, npol, nchan, nbin = data.shape
        clean(data, filename)
        print("Cleaning time: {} sec".format(timer()-start)) 
