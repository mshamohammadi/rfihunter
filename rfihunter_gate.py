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
    patient = ar.clone()
    patient.remove_baseline()
    patient.pscrunch()
    patient.dedisperse()
    data = patient.get_data()
    data_offpulse = main_pulse_wash(data, gate_file)
    return ar, data_offpulse


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
    xr1, xr2 = np.int(16*l/100) , np.int(50*l/100)
    np.random.seed(0)
    sim = np.random.normal(0, get_std(l, xr1, xr2) * np.std(data_sort[xr1 : xr2]) , l)
    shift_ind = np.int((xr1 + xr2)/2.0)
    sim_sort = np.sort(sim)
    diff = data_sort[shift_ind] - sim_sort[shift_ind]
    sim = sim + diff
    max_sim = np.max(sim)
    return max_sim





def chunk_data(data):
    """divide the whole phase bins into parts with n phase bins"""
    n = 1024
    for i in xrange(nbin // n + 1):
        yield (i, data[:, :, :, (i * n) : ((i + 1) * n)])




def clean(data, filename):
    """Clean data based on the number of phase bins """

    # if nbin is less than or equal to 1024
    if nbin//1024 <= 1:
        chan_remove = []
        for s in xrange(nsub):
            "subint = {}".format(s)
            std_list = []
            ptp_list = []
            fft_list = []
            for j in xrange(nchan):
                std_list.append(np.std(data[s,: , j]))
                ptp_list.append(np.ptp(data[s, :, j]))
                fft_list.append(fft_peak(data[s, :, j]))


            max_select_std = max_simulate_find(std_list)
            max_select_ptp = max_simulate_find(ptp_list)
            max_select_fft = max_simulate_find(fft_list)
#            print("std, ptp, fft = ", max_select_std, max_select_ptp, max_select_fft)



            for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
                if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft:
                    chan_remove.append(i)
                    ar.get_Integration(s).set_weight(i, 0.0)
        ar.unload("{}.{}".format(filename, "mohsen"))
        print("{}.{} is unloaded".format(filename, "mohsen"))
        chan_list = np.sort(list(set(chan_remove)))
        print("{} channels are deleted".format(len(chan_list)))






    # if nbin is more than 1025. cleaning is done by deviding the bins into several chunks
    if nbin//1024 > 1:
        print("Phase bins are chunked into {} parts".format(nbin//1024 + 1))
        chan_remove = []
        for s in xrange(nsub):
            for c, chunk in chunk_data(data):
                std_list = []
                ptp_list = []
                fft_list = []
                for j in xrange(nchan):
                    std_list.append(np.std(chunk[s, :, j]))
                    ptp_list.append(np.ptp(chunk[s, :, j]))
                    fft_list.append(fft_peak(chunk[s, :, j]))
       
                max_select_std = max_simulate_find(std_list)
                max_select_ptp = max_simulate_find(ptp_list)
                max_select_fft = max_simulate_find(fft_list)
#               print("std, ptp, fft = ", max_select_std, max_select_ptp, max_select_fft)


                for i, (item_std, item_ptp, item_fft) in enumerate(zip(std_list, ptp_list, fft_list)):
                    if item_std > max_select_std or item_ptp > max_select_ptp or item_fft > max_select_fft:
                        chan_remove.append(i)
                        ar.get_Integration(s).set_weight(i, 0.0)
        ar.unload("{}.{}".format(filename, "mohsen"))
        print("{}.{} is unloaded".format(filename, "mohsen"))
        chan_list = np.sort(list(set(chan_remove))) 
        print("{} channels are deleted".format(len(chan_list)))
        
if __name__ == '__main__':
    for filename in sys.argv[1:-1]:
        print("="*60) 
        start = timer()
        ar, data = load_data(filename, sys.argv[-1])
        nsub, npol, nchan, nbin = data.shape
        clean(data, filename)
        print("Cleaning time: {} sec".format(timer()-start))
