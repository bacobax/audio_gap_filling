# -*- coding: utf-8

"""
Thomas Grill, 2011-2016
http://grrrr.org/nsgt

--
Original matlab code comments follow:

NSGFWIN.M
---------------------------------------------------------------
 [g,rfbas,M]=nsgfwin(fmin,bins,sr,Ls) creates a set of windows whose
 centers correspond to center frequencies to be
 used for the nonstationary Gabor transform with varying Q-factor. 
---------------------------------------------------------------

INPUT : fmin ...... Minimum frequency (in Hz)
        bins ...... Vector consisting of the number of bins per octave
        sr ........ Sampling rate (in Hz)
        Ls ........ Length of signal (in samples)

OUTPUT : g ......... Cell array of window functions.
         rfbas ..... Vector of positions of the center frequencies.
         M ......... Vector of lengths of the window functions.

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

EXTERNALS : firwin
"""

import numpy as np
import torch


def hannwin(l, device="cpu"):
    r = torch.arange(l, dtype=float, device=torch.device(device))
    r *= np.pi * 2.0 / l
    r = torch.cos(r)
    r += 1.0
    r *= 0.5
    return r


def kaiserwin(l, beta, device="cpu"):
    beta = torch.tensor(beta, dtype=float, device=torch.device(device))
    r = torch.arange(l, dtype=float, device=torch.device(device))
    r *= np.pi * 2.0 / l
    r = torch.cos(r)
    r += 1.0
    r *= 0.5
    r = torch.sqrt(r)
    r = torch.i0(beta * torch.sqrt(1.0 - r ** 2)) / (2.0 * torch.i0(beta))
    r = torch.roll(r, l // 2)
    return r


def blackharr(n, l=None, mod=True, device="cpu"):
    if l is None:
        l = n
    nn = (n // 2) * 2
    k = torch.arange(n, device=torch.device(device))
    if not mod:
        bh = (
            0.35875
            - 0.48829 * torch.cos(k * (2 * torch.pi / nn))
            + 0.14128 * torch.cos(k * (4 * torch.pi / nn))
            - 0.01168 * torch.cos(k * (6 * torch.pi / nn))
        )
    else:
        bh = (
            0.35872
            - 0.48832 * torch.cos(k * (2 * torch.pi / nn))
            + 0.14128 * torch.cos(k * (4 * torch.pi / nn))
            - 0.01168 * torch.cos(k * (6 * torch.pi / nn))
        )
    bh = torch.hstack((bh, torch.zeros(l - n, dtype=bh.dtype, device=torch.device(device))))
    bh = torch.hstack((bh[-n // 2 :], bh[: -n // 2]))
    return bh


from math import ceil
from warnings import warn
from itertools import chain


def nsgfwin(f, q, sr, Ls,  min_win=4, Qvar=1, dowarn=True, dtype=np.float64, device="cpu", window="hann"):
    nf = sr/2.

    lim = np.argmax(f > 0)
    if lim != 0:
        # f partly <= 0 
        f = f[lim:]
        q = q[lim:]
            
    lim = np.argmax(f >= nf)
    if lim != 0:
        # f partly >= nf 
        f = f[:lim]
        q = q[:lim]
    
    assert len(f) == len(q)
    assert np.all((f[1:]-f[:-1]) > 0)  # frequencies must be increasing
    assert np.all(q > 0)  # all q must be > 0
    
    qneeded = f*(Ls/(8.*sr))
    #if np.any(q >= qneeded) and dowarn:
    #    warn("Q-factor too high for frequencies %s"%",".join("%.2f"%fi for fi in f[q >= qneeded]))
    
    fbas = f
    lbas = len(fbas)
    
    frqs = np.concatenate(((0.,),fbas,(nf,)))
    
    fbas = np.concatenate((frqs,sr-frqs[-2:0:-1]))

    # at this point: fbas.... frequencies in Hz
    
    fbas *= float(Ls)/sr
    
    # Omega[k] in the paper
    M = np.zeros(fbas.shape, dtype=int)
    M[0] = np.round(2*fbas[1])
    #M[1]=
    M[1] = np.round(fbas[1]/q[0])
    for k in range(2,lbas+1):
        #M[k] = np.round(fbas[k]/q[k-1])
        M[k]= np.round(fbas[k+1]-fbas[k-1]) #this is nyq!
        #M[k] =
    #M[lbas]=np.round(fbas[lbas]/q[-1])
    M[lbas+1]= np.round(fbas[k+1]-fbas[k-1]) #this is nyq!
    M[lbas+2:]=M[lbas:0:-1] #symmetry!
    
    #M[-1] = np.round(Ls-fbas[-2])
        
    # Ensure window lengths are at least `min_win` while preserving integer type
    # The original implementation used ``np.clip`` with ``out=M`` which fails
    # when ``M`` has an integer dtype. Using ``np.maximum`` avoids the casting
    # error by keeping the array type intact.
    M = np.maximum(M, min_win)

    
    if window=="hann":
        print("using a hann window")
        g = [hannwin(m, device=device).to(dtype) for m in M]
    elif window=="blackharr":
        print("using a blackharr window")
        g = [blackharr(m, device=device).to(dtype) for m in M]
    elif window[0]=="kaiser":
        print("using a kaiser window with beta=",window[1])
        str, beta= window
        g = [kaiserwin(m,beta, device=device).to(dtype) for m in M]

    #g[0]=tukeywin(M[0], 0.2, device=device).to(dtype)
    
    fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
    fbas[lbas+2] = Ls-fbas[lbas]
    rfbas = np.round(fbas).astype(int)
        

    return g,rfbas,M
