import numpy as np
import string

def check_line(skip_line,line,CONTINUE):
    """
    check if a line is a header or footer line
    also check for zero and nan
    """

    tmp = line.split()
    imax = 3
    if len(tmp) < imax:
            imax = len(tmp)
    try:
        NAN = 0
        for i in range(imax):
            1/float(tmp[i]) # divide to ensure non-zero values
            if np.isnan(float(tmp[i])):
                NAN = 1
        if NAN:
            skip_line += 1
        else:
            CONTINUE = False
    except:
        skip_line+=1

    return skip_line,CONTINUE

def get_header_footer(file):
    """
    get number of headerlines and footerlines
    """

    f = open(file)
    try:
        lines = f.readlines()
    except:
        print('Error: cannot read lines of file. Do you have some special characters in the file? Try removing them and rerun')
        print('file: %s' % file)

    CONTINUE_H,CONTINUE_F = True,True
    header,footer,j = 0,0,0

    while CONTINUE_H or CONTINUE_F:

        # check if next line from top/bottom of file is a header/footer (or contains zero or nan)
        header,CONTINUE_H = check_line(header,lines[j],CONTINUE_H)
        footer,CONTINUE_F = check_line(footer,lines[-1-j],CONTINUE_F)

        # stop if there are no more lines, else continue to next line
        j += 1
        if j == len(lines):
            CONTINUE_H = False
            CONTINUE_F = False

    return header,footer

def remove(string):
    """
    in a string, remove space
    """
    return string.replace(" ", "")

def remove_comma(string):
    """
    in a string, replace comma by space
    """
    return string.replace(","," ")

def av(x,w):
    """
    weighted average
    """
    d = len(np.shape(x))
    x_av = np.average(x,weights=w,axis=-d)
    return x_av

def check_input(IN,default):
    """
    check a yes/no (boolean) input and convert to 0 or 1
    default: the default answer if IN is empty
    """
    if not IN:
        OUT = default
        print('\n      default used')
    elif IN in ['0','1']:
        OUT = int(IN)
    elif remove(IN) in ['yes','YES','y','Y','ja','JA','yep','YEP']:
        OUT = 1
    elif remove(IN) in ['no','NO','n','N','nej','NEJ','nope','NOPE']:
        OUT = 0
    if OUT:
        print('\n      selected answer: yes\n')
    else:
        print('\n      selected answer: no\n')
    return OUT

def get_fit(q,model,p):
    """
    get the fit with q, the model and parameters (p) as input
    """
    if len(q) == 1:
        fit = model(q[0],*p)
    elif len(q) == 2:
        fit = model(q[0],q[1],*p)
    elif len(q) == 3:
        fit = model(q[0],q[1],q[2],*p)
    elif len(q) == 4:
        fit = model(q[0],q[1],q[2],q[3],*p)
    elif len(q) == 5:
        fit = model(q[0],q[1],q[2],q[3],q[4],*p)
    elif len(M) > 5:
        print('       ERROR: bayesfit cannot (yet) handle more than 5 datasets simultaneously - please contact the developers')
        exit(-1)
    if len(q) == 1:
        return [fit]
    else:
        return fit

def fit_merge(fit):
    fit_merge = []
    for i in range(len(fit)):
        fit_merge = np.concatenate((fit_merge,fit[i]),axis=None)
    return fit_merge

def kl_divergence(posterior,prior):
    """
    calculate KLD by nummerical integration
    """
    idx = np.where(posterior>0)
    if len(idx[0]) == 0:
        KLD = 0
    else:
        KLD = np.sum(posterior[idx]*np.log2(posterior[idx]/prior[idx]))/np.sum(posterior[idx])
    return KLD

def kl_divergence0(mu_post,sig_post,mu_prior,sig_prior):
    """
    calculate KLD by analytical expression, valid for normal (prior and posterior) distributions 
    """
    var_post = sig_post**2
    var_prior = sig_prior**2
    R = mu_prior-mu_post
    var_ratio = var_post/var_prior
    KLD = (R**2/var_prior + var_ratio-np.log(var_ratio)-1)/2
    KLD /= np.log(2) # convert from nat to bit  
    return KLD

