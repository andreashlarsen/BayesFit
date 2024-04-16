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

def calc_y(x,a,M,model,p):
    """
    function to calculate y
    used by convert_function()
    """
    K = len(p)
    if len(M) == 1:
        q = x[:-K]
        y1 = model(q,*p)
        yS = a*p
        y = np.concatenate((y1,yS),axis=None)
    elif len(M) == 2:
        q1 = x[:-(K+M[1])]
        q2 = x[M[0]:-K]
        y1,y2 = model(q1,q2,*p)
        yS = a*p
        y = np.concatenate((y1,y2,yS),axis=None)
    elif len(M) == 3:
        q1 = x[:-(K+M[1]+M[2])]
        q2 = x[M[0]:-(K+M[2])]
        q3 = x[(M[0]+M[1]):-K]
        y1,y2,y3 = model(q1,q2,q3,*p)
        yS = a*p
        y = np.concatenate((y1,y2,y3,yS),axis=None)
    elif len(M) == 4:
        q1 = x[:-(K+M[1]+M[2]+M[3])]
        q2 = x[M[0]:-(K+M[2]+M[3])]
        q3 = x[(M[0]+M[1]):-(K+M[3])]
        q4 = x[(M[0]+M[1]+M[2]):-K]
        y1,y2,y3,y4 = model(q1,q2,q3,q4,*p)
        yS = a*p
        y = np.concatenate((y1,y2,y3,y4,yS),axis=None)
    elif len(M) == 5:
        q1 = x[:-(K+M[1]+M[2]+M[3]+M[4])]
        q2 = x[M[0]:-(K+M[2]+M[3]+M[4])]
        q3 = x[(M[0]+M[1]):-(K+M[3]+M[4])]
        q4 = x[(M[0]+M[1]+M[2]):-(K+M[4])]
        q5 = x[(M[0]+M[1]+M[2]+M[3]):-K]
        y1,y2,y3,y4,y5 = model(q1,q2,q3,q4,q5,*p)
        yS = a*p
        y = np.concatenate((y1,y2,y3,y4,y5,yS),axis=None)
    elif len(M) > 5:
        print('ERROR: bayesfit cannot (yet) handle more than 5 datasets simultaneously - please contact the developers')
        exit(-1)
    return y

def convert_function(a,K,M,model):
    """
    This function reformulates a function so it can be used by curve_fit
    """

    if K == 1:
        def func(x,p1):
            p = np.array([p1])
            return calc_y(x,a,M,model,p)
    elif K == 2:
        def func(x,p1,p2):
            p = np.array([p1,p2])
            return calc_y(x,a,M,model,p)
    elif K == 3:
        def func(x,p1,p2,p3):
            p = np.array([p1,p2,p3])
            return calc_y(x,a,M,model,p)
    elif K == 4:
        def func(x,p1,p2,p3,p4):
            p = np.array([p1,p2,p3,p4])
            return calc_y(x,a,M,model,p)
    elif K == 5:
        def func(x,p1,p2,p3,p4,p5):
            p = np.array([p1,p2,p3,p4,p5])
            return calc_y(x,a,M,model,p)
    elif K == 6:
        def func(x,p1,p2,p3,p4,p5,p6):
            p = np.array([p1,p2,p3,p4,p5,p6])
            return calc_y(x,a,M,model,p)
    elif K == 7:
        def func(x,p1,p2,p3,p4,p5,p6,p7):
            p = np.array([p1,p2,p3,p4,p5,p6,p7])
            return calc_y(x,a,M,model,p)
    elif K == 8:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8])
            return calc_y(x,a,M,model,p)
    elif K == 9:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9])
            return calc_y(x,a,M,model,p)
    elif K == 10:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])
            return calc_y(x,a,M,model,p)
    elif K == 11:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11])
            return calc_y(x,a,M,model,p)
    elif K == 12:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])
            return calc_y(x,a,M,model,p)
    elif K == 13:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13])
            return calc_y(x,a,M,model,p)
    elif K == 14:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14])
            return calc_y(x,a,M,model,p)
    elif K == 15:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15])
            return calc_y(x,a,M,model,p)
    elif K == 16:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16])
            return calc_y(x,a,M,model,p)
    elif K == 17:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17])
            return calc_y(x,a,M,model,p)
    elif K == 18:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18])
            return calc_y(x,a,M,model,p)
    elif K == 18:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19])
            return calc_y(x,a,M,model,p)
    elif K == 18:
        def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20):
            p = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20])
            return calc_y(x,a,M,model,p)
    elif K > 20:
        print('   ERROR: bayesfit cannot (yet) handle models with more than 20 parameters - please contact the developers')
        exit(-1)

    return func

