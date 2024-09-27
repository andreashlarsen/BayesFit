"""
BayesFit
version 2.8

see instructions for running at https://github.com/andreashlarsen/BayesFit

"""

##############################
### IMPORT PYTHON PACKAGES ###
##############################
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from io import StringIO
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.special import rel_entr
from shutil import rmtree

###################################
### IMPORT OWN PYTHON FUNCTIONS ###
###################################
from bayesfit_functions import *
import formfactors as ff

##############################
### PLOT ON LOG-LOG SCALE? ###
##############################
LOG = 0

#######################
### READ USER INPUT ###
#######################

## OUTPUT NAME
print('-- INPUT: OUTPUT DIRECTORY --\n')
output_dir = input('   output directory (full path):')
print(' ')
if not output_dir:
    print('      default used')
    output_dir = 'output_bayesfit'
os.makedirs(output_dir,exist_ok=True)
print('      selected output dir: %s\n' % output_dir)

## NUMBER OF CONTRASTS
print('-- INPUT: NUMBER OF CONTRASTS --\n')
Ncontrasts_str = input('   number of contrasts (datasets): ')
print(' ')
if not Ncontrasts_str:
    print('      default used')
    Ncontrasts_str = '1'
Ncontrasts = int(Ncontrasts_str)
print('      contrasts: %s\n' % Ncontrasts)

## DATA FILE(S)
print('-- INPUT: DATA  --')
M,q,I,dI,dataname = [],[],[],[],[]
for ii in range(Ncontrasts):
    if Ncontrasts == 1:
        datapath  = input('   dataset (full path): ')
        if not datapath:
            print('   default used')
            datapath = 'examples/cylinder/Isim_1.dat'
    else: 
        datapath  = input('   dataset %d (full path) :' % (ii+1))
    print('\n      data : %s' % datapath)
    skip_header,skip_footer = get_header_footer(datapath)
    print('      number of headerlines : %d\n      number of footerlines : %d' % (skip_header,skip_footer))
    q_ii,I_ii,dI_ii = np.genfromtxt(datapath.strip(),usecols=[0,1,2],skip_header=skip_header,skip_footer=skip_footer,unpack=True)
    M_ii = len(q_ii)
    print('      number of points in dataset %d : %d' % (ii,M_ii))
    M.append(M_ii)
    q.append(q_ii)
    I.append(I_ii)
    dI.append(dI_ii)
    dataname.append(datapath.split('/')[-1])
print(' ')

## MODEL USED TO FIT DATA
print('-- INPUT: SELECT MODEL --')
print('   available models:')
#           name      model function  parameter-names
models = [ 
          ['cylinder',ff.cylinder,['radius','length','scaling','background']],
          ['nanodisc',ff.nanodisc,['Bg','c','V_l','V_t','CV_p','Nlip','T','sigmaR','Ar','eps','n_w','Rg']],
          ['sphere3',ff.sphere3,['radius','scale','background']],
          ['coreshell4_2',ff.coreshell4_2,['r1','r2','r3','r4','pX1','pX2','pX3','pX4','pN1','pN2','pN3','pN4','sX','bX','sN','bN']],
          ['coreshell3_2',ff.coreshell3_2,['r1','r2','r3','pX1','pX2','pX3','pN1','pN2','pN3','sX','bX','sN','bN']],
          ['coreshell3',ff.coreshell3,['r1','r2','r3','p1','p2','p3','sX','bX']],
          ['coreshell2',ff.coreshell2,['r1','r2','p1','p2','s','b']],
          ['coreshell2_ratio',ff.coreshell2_ratio,['R1','R2','r2','s','b']],
          ['coreshell3_ratio',ff.coreshell3_ratio,['R1','R2','R3','r2','r3','s','b']],
          ['coreshell4_ratio',ff.coreshell4_ratio,['R1','R2','R3','R4','r2','r3','r4','s','b']],
          ['coreshell4_ratio_2',ff.coreshell4_ratio_2,['R1','R2','R3','R4','rX2','rX3','rX4','rN2','rN3','rN4','sX','bX','sN','bN']],
          ['coreshell4_ratio_2_T',ff.coreshell4_ratio_2_T,['R1','R2','R3','R4','rX2','rX3','rX4','rN2','rN3','rN4','sX','bX','sN','bN']],
          ['coreshell4_ratio_2_res',ff.coreshell4_ratio_2_res,['R1','R2','R3','R4','rX2','rX3','rX4','rN2','rN3','rN4','sX','bX','sN','bN']],
          ['coreshell4_ratio_3',ff.coreshell4_ratio_3,['R1','R2','R3','R4','rX2','rX3','rX4','rN2','rN3','rN4','sX','bX','sN1','bN1','sN2','bN2']],
          ['coreshell4_ratio_HS_2',ff.coreshell4_ratio_HS_2,['R1','R2','R3','R4','rX2','rX3','rX4','rN2','rN3','rN4','sX','bX','sN','bN','eta']],
          ['stacked_2cyl_sameR_ratio',ff.stacked_2cyl_sameR_ratio,['R','L1','L2','p21','s','b']],
          ['stacked_3cyl_sameR_ratio',ff.stacked_3cyl_sameR_ratio,['R','L1','L2','L3','p21','p31','s','b']],
          ['stacked_3cyl_sameR_ratio_smooth',ff.stacked_3cyl_sameR_ratio_smooth,['R','L1','L2','L3','p21','p31','sigma','s','b']],
          ['stacked_3cyl_sameR_ratio_2',ff.stacked_3cyl_sameR_ratio_2,['R','L1','L2','L3','pX21','pX31','pN21','pN31','sX','bX','sN','bN']],
          ['cylinder_dimer',ff.cylinder_dimer,['R','L','s','b']],
          ['cylinder_dimer_ratio',ff.cylinder_dimer_ratio,['R','L','p','s','b']],
          ['cylinder_trimer_ratio',ff.cylinder_trimer_ratio,['R','L','p1','p2','s','b']],
          ['cylinder_trimer_ratio_L',ff.cylinder_trimer_ratio_L,['R','L1','L2','L3','p21','p31','scaling','background']],
          ['ellipsoid',ff.ellipsoid,['short_axis','long_axis','scaling','background']],
          ['ellipsoid_eps',ff.ellipsoid_eps,['short_axis','elipticity','scaling','background']],
          ['elliptical_cylinder',ff.elliptical_cylinder,['short_axis','length','elliticity','scaling','background']],
          ['elliptical_cylinder_coreshell',ff.elliptical_cylinder_coreshell,['short_axis','length total','length core','elliticity','contrast shell/contrast core','scaling','background']],
          ['elliptical_nanodisc',ff.elliptical_nanodisc,['short_axis','height total','height core','height belt','thickness belt','elliticity','contrast shell/core','contrast belt/core','scaling','background']],
          ['elliptical_torus',ff.elliptical_torus,['short_axis','long_axis','radius','scaling','background']],
          ['torus',ff.torus,['radius_cross_section','radius','scaling','background']],
          ['supercylinder',ff.supercylinder,['radius','shape_parameter_t','ellipticity','scaling','background']],
          ['hollow_supercylinder',ff.hollow_supercylinder,['radius','shape_parameter_t','ellipticity','hole_radius','scaling','background']],
          ['hollow_supercylinder_torus',ff.hollow_supercylinder_torus,['radius','shape_parameter_t','ellipticity','hole_radius','torus_cross_radius','torus_center_radius','scaling','background']],
          ['hollow_supercylinder_torus_smooth',ff.hollow_supercylinder_torus_smooth,['radius','shape_parameter_t','ellipticity','hole_radius','torus_cross_radius','torus_center_radius','smoothness','scaling','background']],
          ['hollow_cylinder',ff.hollow_cylinder,['radius_inner','radius_outer','length','scaling','background']],
          ['cylinder_torus',ff.cylinder_torus,['radius_inner','radius_outer','length','torus_cross_radius','torus_center_radius','torus_shift','scaling','background']],
          ['cylinder_torus_fix',ff.cylinder_torus_fix,['cylinder_radius_inner','cylinder_length','torus_cross_radius','torus_center_radius','scaling','background']],
          ['hollow_cylinders_spaced_torus_fix',ff.hollow_cylinders_spaced_torus_fix,['cylinder_radius_inner','cylinder_length','cylinder_space_distance','torus_cross_radius','torus_center_radius','scaling','background']],
          ['cylinders_spaced',ff.cylinders_spaced,['cylinder_radius','length_total','length_space','scaling','background']],
          ['hollow_cylinders_spaced',ff.hollow_cylinders_spaced,['inner_radius','outer_radius','length','distance_space','scaling','background']],
          ['hollow_cylinder_crown',ff.hollow_cylinder_crown,['inner_radius','outer_radius','length','crown_inner_radius','crown_outer_radius','length_crown','crown_shift','scaling','background']],
          ['ellipsoid_torus',ff.ellipsoid_torus,['minor_radius','ellipticity','torus_inner_radius','torus_outer_radius','torus_shift','scaling','background']],
          ['tri_tori',ff.tri_tori,['cross-sec radius 1','center radius 1','cross sec radius 2','center radius 2','shift 2','cross sec radius 3','center radius 3','shift 3','scaling','background']],
          ['four_tori',ff.four_tori,['cross-sec radius 1','center radius 1','cross sec radius 2','center radius 2','shift 2','cross sec radius 3','center radius 3','shift 3','cross sec radius 4','center radius 4','shift 4','scaling','background']],
          ['four_tori_constraint',ff.four_tori_constraint,['cross section radius 1','center radius 1','cross section radius 3','cross section radius 4','center radius 4','spacer','scaling','background']],
          ['four_elliptical_tori_constraint',ff.four_elliptical_tori_constraint,['cross section radius 1','ellipticity 1','center radius 1','cross section radius 3','ellipticity 3','cross section radius 4','center radius 4','ellipticity 4','scaling','background']],
          ['ellipsoid_channel',ff.ellipsoid_channel,['ellipsoid_radius','ellipsoid_ellipticity','channel radius','channel_length','scaling','background']],
          ['ellipsoid_channel_torus',ff.ellipsoid_channel_torus,['ellipsoid_radius','ellipsoid_ellipticity','channel radius','torus_cross_radius','torus_center_radius','scaling','background']],
          ['MD_SAXS',ff.md_saxs,['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','scale','background']],
          ['MD_SANS',ff.md_sans,['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','scale','background']],
          ['MD_SAS',ff.md_sas,['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','sx','bx','sn','bn']],
          ['MD_SAXS_w4',ff.md_saxs_w4,['w1','w2','w3','w4','sx','bx']],
          ['MD_SANS_w4',ff.md_sans_w4,['w1','w2','w3','w4','sn','bn']],
          ['MD_SAS_w4',ff.md_sas_w4,['w1','w2','w3','w4','sx','bx','sn','bn']],
          ['MD_SAXS_w3',ff.md_saxs_w3,['w1','w2','w3','sx','bx']],
          ['MD_SANS_w3',ff.md_sans_w3,['w1','w2','w3','sn','bn']],
          ['MD_SAS_w3',ff.md_sas_w3,['w1','w2','w3','sx','bx','sn','bn']],
          ['MD_SAXS_w2',ff.md_saxs_w2,['w1','w2','b1','b2']],
          ['MD_SANS_w2',ff.md_sans_w2,['w1','w2','b1','b2']],
          ['MD_SAS_w2',ff.md_sas_w2,['w1','w2','bx1','bx2','bn1','bn2']],
          ['MD_SAXS_ff2_b',ff.md_saxs_ff2_b,['w1','w2','bx1','bx2']],
          ['MD_SAXS_ff2',ff.md_saxs_ff2,['w1','w2']],
          ['MD_SAXS_ff2_w1',ff.md_saxs_ff2_w1,['w1','scale']],
          ['MD_SANS_ff2_w1',ff.md_sans_ff2_w1,['w1','scale']],
          ['MD_SAS_ff2_w1',ff.md_sas_ff2_w1,['w1','scale_x','scale_n']],
          ['MD_SAS_ff2_w1_nos',ff.md_sas_ff2_w1_nos,['w1']],
         ]

for i in range(len(models)):
    name,model,p_name = models[i]
    print('   %d: %s' % (i,name))
modelinput = input('\n   select model: ')
print(' ')
if not modelinput:
    print('      default used')
    modelinput = 'cylinder'
name = 'none'
try: 
    name,model,p_name = models[int(modelinput)]
except:
    for i in range(len(models)):
        name_i,model_i,p_name_i = models[i]
        if remove(modelinput) == name_i:  
            name,model,p_name = models[i]
    if name == 'none':
        print('      ERROR: unknown model:  %s' % modelinput)
        print('      check input or add model to bayesfit.py')
        exit(-1)
print('      selected model: %s\n' % name)

## PRIOR
print('-- INPUT: PRIOR --')
print('   type prior value(s) and press enter')
print('   mean, sigma (optional), min (optional), max (optional): ')  
K = len(p_name)
p0,dp0,lb,ub = np.zeros(K),np.zeros(K),np.zeros(K),np.zeros(K)
for i in range(K):
    CONTINUE = True
    while CONTINUE:
        CONTINUE = False
        tmp1 = input('   %-s : ' % p_name[i])
        print(' ')
        tmp = remove_comma(tmp1)
        if not tmp:
            if i == 0:
                tmp = '30 10 0'
                print('      default used')
            if i == 1:
                tmp = '100 20 0 180'
                print('      default used')
            if i == 2:
                tmp = '1.0 100.0'
                print('      default used')
            if i == 3:
                tmp = '0.0 10.0'
                print('      default used')
        try:
            n_input = len(np.genfromtxt(StringIO(tmp),unpack=True))
            if n_input == 2:
                p0[i],dp0[i]= np.genfromtxt(StringIO(tmp),unpack=True)
                lb[i],ub[i] = p0[i]-5*dp0[i],p0[i]+5*dp0[i]
            elif n_input == 3:
                p0[i],dp0[i],lb_tmp = np.genfromtxt(StringIO(tmp),unpack=True)
                lb[i] = np.max([lb_tmp,p0[i]-5*dp0[i]])
                ub[i] = p0[i]+5*dp0[i]
            elif n_input == 4:
                p0[i],dp0[i],lb_tmp,ub_tmp = np.genfromtxt(StringIO(tmp),unpack=True)
                lb[i] = np.max([lb_tmp,p0[i]-5*dp0[i]])
                ub[i] = np.min([ub_tmp,p0[i]+5*dp0[i]])
            else:
                print('\n\n!!')
                print('      you should type between 1 and 4 prior values: mean, sigma, min, max')
                print('      try again:')
                print('\n\n!!')
                CONTINUE=True
            if lb[i] > p0[i]:
                print('\n\n!!')
                print('      lower bound (%f) should be smaller than the mean (%f)' % (lb[i],p0[i]))
                print('      try again:')
                print('\n\n!!')
                CONTINUE=True
            if ub[i] < p0[i]:
                print('\n\n!!')
                print('      upper bound (%f) should be larger than the mean (%f)' % (ub[i],p0[i]))
                print('      try again:')
                print('\n\n!!')
                CONTINUE=True
        except:
            p0[i] = np.genfromtxt(StringIO(tmp),unpack=True)
            dp0[i] = 1e6
            lb[i],ub[i] = p0[i]-5*dp0[i],p0[i]+5*dp0[i]
    print('      %s = %f +/- %f (min %f, max %f)' % (p_name[i],p0[i],dp0[i],lb[i],ub[i]))
print(' ')

## ALPHA SCAN
print('-- INPUT: LOGALPHA SCAN --')
tmp = input('   provide min max nsteps :')
if not tmp:
    tmp = '-5 5 15'
    print('\n      default used')
CALC=False
try:
    logalpha_min,logalpha_max,logalpha_n_tmp = np.genfromtxt(StringIO(tmp),unpack=True)
    logalpha_n = int(logalpha_n_tmp)
    logalpha_scan = np.linspace(logalpha_min,logalpha_max,logalpha_n)
    print('\n      selected logalpha scan: from %1.2f to %1.2f (n = %d)\n' % (logalpha_min,logalpha_max,logalpha_n))
except:
    FIT = np.genfromtxt(StringIO(tmp),unpack=True)
    if FIT:
        print('\n      fit with alpha = 1\n')
        logalpha_scan = [0]
    else:
        CALC=True
        print('\n      calculate with default values, and alpha=1, no fitting\n')
        logalpha_scan = [0]
    logalpha_n = 1 

## PLOT DATA?
print('-- INPUT: PLOT DATA? --')
PLOT_IN = input('   (y)es or (n)o : ')
PLOT = check_input(PLOT_IN,1)

##  PLOT PRIOR AND POSTERIOR DISTRIBUTIONS?
print('-- INPUT: PLOT PLOT PRIOR and POSTERIOR? --')
PLOT_POST_IN = input('   (y)es or (n)o : ')
PLOT_POST = check_input(PLOT_POST_IN,0)

## SET WEIGHT IN MINIMIZATION
if Ncontrasts > 1:
    print('-- INPUT: WEIGHT SCHEME --')
    print('   available weight schemes:')
    print('   0: chi-square (not reduced chi-square) - default weight scheme')
    print('   1: chi-square/M (M: number of points in dataset)')
    print('   2: Ng*chi-square/M (Ng: number of good parameters, as provided by user)')
    print('   10: only consider the 1st dataset')
    print('   11: only consider the 2nd dataset')
    print('   12: only consider the 3rd dataset')
    print('   13: only consider the 4th dataset')
    print('   14: only consider the 5th dataset')
    WEIGHT_IN = input('   select weight scheme : ')
    if not WEIGHT_IN:
        WEIGHT = 0
        print('\n      default weight scheme used (0: chi-square)')
    else:
        WEIGHT = int(WEIGHT_IN)
    weights = np.ones(Ncontrasts) 
    tiny = 1e-10 # not zero to avoid division by zero
    string_weight = '\n      selected weight scheme:'
    if WEIGHT == 0:
        print('%s %d, chi-square' % (string_weight,WEIGHT))
    elif WEIGHT == 1:
        print('%s %d, chi2-square/M' % (string_weight,WEIGHT))
        weights *= np.array(M)**-0.5
    elif WEIGHT == 2:
        print('%s %d: sum(Ng*chi-square/M)' % (string_weight,WEIGHT))
        Ng_BIFT = np.ones(Ncontrasts)
        for ii in range(Ncontrasts):
            Ng_ii = input()
            Ng_BIFT[ii] = float(Ng_ii)
            print('      Ng of dataset %d: %s' % (ii,Ng_ii))
        weights *= (Ng_BIFT/np.array(M))**0.5
    elif WEIGHT == 3:
        print('%s %d, first dataset not considered' % (string_weight,WEIGHT))
        weights[0] = tiny
    elif WEIGHT == 4:
        print('%s %d, second dataset not considered' % (string_weight,WEIGHT))
        weights[1] = tiny
    elif WEIGHT == 10:
        print('%s %d, only first dataset considered' % (string_weight,WEIGHT))
        weights *= tiny
        weights[0] = 1
    elif WEIGHT == 11:
        print('%s %d, only second dataset considered' % (string_weight,WEIGHT))
        weights *= tiny
        weights[1] = 1
    elif WEIGHT == 12:
        print('%s %d, only third dataset considered' % (string_weight,WEIGHT))
        weights *= tiny
        weights[2] = 1
    elif WEIGHT == 13:
        print('%s %d, only forth dataset considered' % (string_weight,WEIGHT))
        weights *= tiny
        weights[3] = 1
    elif WEIGHT == 14:
        print('%s %d, only fifth dataset considered' % (string_weight,WEIGHT))
        weights *= tiny
        weights[4] = 1
    else:
        print('%s %d' % (string_weight,WEIGHT))
        print('      ERROR: selected weight scheme must be between 0 and 4 or between 10 and 14')
        exit(-1)
    print(' ')

    ## merge q, I and dI from all datasets using weights
    q_merge,I_merge,dI_merge = [],[],[]
    for ii in range(Ncontrasts):
        q_merge = np.concatenate((q_merge,q[ii]),axis=None)
        I_merge = np.concatenate((I_merge,I[ii]),axis=None)
        dI_merge = np.concatenate((dI_merge,dI[ii]/weights[ii]),axis=None)
else:
    q_merge,I_merge,dI_merge = q[0],I[0],dI[0] 

##############################
### FINISHED READING INPUT ###
##############################

## timing
start_time = time.time()

## make summary file
with open('%s/summary_%s.dat' % (output_dir,output_dir),'w') as f:
    f.write('alpha chi2 chi2r S logT P Ng\n')

## make lists
fit_list,p_list,dp_list,Ng_ii_list,Ng_X_ii_list = [],[],[],[],[]
if Ncontrasts == 2:
    chi2r_ii_matrix = [[],[]]
elif Ncontrasts == 3:
    chi2r_ii_matrix = [[],[],[]]
elif Ncontrasts == 4:
    chi2r_ii_matrix = [[],[],[],[]]
elif Ncontrasts == 5:
    chi2r_ii_matrix = [[],[],[],[]]
elif Ncontrasts > 5:
    print('   ERROR: bayesfit cannot (yet) handle more than 5 datasets simultaneously - please contact the developers') 
    exit(-1)

G,Ng,chi2r = np.zeros(logalpha_n),np.zeros(logalpha_n),np.zeros(logalpha_n)
print('-- OUTPUT: PROGRESS LOGALPHA SCAN --')
for ia in range(logalpha_n):
    alpha = 10**logalpha_scan[ia]
    a = np.sqrt(alpha)
    func = convert_function(a,K,M,model)
    
    # merge q_merge,I_merge and dI_merge with p,dp
    q_dummy = np.ones(K)*99
    x = np.concatenate((q_merge,q_dummy),axis=None)
    y = np.concatenate((I_merge,a*p0),axis=None)
    dy = np.concatenate((dI_merge,dp0),axis=None)
    
    # fit
    if CALC:
        popt = p0
        dpopt = dp0
    else:
        popt,pcov = curve_fit(func,x,y,sigma=dy,absolute_sigma=True,p0=p0,bounds=(lb,ub),ftol=1e-3,xtol=1e-3)
        dpopt = np.sqrt(np.diag(pcov)) 
    fit = get_fit(q,model,popt)

    # calc chi2 and chi2r
    chi2 = 0
    for ii in range(Ncontrasts):
        R = (I[ii]-fit[ii])/dI[ii]
        chi2_ii = np.sum(R**2)
        chi2r_ii = chi2_ii/(M[ii]-K)
        if Ncontrasts > 1:
            chi2r_ii_matrix[ii].append(chi2r_ii)
        chi2 += chi2_ii 
    M_merge = len(q_merge) 
    chi2r[ia] = chi2/(M_merge-K)

    # calc S
    R_S = (p0-popt)/dp0
    S = np.sum(R_S**2)

    # calc Q 
    Q = chi2 + alpha*S
    
    # estimate matrix B = nabla nabla chi2
    # BB is the unitless version of B, also without factor 2
    def get_BB(K,dI_merge,Ncontrasts,dI,popt,q,model,fit,dp0):
        BB = np.zeros((K,K))
        dI2 = dI_merge**2.0
        BB_ii,dI2_ii = [],[]
        for ii in range(Ncontrasts):
            BB_ii.append(np.zeros((K,K)))
            dI2_ii.append(dI[ii]**2.0)
        eps = 0.001
        for i in range(K):
            di = popt[i]*eps
            popt[i] += di
            fit_plus = get_fit(q,model,popt)
            popt[i] -= di
            dIdi = (fit_merge(fit) - fit_merge(fit_plus))/di
            dIdi_ii = []
            for ii in range(Ncontrasts):
                dIdi_ii.append((fit[ii] - fit_plus[ii])/di)
            for j in range(K):
                dj = popt[j]*eps
                popt[j] += dj
                fit_plus = get_fit(q,model,popt)
                popt[j] -= dj
                dIdj = (fit_merge(fit) - fit_merge(fit_plus))/dj
                dI2dr = 2*np.sum(dIdi*dIdj/dI2)
                dp2 = dp0[i]*dp0[j]
                BB[i,j] = dI2dr*dp2/2.0
                for ii in range(Ncontrasts):
                    dIdj_ii = (fit[ii] - fit_plus[ii])/dj
                    dI2dr_ii = 2*np.sum(dIdi_ii[ii]*dIdj_ii/dI2_ii[ii])
                    BB_ii[ii][i,j] = dI2dr_ii*dp2/2.0
        return BB,BB_ii
    BB,BB_ii = get_BB(K,dI_merge,Ncontrasts,dI,popt,q,model,fit,dp0)

    # calc number of good parameters
    etaBB = np.linalg.eigh(BB)[0]
    Ng[ia] = np.sum(etaBB/(alpha+etaBB))
    
    # calc number of good parameters for each dataset
    Ng_ii = np.zeros(Ncontrasts)
    for ii in range(Ncontrasts):
        #etaBB_ii = np.linalg.eig(BB_ii[ii])[0]
        etaBB_ii = np.linalg.eigh(BB_ii[ii])[0]
        Ng_ii[ii] = np.sum(etaBB_ii/(alpha+etaBB_ii))
    Ng_sum = np.sum(Ng_ii)
    fraction = Ng_ii[ii]/Ng_sum
    if Ncontrasts > 1:
        wx = (1-fraction)/(Ncontrasts-1)
    else:
        wx = 1
    Ng_X_ii = Ng_ii - wx*(Ng_sum-Ng[ia])

    # matrix C = nabla nabla Q
    # CC is the unitless version of C, also without factor 2
    CC = BB
    for ii in range(K):
        CC[ii,ii] += alpha
    
    # calc detC
    detCC = np.linalg.det(CC)

    # calc logT
    # A = nabla nabla alpha*S
    # AA is the unitless version of A, also without factor 2 
    detAA = alpha**K
    if detCC <= 0:
        # for some reason, detCC is sometimes negative - numerical instability??
        print('detCC (in bayesapp.py) is negative: %e' % detCC)
        print('using logT = log(detAA) instead of logT = log(detAA/detCC)')
        logT = -np.log(detAA)
    else:
        logT = np.log(detCC)-np.log(detAA)

    # Jeffreys prior
    jef = 2*np.log(alpha)

    # score function G (denoted "evidence" in Hansen2000 and Larsen2018)
    G[ia] = Q + logT + jef
    #aS = alpha*S

    # add to lists
    p_list.append(popt)
    dp_list.append(dpopt)
    Ng_ii_list.append(Ng_ii)
    Ng_X_ii_list.append(Ng_X_ii)
    fit_list.append(fit)
    
    # write summary file 
    with open('%s/summary_%s.dat' % (output_dir,output_dir),'a') as f:
        f.write('%e %f %f %f %f %f\n' % (alpha,chi2,chi2r[ia],S,logT,Ng[ia])) 
   
    # output status 
    print('   logalpha = %1.4f, score function G (should be as small as possible) = %1.4f'% (logalpha_scan[ia],G[ia]))
print(' ')

## CALCULATE WEIGHTED AVERAGES
G_min = np.amin(G)
delta_G = G - G_min
Pnorm = np.exp(-delta_G/2)
sumP = np.sum(Pnorm) # probability used in weighted averages 
fit_av = []
for i in range(Ncontrasts):
    fit_tmp = [sublist[i] for sublist in fit_list]
    fit_av.append(av(fit_tmp,Pnorm))
Ng_av = av(Ng,Pnorm)
Ng_ii_av,Ng_X_ii_av = [],[]
for ii in range(Ncontrasts):
    tmp_ii = [item[ii] for item in Ng_ii_list]
    tmp_X_ii = [item[ii] for item in Ng_X_ii_list]
    Ng_ii_av.append(av(tmp_ii,Pnorm))
    Ng_X_ii_av.append(av(tmp_X_ii,Pnorm))
logalpha_av = av(logalpha_scan,Pnorm)
p = av(p_list,Pnorm)
dp = av(dp_list,Pnorm)
chi2r_av = av(chi2r,Pnorm)
if Ncontrasts > 1:
    chi2r_ii_av = []
for ii in range(Ncontrasts):
    if Ncontrasts > 1:
        chi2r_ii_av.append(av(chi2r_ii_matrix[ii],Pnorm))

## OUTPUT: RESULTS TO FILE
Iprior = get_fit(q,model,p0)
for ii in range(Ncontrasts):
    with open('%s/fit_%s_dataset%d.dat' % (output_dir,output_dir,ii),'w') as f:
        f.write('# q I dI Iprior Ifit\n')
        for j in range(M[ii]):
            f.write('%f %f %f %f %f\n' % (q[ii][j],I[ii][j],dI[ii][j],Iprior[ii][j],fit_av[ii][j]))

## OUTPUT: GOODNESS OF FIT
print('-- OUTPUT: GOODNESS OF FIT --')
chi2r_Ng = chi2r_av*(M_merge-K)/(M_merge-Ng_av)
chi2r_M  = chi2r_av*(M_merge-K)/M_merge
#print('   chi2r_K = %1.6f' % chi2r_av)
#print('   chi2r_Ng = %1.6f' % chi2r_Ng)     
#print('   chi2r_M = %1.6f' % chi2r_M) 
print('   reduced chi-square (total) = %1.6f' % chi2r_Ng)     
if Ncontrasts > 1:
    for ii in range(Ncontrasts):
        chi2 = chi2r_ii_av[ii]*(M[ii]-K)
        chi2r_ii_av_K = chi2/(M[ii]-K)
        chi2r_ii_av_Ng_tot = chi2/(M[ii]-Ng_av)
        chi2r_ii_av_M = chi2/M[ii]
        chi2r_ii_av_Ng_ii = chi2/(M[ii]-Ng_ii_av[ii])
        chi2r_ii_av_Ng_X_ii = chi2/(M[ii]-Ng_X_ii_av[ii])
        print('   reduced chi-square for dataset %d = %1.6f' % (ii,chi2r_ii_av_Ng_X_ii))
print(' ')

## OUTPUT: INFORMATION CONTENT (Number of good parameters)
print('-- OUTPUT: INFORMATION GAIN --')
print('   logalpha of minimum = %1.6f' % logalpha_av)
print('   Number of good parameters (total) = %1.6f out of %d' % (Ng_av,K))
for ii in range(Ncontrasts):
    print('   Number of good parameters for dataset %d = %1.6f' % (ii,Ng_X_ii_av[ii]))
print(' ')

## OUTPUT: TIMING
end_time = time.time() - start_time
print('-- OUTPUT: TIMING --')
print('   run time: %1.1f sec' % end_time)
print(' ')

## POSTERIOR
print('-- OUTPUT: POSTERIOR --')

for i in range(K):
    # Kullbeck-Leibler divergence
    kld = kl_divergence0(p[i],dp[i],p0[i],dp0[i])
    x = np.linspace(p0[i]-5*dp0[i],p0[i]+5*dp0[i],10000)
    posterior_pdf = norm.pdf(x,p[i],dp[i])
    prior_pdf = norm.pdf(x,p0[i],dp0[i])
    prior_pdf_uniform = np.ones(len(x))/len(x)
    kld = kl_divergence(posterior_pdf,prior_pdf)
    kld_uniform = kl_divergence(posterior_pdf,prior_pdf_uniform)
    print('   %s = %f +/- %f' % (p_name[i],p[i],dp[i]))

    # plot prior and posterior distributions
    if PLOT_POST:         
        plt.plot(x,prior_pdf/np.amax(prior_pdf),linestyle='--',color='grey',label='prior')
        plt.plot(x,posterior_pdf/np.amax(posterior_pdf),color='black',label='posterior, kld = %1.1f' % kld)
        plt.title(p_name[i])
        plt.legend(frameon=False)
        plt.show()  
print(' ')

## PLOT PROBABILITY
PLOT_P = 1
if PLOT_P and len(logalpha_scan) > 1:
    print('-- OUTPUT: PLOT PROBABILITY --')
    print('   done')
    plt.plot(logalpha_scan,Pnorm,color='black')
    plt.plot(logalpha_scan,logalpha_scan*0,linestyle='--',color='grey')
    plt.xlabel('log(alpha)')
    plt.ylabel('Probability = exp(-G/2)')
    plt.show()
    print(' ')

## PLOT DATA
if PLOT:
    print('-- OUTPUT: PLOT DATA AND FIT --')
    print('   done')    
    color = ['red','blue','green','orange']
    offset = [1,1e2,1e4,1e6]
    fig,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
    Rmax = 0
    for i in range(Ncontrasts):
        R = (I[i]-fit_av[i])/dI[i]
        Rprior = (I[i]-Iprior[i])/dI[i]
        chi2_prior = np.sum(Rprior**2)
        chi2r_prior = chi2_prior/len(I[i])
        p0.errorbar(q[i],offset[i]*I[i],yerr=offset[i]*dI[i],linestyle='none',marker='.',color=color[i],label=dataname[i],zorder=100+i)
        if i == 0:
            p0.plot(q[i],offset[i]*Iprior[i],linestyle='--',color='grey',label=r'prior $\chi^2_r = %1.2f$' % chi2r_prior,zorder=300+i)
            p0.plot(q[i],offset[i]*fit_av[i],color='black',label=r'fit $\chi^2_r = %1.2f$' % chi2r_Ng,zorder=200+i)
        else: 
            p0.plot(q[i],offset[i]*Iprior[i],linestyle='--',color='grey',zorder=300+i)
            p0.plot(q[i],offset[i]*fit_av[i],color='black',zorder=200+i)
        p1.plot(q[i],R,linestyle='none',marker='.',color=color[i])
        Rmax_i = np.amax(abs(R))
        if Rmax_i > Rmax:
            Rmax = Rmax_i
    Rmax = np.ceil(Rmax)
    xlim = p0.get_xlim()
    p1.plot(xlim,np.zeros(2),color='black')
    if Rmax > 3:
        p1.set_ylim(-Rmax,Rmax)
        if Rmax < 10:
            p1.set_yticks([-Rmax,-3,0,3,Rmax])
            p1.plot(xlim,np.ones(2)*-3,color='black',linestyle='--')
            p1.plot(xlim,np.ones(2)*3,color='black',linestyle='--')
        else:
            p1.set_yticks([-Rmax,0,Rmax])
    if LOG:
        p0.set_xscale('log')
        p1.set_xscale('log')    
    p0.set_yscale('log')
    p1.set_xlabel(r'$q$ [$\AA^{-1}$]')
    p0.set_ylabel(r'$I(q)$')
    p1.set_ylabel(r'$\Delta I/\sigma$')
    p0.legend(frameon=False)
    plt.savefig('fit.pdf')
    plt.show()
print(' ')
    
## clean up
rmtree('__pycache__')

