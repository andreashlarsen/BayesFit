"""
This script reformulates a function so it can be used by curve_fit
"""

import numpy as np

def calc_y(x,a,M,model,p):
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

def function(a,K,M,model):
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
        print('ERROR: bayesfit cannot (yet) handle models with more than 20 parameters - please contact the developers') 
        exit(-1)  
          
    return func
