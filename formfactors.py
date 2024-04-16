import numpy as np
from scipy.special import jv
from numpy import exp, sin, cos, sqrt
import time

####################################################################
# basic functions 
####################################################################

def bessj0(x):
    """
    Bessel function of the first kind of zeroth order
    """
    return jv(0,x)

def bessj1(x):
    """"
    Bessel function of the first kind of first order
    """
    return jv(1,x)

def bessj1c(x):
    """
    bessj1(x)/x
    take care of zeros
    """
    try:
        y = np.ones(len(x))*0.5
        idx = np.where(x != 0)
        y[idx] = bessj1(x[idx])/x[idx]
    except:
        y = bessj1(x)/x
    return y

def sinc(x):
    """
    function for calculating sinc = sin(x)/x
    numpy.sinc is defined as sinc(x) = sin(pi*x)/(pi*x)
    """
    return np.sinc(x/np.pi)

pi = np.pi

####################################################################
# sphere
####################################################################

def psi_sphere(q,r):
    """
    Form factor amplitude of a sphere
    """
    x = q*r     
    return 3*(np.sin(x)-x*np.cos(x))/x**3
    
def P_sphere(q,r):
    """
    Form factor of a sphere
    """
    return psi_sphere(q,r)**2   
   
def V_sphere(r):
    """
    Volume of a sphere
    """
    return 4*r**3*np.pi/3

def sphere2(q,r,s):
    """
    Model: Sphere - with scaling
    """
    return s*P_sphere(q,r)  

def sphere3(q,r,s,b):
    """
    Model: Sphere - with scaling and background
    """
    return s*P_sphere(q,r)+b


####################################################################
# cylinder
####################################################################

def V_cyl(R,L):
    """
    Volume of a cylinder with radius R and length L
    """
    return np.pi*R**2*L

def psi_cyl(q,R,L,a):
    """ 
    form factor amplitude: cylinder with radius R and length L
    depends on view angle a (for alpha)
    reference: Pedersen1997
    """
    x=q*R*sin(a)
    z=q*L*cos(a)/2.0
    psi=2.0*bessj1c(x)*sinc(z)
    return psi

def P_cyl(q,R,L):
    """
    form factor: cylinder with radius R and length L
    """
    N_alpha = 30
    alpha = np.linspace(0,np.pi/2,N_alpha)
    P_sum = 0
    for a in alpha:
        P_sum += psi_cyl(q,R,L,a)**2*sin(a)
    P = P_sum/P_sum[0]

    return P  

def cylinder(q,R,L,s,b):
    """
    Model: cylinder with radius R and lenght L
    scale and background
    """
    return s*P_cyl(q,R,L)+b

####################################################################
# stacked cylinder (cylinders stacked along length axis)
####################################################################

def psi_stacked_2cyl(q,R1,R2,L1,L2,p1,p2,a):
    """
    Form factor amplitude: 2 stacked cylinders, stacked along the length axis
    Ri,Li,pi: radius and length and contrast of ith cylinder
    """

    # distance com to com
    d = L1/2.0 + L2/2.0
    qr = q*d*cos(a)

    # phase factors
    pf1 = 1.0
    pf2 = np.exp(1j*qr)
    
    # volumes
    V1,V2 = V_cyl(R1,L1),V_cyl(R2,L2)
    
    # form factor amplitudes
    psi1,psi2 = psi_cyl(q,R1,L1,a),psi_cyl(q,R2,L2,a) 

    # collective form factor amplitude
    pV1,pV2 = p1*V1,p2*V2 
    psi_dim = pV1*psi1*pf1 + pV2*psi2*pf2

    return psi_dim/(pV1+pV2)

def P_stacked_2cyl(q,R1,R2,L1,L2,p1,p2):
    """ 
    Form factor: 2 stacked cylinders, stacked along the length axis
    R1,L1: radius and length of first cyl
    R2,L2: radius and lenght of second cyl
    """

    N_alpha,alpha_max = 50,np.pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)

    # this is slower than a simple for loop 
    #t0 = time.time()
    #a = alpha.reshape(-1,1)
    #P_a = np.sum(abs(psi_stacked_2cyl(q,R1,R2,L1,L2,p1,p2,a))**2*sin(a),axis=0)
    #t = time.time()-t0
    #print(t)
    #t0 = time.time()
    P_a = np.zeros(len(q))
    for a in alpha:
        P_a += abs(psi_stacked_2cyl(q,R1,R2,L1,L2,p1,p2,a))**2 * sin(a)
    #t = time.time()-t0
    #print(t)

    # normalization
    P = P_a/P_a[0]

    return P

def stacked_2cyl_sameR_ratio(q,R,L1,L2,p21,s,b):
    """
    Model: stacked cylinders (2 cylinders), stacked along the lenght direction
    they have the same radius: R
    L1,L2: length of cylinder 1 and 2
    p21 = p2/p1 (ratio of excess scattering length densities) 
    scaled and background subtracted
    """
    return s*P_stacked_2cyl(q,R,R,L1,L2,1,p21)+b

def psi_stacked_3cyl(q,R1,R2,R3,L1,L2,L3,p1,p2,p3,a):
    """
    Form factor amplitude: 3 stacked cylinders, stacked along the length axis
    Ri,Li,pi: radius and length and contrast of ith cylinder
    """

    # distance com to com
    d2 = L1/2.0 + L2/2.0
    d3 = L1/2.0 + L2 + L3/2.0
    qr2 = q*d2*cos(a)
    qr3 = q*d3*cos(a)

    # phase factors
    pf1 = 1.0
    pf2 = np.exp(1j*qr2)
    pf3 = np.exp(1j*qr3)

    # volumes
    V1,V2,V3 = V_cyl(R1,L1),V_cyl(R2,L2),V_cyl(R3,L3)

    # form factor amplitudes
    psi1,psi2,psi3 = psi_cyl(q,R1,L1,a),psi_cyl(q,R2,L2,a),psi_cyl(q,R3,L3,a)

    # collective form factor amplitude
    pV1,pV2,pV3 = p1*V1,p2*V2,p3*V3
    psi_tri = pV1*psi1*pf1 + pV2*psi2*pf2 + pV3*psi3*pf3

    return psi_tri/(pV1+pV2+pV3)

def P_stacked_3cyl(q,R1,R2,R3,L1,L2,L3,p1,p2,p3):
    """ 
    Form factor: 3 stacked cylinders, stacked along the length axis
    Ri,Li: radius and length of ith cyl
    """

    N_alpha,alpha_max = 50,np.pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)

    P_a = np.zeros(len(q))
    for a in alpha:
        P_a += abs(psi_stacked_3cyl(q,R1,R2,R3,L1,L2,L3,p1,p2,p3,a))**2 * sin(a)

    # normalization
    P = P_a/P_a[0]
    
    return P

def stacked_3cyl_sameR_ratio(q,R,L1,L2,L3,p21,p31,s,b):
    """
    Model: stacked cylinders (3 cylinders), stacked along the lenght direction
    they have the same radius: R
    Li: length of cylinder i
    pi1 = pi/p1 (ratio of excess scattering length densities) 
    scaled and background subtracted
    """
    return s*P_stacked_3cyl(q,R,R,R,L1,L2,L3,1,p21,p31)+b

def stacked_3cyl_sameR_ratio_2(qX,qN,R,L1,L2,L3,pX21,pX31,pN21,pN31,sX,bX,sN,bN):
    """
    Model: stacked cylinders (3 cylinders), stacked along the lenght direction
    they have the same radius: R
    Li: length of cylinder i
    pi1 = pi/p1 (ratio of excess scattering length densities) 
    scaled and background subtracted
    fitted to two datasets: X and N
    """
    yX = sX*P_stacked_3cyl(qX,R,R,R,L1,L2,L3,1,pX21,pX31)+bX
    yN = sN*P_stacked_3cyl(qN,R,R,R,L1,L2,L3,1,pN21,pN31)+bN

    return [yX,yN]

def stacked_3cyl_sameR_ratio_smooth(q,R,L1,L2,L3,p21,p31,sigmaR,s,b):
    """
    Model: stacked cylinders (3 cylinders), stacked along the lenght direction
    they have the same radius: R
    Li: length of cylinder i
    pi1 = pi/p1 (ratio of excess scattering length densities) 
    smoothness
    scaled and background subtracteid
    """
    return s*P_stacked_3cyl(q,R,R,R,L1,L2,L3,1,p21,p31)*exp(-q**2*sigmaR**2)+b

####################################################################
# closely packed parallel cylinder 
####################################################################

def psi_cylinder_dimer(q,R,L1,L2,p1,p2,a,b):
    """ 
    Form factor amplitude: cylinder dimer (closely packed parallel cylinders)
    """
    V1,V2 = V_cyl(R,L1),V_cyl(R,L2)
    psi1,psi2 = psi_cyl(q,R,L1,a),psi_cyl(q,R,L2,a)
    
    # phase factors
    d = 2*R
    pf1 = 1.0
    pf2 = np.exp(1j*q*d*sin(a)*cos(b)) 
    
    # form factor amplitude
    pV1,pV2 = p1*V1,p2*V2
    psi_cyl_dim = p1*psi1*pf1 + pV2*psi2*pf2 
    
    return psi_cyl_dim/(pV1 + pV2)

def psi_cylinder_trimer(q,R,L1,L2,L3,p1,p2,p3,a,b):
    """ 
    Form factor amplitude: cylinder trimer (closely packed parallel cylinders)
    """
    V1,V2,V3 = V_cyl(R,L1),V_cyl(R,L2),V_cyl(R,L3)

    psi1,psi2,psi3 = psi_cyl(q,R,L1,a),psi_cyl(q,R,L2,a),psi_cyl(q,R,L3,a)

    # phase factors
    d = 2*R
    pf1 = 1.0
    pf2 = np.exp(1j*q*d*sin(a)*cos(b)) 
    pos3 = cos(b)*cos(np.pi/3.0)+sin(b)*sin(np.pi/3.0)
    pf3 = np.exp(1j*q*d*sin(a)*pos3)

    # form factor amplitude
    pV1,pV2,pV3 = p1*V1,p2*V2,p3*V3
    psi_cyl_tri = pV1*psi1*pf1 + pV2*psi2*pf2 + pV3*psi3*pf3

    return psi_cyl_tri/(pV1 + pV2 + pV3)

def P_cylinder_dimer(q,R,L1,L2,p1,p2):
    """
    Form factor: cylinder dimer
    """
    
    N_alpha,alpha_max = 100,np.pi/2
    N_beta,beta_max = 100,np.pi
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)
    beta = np.linspace(beta_max/N_beta,beta_max,N_beta)

    # inner loop vectorized
    b = beta.reshape(-1,1)
    P_a = np.zeros(len(q))
    for a in alpha:
        P_b = np.sum(abs(psi_cylinder_dimer(q,R,L1,L2,p1,p2,a,b))**2,axis=0)
        P_a += P_b*sin(a)
    
    # normalization
    P = P_a/P_a[0] 

    return P

def P_cylinder_trimer(q,R,L1,L2,L3,p1,p2,p3):
    """
    Form factor: cylinder trimer
    """
    
    N_alpha,alpha_max = 50,np.pi
    N_beta,beta_max = 50,2*np.pi
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)
    beta = np.linspace(beta_max/N_beta,beta_max,N_beta)
    M = len(q)

    b = beta.reshape(-1,1)
    P_a = np.zeros(M)
    for a in alpha:
        P_b = np.sum(abs(psi_cylinder_trimer(q,R,L1,L2,L3,p1,p2,p3,a,b))**2,axis=0)
        P_a += P_b*sin(a)

    # normalization
    P = P_a/P_a[0]

    return P

def cylinder_dimer(q,R,L,s,b):
    """
    Model: dimer of cylinders, closely packed in parallel
    same radius: R
    same length: L
    scaled and background subtracted
    """
    return s*P_cylinder_dimer(q,R,L,L,1,1)+b

def cylinder_dimer_ratio(q,R,L,p,s,b):
    """
    Model: dimer of cylinders, closely packed in parallel
    same radius: R
    same length: L
    deltaSLD differ by factor p (ratio) 
    scaled and background subtracted
    """
    return s*P_cylinder_dimer(q,R,L,L,1,p)+b

def cylinder_trimer_ratio(q,R,L,p1,p2,s,b):
    """
    Model: trimer of cylinders, closely packed in parallel
    same radius: R
    same length: L
    deltaSLD differ by factors p1 and p2 (ratios) 
    scaled and background subtracted
    """
    return s*P_cylinder_trimer(q,R,L,L,L,1,p1,p2)+b

def cylinder_trimer_ratio_L(q,R,L1,L2,L3,p1,p2,s,b):
    """
    Model: trimer of cylinders, closely packed in parallel
    same radius: R
    lengths: L1,L2,L3
    deltaSLD differ by factors p1 and p2 (ratios) 
    scaled and background subtracted
    """
    return s*P_cylinder_trimer(q,R,L1,L2,L3,1,p1,p2)+b

####################################################################
# nanodisc
####################################################################

def nanodisc(q,Bg,c,V_l,V_t,CV_p,Nlip,T,sigmaR,Ar,eps,n_w,Rg):
    """
    Model: elliptical nanodisc
    """

    # Volumes (V) and scattering lengths (b)
    # specific for the DLPC/MSP1D1 nanodisc with tags
    V_p = 54293.0
    V_c = 3143.0
    V_s = 30.0

    b_p = 23473.0
    b_h = 164.0723
    b_t = 178.0440
    b_c = 1.4250e+03
    b_s = 10.0

    # constants(avogadros number,electron scattering length)
    N_A = 6.022e23
    b_e = 2.82e-13

    # derived params
    V_h = V_l - V_t
    V_p = CV_p*V_p
    V_c = CV_p*V_c

    # add 7.9 water molecules per lipid headgroup (Kucerka et al., 2005)
    b_h = b_h + n_w * b_s
    V_h = V_h + n_w * V_s

    # reparametrization (from vol to scattering contrasts)
    p_s = b_s/V_s # scattering length density of solvent
    dp_p = b_p/V_p - p_s
    dp_c = b_c/V_c - p_s
    dp_t = b_t/V_t - p_s
    dp_h = b_h/V_h - p_s

    xx=(q*Rg)**2
    P_c=(exp(-xx)+xx-1)/(xx/2.)
    P_tot=0
    F_tot=0

    b=sqrt(abs(Nlip*Ar/(2*pi*eps)))
    a=eps*b
    jmax,kmax=20,20
    dalf=pi/(2.*jmax)
    dfi=pi/(2.*kmax)

    for j in range(1,jmax+1):
        alf=j*dalf
        for k in range(1,kmax+1):
            fi=k*dfi
            r_t=sqrt((a*sin(fi))**2+(b*cos(fi))**2)
            R=sqrt( ((a+T)*sin(fi))**2 +((b+T)*cos(fi))**2)
            h_p=V_p/(pi*((a+T)*(b+T)-a*b))
            h_t=2.0*V_t/Ar
            h_h=V_h/Ar
            H=h_t+2.0*h_h

            Reff=R+abs(Rg)
            yy=q*Reff*sin(alf)
            ya=q*h_p*cos(alf)/2.

            psi_cc=(1-exp(-xx))/xx*bessj0(yy)*sin(ya)/ya

            tail=psi_cyl(q,r_t,h_t,alf)

            pro=V_cyl(R,h_p)*psi_cyl(q,R,h_p,alf)-V_cyl(r_t,h_p)*psi_cyl(q,r_t,h_p,alf)
            pro=pro/(V_cyl(R,h_p)-V_cyl(r_t,h_p))

            head=(H*psi_cyl(q,r_t,H,alf)-h_t*psi_cyl(q,r_t,h_t,alf))/(2*h_h)

            V_nd=Nlip*(V_t+V_h)+V_p
            dp_nd=(dp_t*Nlip*V_t+dp_h*Nlip*V_h+dp_p*V_p)/V_nd

            psi_nd=dp_t*(Nlip*V_t)*tail+dp_h*(Nlip*V_h)*head+dp_p*V_p*pro

            psi_nd=psi_nd/(dp_nd*V_nd)

            S_nd=psi_nd**2
            S_nd_c=psi_cc*psi_nd
            S_cc=psi_cc**2


            F=(dp_nd*V_nd)**2*S_nd+4*dp_c*V_c*dp_nd*V_nd*S_nd_c+2*(dp_c*V_c)**2*(S_cc+P_c)

            F_tot=F_tot+F*sin(alf)

    F_tot=F_tot*dalf*dfi*(2./pi)

    V_tot=V_nd+2*V_c
    dp_tot=(V_nd*dp_nd+2*V_c*dp_c)/V_tot
    P_tot=F_tot/(dp_tot*V_tot)**2
    y=c*1.e-9*N_A*F_tot*exp(-q**2*sigmaR**2)*b_e**2+Bg

    return y

####################################################################
# spherical core-shell particle
####################################################################

def psi_coreshell2(q,r1,r2,p1,p2):
    """
    Form factor amplitude: spherical core-shell particle shell
    """
    V1,V2 = V_sphere(r1),V_sphere(r2)
    s1,s2 = psi_sphere(q,r1),psi_sphere(q,r2)
    Vs1 = V1*s1
    Vs2 = V2*s2

    A = p1*Vs1 + p2*(Vs2-Vs1)
    B = p1*V1  + p2*(V2 -V1)

    return A/B

def P_coreshell2(q,r1,r2,p1,p2):
    """
    Form factor: spherical core-shell particle
    """
    return psi_coreshell2(q,r1,r2,p1,p2)**2

def coreshell2(q,r1,r2,p1,p2,s,b):
    """
    Model: Spherical core-shell particle
           scale and background fitted
    """
    return s * P_coreshell2(q,r1,r2,p1,p2) + b

def coreshell2_ratio(q,R1,R2,r2,s,b):
    """
    Model: spherical core-shell particle
           contrasts constraint so only 1 of 2 are fitted
           scale and background fitted
    """
    return s * P_coreshell2(q,R1,R2,1,r2) + b

####################################################################
# spherical core-multishell particles (1 core, 2 shells)
####################################################################

def psi_coreshell3(q,r1,r2,r3,p1,p2,p3):
    """
    Form factor amplitude: spherical core-multishell particle with 2 shells
    """
    V1,V2,V3 = V_sphere(r1),V_sphere(r2),V_sphere(r3)
    s1,s2,s3 = psi_sphere(q,r1),psi_sphere(q,r2),psi_sphere(q,r3)
    Vs1 = V1*s1
    Vs2 = V2*s2
    Vs3 = V3*s3

    A = p1*Vs1 + p2*(Vs2-Vs1) + p3*(Vs3-Vs2)
    B = p1*V1  + p2*(V2 -V1)  + p3*(V3 -V2) 

    return A/B

def P_coreshell3(q,r1,r2,r3,p1,p2,p3):
    """
    Form factor: spherical core-multishell particle with 2 shells
    """
    return psi_coreshell3(q,r1,r2,r3,p1,p2,p3)**2

def coreshell3(q,r1,r2,r3,p1,p2,p3,s,b):
    """
    Model: spherical core-multishell particle with 2 shells 
           scale and background fitted
    """
    return s * P_coreshell3(q,r1,r2,r3,p1,p2,p3) + b

def coreshell3_2(q1,q2,r1,r2,r3,p11,p12,p13,p21,p22,p23,s1,b1,s2,b2):
    """
    Model: spherical core-multishell particle with 2 shells 
           2 datasets
           scale and background fitted for each dataset
    """
    y1 = coreshell3(q1,r1,r2,r3,p11,p12,p13,s1,b1)
    y2 = coreshell3(q2,r1,r2,r3,p21,p22,p23,s2,b2)
    return [y1,y2]

def coreshell3_ratio(q,R1,R2,R3,r2,r3,s,b):
    """
    Model: spherical core-multishell particle with 2 shells
           contrasts constraint so only 2 of 3 are fitted
           scale and background fitted
    """
    return s * P_coreshell3(q,R1,R2,R3,1,r2,r3) + b


####################################################################
# spherical core-multishell particles (1 core, 3 shells)
####################################################################

def psi_coreshell4(q,r1,r2,r3,r4,p1,p2,p3,p4):
    """
    Form factor amplitude: spherical core-multishell particle with 3 shells
    """
    V1,V2,V3,V4 = V_sphere(r1),V_sphere(r2),V_sphere(r3),V_sphere(r4)
    s1,s2,s3,s4 = psi_sphere(q,r1),psi_sphere(q,r2),psi_sphere(q,r3),psi_sphere(q,r4)
    Vs1 = V1*s1
    Vs2 = V2*s2
    Vs3 = V3*s3
    Vs4 = V4*s4
    A = p1*Vs1 + p2*(Vs2-Vs1) + p3*(Vs3-Vs2) + p4*(Vs4-Vs3)
    B = p1*V1  + p2*(V2-V1)   + p3*(V3-V2)   + p4*(V4-V3) 
    return A/B

def P_coreshell4(q,r1,r2,r3,r4,p1,p2,p3,p4):
    """
    Form factor: spherical core-multishell particle with 3 shells
    """
    return psi_coreshell4(q,r1,r2,r3,r4,p1,p2,p3,p4)**2

def coreshell4(q,r1,r2,r3,r4,p1,p2,p3,p4,s,b):
    """
    Model: spherical core-multishell particle with 3 shells
           scale and background fitted
    """
    return s * P_coreshell4(q,r1,r2,r3,r4,p1,p2,p3,p4) + b

def coreshell4_2(q1,q2,r1,r2,r3,r4,p11,p12,p13,p14,p21,p22,p23,p24,s1,b1,s2,b2):
    """
    Model: spherical core-multishell particle with 3 shells
           scale and background fitted
           2 datasets
    """
    y1 = coreshell4(q1,r1,r2,r3,r4,p11,p12,p13,p14,s1,b1)
    y2 = coreshell4(q2,r1,r2,r3,r4,p21,p22,p23,p24,s2,b2)
    return [y1,y2]

def coreshell4_ratio(q,R1,R2,R3,R4,r2,r3,r4,s,b):
    """
    Model: spherical core-multishell particle with 3 shells
           contrasts constraint so only 3 of 4 are fitted
           scale and background fitted
    """
    return s * P_coreshell4(q,R1,R2,R3,R4,1,r2,r3,r4) + b

def coreshell4_ratio_2(q1,q2,R1,R2,R3,R4,r12,r13,r14,r22,r23,r24,s1,b1,s2,b2):
    """
    Model: spherical core-multishell particle with 3 shells
           contrasts constraint so only 3 of 4 are fitted
           scale and background fitted for each dataset
           2 datasets
    """
    y1 = s1 * P_coreshell4(q1,R1,R2,R3,R4,1,r12,r13,r14) + b1
    y2 = s2 * P_coreshell4(q2,R1,R2,R3,R4,1,r22,r23,r24) + b2
    return [y1,y2]

def coreshell4_ratio_2_T(q1,q2,R1,T2,T3,T4,r12,r13,r14,r22,r23,r24,s1,b1,s2,b2):
    """
    Model: spherical core-multishell particle with 3 shells
           contrasts constraint so only 3 of 4 are fitted
           scale and background fitted for each dataset
           2 datasets
           reparametrized in terms of core redius and 3 shell thicknesses
    """
    R2 = R1+T2
    R3 = R2+T3
    R4 = R3+T4
    y1 = s1 * P_coreshell4(q1,R1,R2,R3,R4,1,r12,r13,r14) + b1
    y2 = s2 * P_coreshell4(q2,R1,R2,R3,R4,1,r22,r23,r24) + b2
    return [y1,y2]

def coreshell4_ratio_3(q1,q2,q3,R1,R2,R3,R4,r12,r13,r14,r22,r23,r24,s1,b1,s2,b2,s3,b3):
    """
    Model: spherical core-multishell particle wih 2 shells
           scale and background fitted for each dataset
           3 datasets
    """
    y1 = s1 * P_coreshell4(q1,R1,R2,R3,R4,1,r12,r13,r14) + b1
    y2 = s2 * P_coreshell4(q2,R1,R2,R3,R4,1,r22,r23,r24) + b2
    y3 = s3 * P_sphere(q3,R1) + b3
    return [y1,y2,y3]

####################################################################
# spherical cores shell (1 core, 3 shells) with structure factor
####################################################################

def S_HS(q,eta,R):
    """
    calculate the hard-sphere structure factor
    calls function calc_G()
    
    input
    q       : momentum transfer
    eta     : volume fraction
    R       : estimation of the hard-sphere radius
    
    output
    S_HS    : hard-sphere structure factor
    """

    if eta > 0.0:
        A = 2*R*q 
        G = calc_G(A,eta)
        S_HS = 1/(1 + 24*eta*G/A)
    else:
        S_HS = np.ones(len(q))

    return S_HS

def calc_G(A,eta):
    """ 
    calculate G in the hard-sphere potential
    
    input
    A  : 2*R*q
    q  : momentum transfer
    R  : hard-sphere radius
    eta: volume fraction
    
    output:
    G  
    """

    a = (1+2*eta)**2/(1-eta)**4
    b = -6*eta*(1+eta/2)**2/(1-eta)**4 
    c = eta * a/2
    sinA = np.sin(A)
    cosA = np.cos(A)
    fa = sinA-A*cosA
    fb = 2*A*sinA+(2-A**2)*cosA-2
    fc = -A**4*cosA + 4*((3*A**2-6)*cosA+(A**3-6*A)*sinA+6)
    G = a*fa/A**2 + b*fb/A**3 + c*fc/A**5
    
    return G

def coreshell4_ratio_HS_2(q1,q2,R1,R2,R3,R4,r12,r13,r14,r22,r23,r24,s1,b1,s2,b2,eta):
    """
    Model: spherical core-multishell particle with 3 shells
           contrasts constraint so only 3 of 4 are fitted
           Hard-sphere structure factor on each dataset, with R4 (outer radius) as effective HS radius
           scale and background fitted for each dataset
           2 datasets
    """
    y1 = s1 * P_coreshell4(q1,R1,R2,R3,R4,1,r12,r13,r14)*S_HS(q1,eta,R4) + b1
    y2 = s2 * P_coreshell4(q2,R1,R2,R3,R4,1,r22,r23,r24)*S_HS(q2,eta,R4) + b2
    return [y1,y2]
 
####################################################################
# special cases: core-multishell (1 core, 3 shells) with ad hoc addition of resolution effect
####################################################################


def get_dq():
    filename = 'RES_EFF_DATA'
    dq = np.genfromtxt(filename,usecols=[4],unpack=True) 
   
    return dq

def res_eff(q,y):
    """
    Add resolution effect
    """
    dq = get_dq() #np.genfromtxt('data/%s' % filename,usecols=[4],unpack=True)

    y_res = np.zeros(len(y))
    for i in range(len(y)):
        w = np.exp(-0.5*(q-q[i])**2/dq[i]**2)
        y_res[i] = np.sum(w*y)/np.sum(w)
    return y_res

def coreshell4_ratio_2_res(q1,q2,R1,R2,R3,R4,r12,r13,r14,r22,r23,r24,s1,b1,s2,b2):
    """
    Model: spherical core-multishell particle with 3 shells
           contrasts constraint so only 3 of 4 are fitted
           scale and background fitted for each dataset
           2 datasets
           resolution effects added to the dataset named data/Isim2res_f1.0.dat
           note: this is only for the proof-of-principle, used in the paper presenting the method
                 resolution effects should be included in a better and more general way
    """
    y1 = s1 * P_coreshell4(q1,R1,R2,R3,R4,1,r12,r13,r14) + b1
    y2 = s2 * P_coreshell4(q2,R1,R2,R3,R4,1,r22,r23,r24) + b2 
   
    y2_res = res_eff(q2,y2)
    
    return [y1,y2_res]

####################################################################
# coreshell with rasberry layer
####################################################################

def P_coreshell3_rasberry(q,r1,r2,r3,r4,p1,p2,p3,p4):
    """
    Form factor: core-shell particle (3 layers: 1 core and 2 shells) with closely packed spheres on the surface (like a rasberry)
    parameters
        r1: core radius
        r2: first shell outer radius
        r3: second shell outer radius
        r4: radii of small spheres on surface
        p1: SLD of core
        p2: SLD of shell 1
        p3: SLD of shell 2
        p4: SLD of small spheres on surface
    references: 
        Pedersen 2001: https://doi.org/10.1063/1.1339221
        Larson-Smith Jackson Pozza 2010: https://doi.org/10.1016/j.jcis.2009.11.033
    """
    P_s = P_coreshell3(q,r1,r2,r3,p1,p2,p3) # form factor core+shell1+shell2
    P_c = P_sphere(q,r4) # form factor sphere
    r_eff = r3+r4
    sincx = sinc(q*r_eff)
    S_sc = psi_coreshell3(q,r1,r2,r3,p1,p2,p3)*psi_sphere(q,r4)*sincx
    S_cc = P_sphere(q,r4)*sincx**2
    V1,V2,V3,V4 = V_sphere(r1),V_sphere(r2),V_sphere(r3),V_sphere(r4)
    b_s = V1*p1 + (V2-V1)*p2 + (V3-V2)*p2
    b_c = V4*p4
    N_agg = (r3+r4)*np.pi/r4
    A = N_agg*b_s**2*P_s + b_c**2*P_c + (N_agg-1)*b_c**2*S_cc + 2*N_agg*b_s*b_c*S_sc
    B = N_agg*b_s**2 + b_c**2 + (N_agg-1)*b_c**2 + 2*N_agg*b_s*b_c
    return A/B

def coreshell3_ratio_2_rasberry(q1,q2,R1,R2,R3,R4,r12,r13,r14,r22,r23,r24,s1,b1,s2,b2):
    """
    Model: core-shell particle (3 layers: 1 core and 2 shells) with closely packed spheres on the surface (like a rasberry)
    2 datasets
    parameters
        r1: core radius
        r2: first shell outer radius
        r3: second shell outer radius
        r4: radii of small spheres on surface
        p1: SLD of core
        p2: SLD of shell 1
        p3: SLD of shell 2
        p4: SLD of small spheres on surface
        s1,s2: scaling of each dataset
        b1,b2: background to each dataset
    references: 
        Pedersen 2001: https://doi.org/10.1063/1.1339221
        Larson-Smith Jackson Pozza 2010: https://doi.org/10.1016/j.jcis.2009.11.033
    """
    y1 = s1 * P_coreshell3_rasberry(q1,R1,R2,R3,R4,1,r12,r13,r14) + b1
    y2 = s2 * P_coreshell3_rasberry(q2,R1,R2,R3,R4,1,r22,r23,r24) + b2
    return [y1,y2]

 
