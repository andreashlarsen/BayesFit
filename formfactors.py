import numpy as np
from scipy.special import jv,gamma
from numpy import exp,sin,cos,sqrt,pi

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
    return np.sinc(x/pi)

####################################################################
# sphere
####################################################################

def psi_sphere(q,r):
    """
    Form factor amplitude of a sphere
    """
    x = q*r     
    return 3*(sin(x)-x*cos(x))/x**3
    
def P_sphere(q,r):
    """
    Form factor of a sphere
    """
    return psi_sphere(q,r)**2   
   
def V_sphere(r):
    """
    Volume of a sphere
    """
    return 4*r**3*pi/3

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
# ellipsoid
####################################################################

def psi_ellipsoid(q,R,eps,alpha):
    """
    Form factor amplitude: ellipsoid of revolution
    """
    r_effective = R*sqrt(sin(alpha)**2 + eps**2*cos(alpha)**2)
    psi = psi_sphere(q,r_effective)

    return psi

def P_ellipsoid(q,R,eps):
    """
    Form factor amplitude: ellipsoid of revolution
    """
    N_alpha,alpha_max = 20,pi/2
    da = alpha_max/N_alpha
    alpha = np.linspace(da/2,pi/2-da/2,N_alpha)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += psi_ellipsoid(q,R,eps,a)**2*sin(a)
    return P_sum*da

def V_ellipsoid(R,eps):
    """
    Volume: ellipsoid of revolution
    """
    return V_sphere(R)*eps

def ellipsoid(q,a,b,scale,background):
    """
    Model: ellipsoid of revolution
    """
    R = a
    eps = b/a
    I = scale * P_ellipsoid(q,R,eps) + background

    return I

def ellipsoid_eps(q,R,eps,scale,background):
    """
    Model: ellipsoid of revolution
    parametrized with ellipticity (eps) and minor axis (R)
    """
    I = scale * P_ellipsoid(q,R,eps) + background

    return I

def psi_ellipsoid_rotate(q,R,eps,alpha):
    """
    Form factor amplitude: ellipsoid of revolution - rotated
    """
    r_effective = R*sqrt(eps**2*sin(alpha)**2 + cos(alpha)**2)
    psi = psi_sphere(q,r_effective)

    return psi

####################################################################
# cylinder
####################################################################

def V_cyl(R,L):
    """
    Volume of a cylinder with radius R and length L
    """
    return pi*R**2*L

def psi_cyl(q,R,L,a):
    """ 
    form factor amplitude: cylinder with radius R and length L
    depends on view angle a (for alpha)
    reference: Pedersen1997
    """
    x = q*R*sin(a)
    z = q*L*cos(a)/2.0
    psi=2.0*bessj1c(x)*sinc(z)
    return psi

def psi_elliptical_cyl(q,r,L,eps,a,b):
    """
    form factor amplitude: elliptical cylinder
    eps: ellipticity (epsilon)
    R: effective radius
    """
    re = r*eps
    R = sqrt(r**2*sin(b)**2+re**2*cos(b)**2)

    psi = psi_cyl(q,R,L,a)

    #x = q*R
    #z = q*L*cos(a)/2.0
    #psi = 2.0 * bessj1c(x)*sinc(z)

    return psi 

def P_elliptical_cyl(q,r,L,eps):
    """ 
    form factor: elliptical cylinder
    """

    N_alpha,alpha_max = 160,pi/2
    N_beta,beta_max = 160,pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)
    beta = np.linspace(beta_max/N_beta,beta_max,N_beta)

    # inner loop vectorized
    b = beta.reshape(-1,1)
    P_a = np.zeros(len(q))
    for a in alpha:
        P_b = np.sum(psi_elliptical_cyl(q,r,L,eps,a,b)**2,axis=0)
        P_a += P_b*sin(a)

    P_a *= 2/pi

    # normalization
    P = P_a/P_a[0]
   
    print('.',end='')
 
    return P

def V_elliptical_cyl(R,L,eps):
    """"
    Volume elliptical cylinder with radius R, lenght L and ellipticity eps
    """
    return V_cyl(R,L)*eps

def psi_elliptical_cyl_coreshell(q,R,L,Lc,eps,pc,ps,a,b):
    """
    form factor amplitude: elliptical cylinder core-shell
    R: minor axis
    L: total length
    Lc: length core
    eps: ellipticity
    pc: delta SLD core
    ps: delta SLD shell
    a,b: orientational angles
    """
    
    A_core = pc * V_elliptical_cyl(R,Lc,eps) * psi_elliptical_cyl(q,R,Lc,eps,a,b)
    norm_core = pc * V_elliptical_cyl(R,Lc,eps)

    A_shell = ps * ( V_elliptical_cyl(R,L,eps)*psi_elliptical_cyl(q,R,L,eps,a,b) - V_elliptical_cyl(R,Lc,eps) * psi_elliptical_cyl(q,R,Lc,eps,a,b) )
    norm_shell = ps * ( V_elliptical_cyl(R,L,eps) - V_elliptical_cyl(R,Lc,eps) )

    A_coreshell = A_core + A_shell
    norm_coreshell = norm_core + norm_shell

    psi = A_coreshell/norm_coreshell

    return psi 

def P_elliptical_cyl_coreshell(q,R,L,Lc,eps,pc,ps):
    """
    form factor: elliptical cylinder core-shell
    R: minor axis
    L: total length
    Lc: length core
    eps: ellipticity
    pc: delta SLD core
    ps: delta SLD shell
    """
    N_alpha,alpha_max = 80,pi/2
    N_beta,beta_max = 80,pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)
    beta = np.linspace(beta_max/N_beta,beta_max,N_beta)

    # inner loop vectorized
    b = beta.reshape(-1,1)
    P_a = np.zeros(len(q))
    for a in alpha:
        P_b = np.sum(psi_elliptical_cyl_coreshell(q,R,L,Lc,eps,pc,ps,a,b)**2,axis=0)
        P_a += P_b*sin(a)

    P_a *= 2/pi

    # normalization
    P = P_a/P_a[0]

    return P

def elliptical_cylinder_coreshell(q,R,L,Lc,eps,psc,scale,background):
    """
    model: elliptical cylinder, core shell
    R: minor axis
    L: total length
    L: length of core
    eps: ellipticity
    psc: ratio of delta SLD for shell with respect to core
    scaled and background subtracted
    """

    return scale * P_elliptical_cyl_coreshell(q,R,L,Lc,eps,1,psc) + background

def elliptical_cylinder(q,R,L,eps,scale,background):
    """
    model: elliptical cylinder
    with background and scaling
    """    
    return scale * P_elliptical_cyl(q,R,L,eps) + background

def P_cyl(q,R,L):
    """
    form factor: cylinder with radius R and length L
    """
    N_alpha = 30
    alpha = np.linspace(0,pi/2,N_alpha)
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

    N_alpha,alpha_max = 50,pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)

    # this is slower than a simple for loop 
    #a = alpha.reshape(-1,1)
    #P_a = np.sum(abs(psi_stacked_2cyl(q,R1,R2,L1,L2,p1,p2,a))**2*sin(a),axis=0)
    P_a = np.zeros(len(q))
    for a in alpha:
        P_a += abs(psi_stacked_2cyl(q,R1,R2,L1,L2,p1,p2,a))**2 * sin(a)

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

    N_alpha,alpha_max = 50,pi/2
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

def psi_cylinders_spaced(q,R,L,d,a):
   """
   Form factor amplitude: cylinders spaced by void
   """
   A = V_cyl(R,L)*psi_cyl(q,R,L,a) - V_cyl(R,d)*psi_cyl(q,R,d,a)
   B = V_cyl(R,L) - V_cyl(R,d)
   psi = A/B

   return psi

def P_cylinders_spaced(q,R,L,d):
   """
   Form factor: 2 cylinders spaced by void
   """
   alpha = np.linspace(0,pi/2,60)
   P_sum = np.zeros(len(q))
   for a in alpha:
       P_sum += psi_cylinders_spaced(q,R,L,d,a)**2 * sin(a)

   P = P_sum/P_sum[0]
 
   return P

def cylinders_spaced(q,R,L,d,scale,background):
   """
   Model: 2 cylinders separated by void
   """
   
   return scale * P_cylinders_spaced(q,R,L,d) + background

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
    pos3 = cos(b)*cos(pi/3.0)+sin(b)*sin(pi/3.0)
    pf3 = np.exp(1j*q*d*sin(a)*pos3)

    # form factor amplitude
    pV1,pV2,pV3 = p1*V1,p2*V2,p3*V3
    psi_cyl_tri = pV1*psi1*pf1 + pV2*psi2*pf2 + pV3*psi3*pf3

    return psi_cyl_tri/(pV1 + pV2 + pV3)

def P_cylinder_dimer(q,R,L1,L2,p1,p2):
    """
    Form factor: cylinder dimer
    """
    
    N_alpha,alpha_max = 100,pi/2
    N_beta,beta_max = 100,pi
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
    
    N_alpha,alpha_max = 50,pi
    N_beta,beta_max = 50,2*pi
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

def psi_nanodisc(q,R,Hc,H,Hb,T,pc,ps,pb,a):
    """
    form factor amplitude: nanodisc
    R: radius core
    Hc: height core (2xlipid tail group)
    H: total height (core and shells) 
    Hb: height belt (protein or polymer)
    T: thickness belt (protein or polymer)
    pc: deltaSLD core
    ps: deltaSLD shell
    pb: deltaSLD belt
    a: alpha (orientation parameter)
    """
    A_core = pc * V_cyl(R,Hc) * psi_cyl(q,R,Hc,a)
    norm_core = pc * V_cyl(R,Hc)

    A_shell = ps * (V_cyl(R,H) * psi_cyl(q,R,H,a) - V_cyl(R,Hc) * psi_cyl(q,R,Hc,a) )
    norm_shell = ps * (V_cyl(R,H) - V_cyl(R,Hc))

    A_belt = pb * (V_cyl(R+T,Hb) * psi_cyl(q,R+T,Hb,a) - V_cyl(R,Hb) * psi_cyl(q,R,Hb,a))
    norm_belt = pb * (V_cyl(R+T,Hb) - V_cyl(R,Hb))

    A_nanodisc = A_core + A_shell + A_belt
    norm_nanodisc = nomr_core + norm_shell + norm_belt

    psi_nanodisc = A_nanodisc/norm_nanodisc

    return psi_nanodisc

def P_nanodisc(q,R,Hc,H,Hb,T,pc,ps,pb):
    """
    form factor: nanodisc
    """ 
    N_alpha,alpha_max = 50,pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)

    P_a = np.zeros(len(q))
    for a in alpha:
        P_a += abs(psi_nanodisc(q,R,Hc,H,Hb,T,pc,ps,pb,a))**2 * sin(a)

    # normalization
    P = P_a/P_a[0]

        
def P_nanodisc_Nlip(q,Nlip,Vtail,Vhead,T,pc,ps,pb):
    """
    reparametrization of the nanodisc form factor in terms of Nlip
    N: total number of lips
    Ac: area of core
    R : radius of core
    Vlip: volume of 1 lipid
    Vc: volume of core
    Hc: height core
    Hb: height belt
    V: total volume of core and shells
    H: total height of core and shells
    """
    Ac = Alip*Nlip/2
    R = sqrt(Ac/pi**2)

    Vc = Vtail*Nlip
    Hc = Vc/Ac
    Hb = Hc

    V = (Vhead+Vtail)*Nlip
    H = Vcs/Ac
   
    return P_nanodisc(q,R,Hc,H,Hb,T,pc,ps,pb)

def psi_elliptical_nanodisc(q,R,H,Hc,Hb,T,eps,pc,ps,pb,a,b):
    """
    form factor amplitude: elliptical nanodiscs
    """
    A_core = pc * V_elliptical_cyl(R,Hc,eps) * psi_elliptical_cyl(q,R,Hc,eps,a,b)
    norm_core = pc * V_elliptical_cyl(R,Hc,eps)

    A_shell = ps * ( V_elliptical_cyl(R,H,eps)*psi_elliptical_cyl(q,R,H,eps,a,b) - V_elliptical_cyl(R,Hc,eps) * psi_elliptical_cyl(q,R,Hc,eps,a,b) )
    norm_shell = ps * ( V_elliptical_cyl(R,H,eps) - V_elliptical_cyl(R,Hc,eps) )

    A_belt = pb * ( V_elliptical_cyl(R+T,Hb,eps) * psi_elliptical_cyl(q,R+T,Hb,eps,a,b) - V_elliptical_cyl(R,Hb,eps) * psi_elliptical_cyl(q,R,Hb,eps,a,b) )
    norm_belt = pb * ( V_elliptical_cyl(R+T,Hb,eps) - V_elliptical_cyl(R,Hb,eps) )

    A_elliptical_nanodisc = A_core + A_shell + A_belt
    norm_elliptical_nanodisc = norm_core + norm_shell + norm_belt

    psi = A_elliptical_nanodisc/norm_elliptical_nanodisc

    return psi

def P_elliptical_nanodisc(q,R,Hc,H,Hb,T,eps,pc,ps,pb):
    """
    form factor: elliptical nanodiscs
    """
    N_alpha,alpha_max = 30,pi/2
    N_beta,beta_max = 30,pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)
    beta = np.linspace(beta_max/N_beta,beta_max,N_beta)

    # inner loop vectorized
    b = beta.reshape(-1,1)
    P_a = np.zeros(len(q))
    for a in alpha:
        P_b = np.sum(psi_elliptical_nanodisc(q,R,H,Hc,Hb,T,eps,pc,ps,pb,a,b)**2,axis=0)
        P_a += P_b*sin(a)

    P_a *= 2/pi

    # normalization
    P = P_a/P_a[0]

    return P

def elliptical_nanodisc(q,R,Hc,H,Hb,T,eps,psc,pbc,scale,background):
    """
    model: elliptical nanodisc
    psc: delta SLD of shell / delta SLD of core
    pbc: delta SLD of belt / delta SLD of core
    """ 
    pc = 1
    ps = psc
    pb = pbc

    return scale * P_elliptical_nanodisc(q,R,Hc,H,Hb,T,eps,pc,ps,pb) + background    

def elliptical_nanodisc2(q,R,Hc,H,Hb,T,eps,psc1,pbc1,psc2,pbc2,scale1,background1,scale2,background2):
    """
    model: elliptical nanodisc, 2 datasets
    psc1: delta SLD of shell / delta SLD of core, for dataset 1
    pbc1: delta SLD of belt / delta SLD of core, for dataset 1
    """

    y1 = elliptical_nanodisc(q,R,Hc,H,Hb,T,eps,psc1,pbc1,scale1,background1)     
    y2 = elliptical_nanodisc(q,R,Hc,H,Hb,T,eps,psc2,pbc2,scale2,background2)
    
    return [y1,y2] 
    
####################################################################
# nanodisc (specific parametrization)
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
# torus
####################################################################

# Gummel et al, Soft Matter, 2011, 7: 5731-5738, Concentration dependent pathways in spontaneous self-assembly of unilamellar vesicles

def psi_elliptical_torus(q,a,b,c,alpha):
    """
    Form factor amplitude of torus with elliptical cross section
    a: major radii of elliptical cross section
    b: minor radii of elliptical cross section
    c: distance from center of torus to middle of cross section
    alpha: parameter for orientational averaging
    """
    
    radius = np.linspace(c-0.999*a,c+0.999*a,20)
    gamma = (b/a)*sqrt(a**2-(radius-c)**2)
    psi_sum = np.zeros(len(q))
    for (r,g) in zip(radius,gamma):
        psi_sum += 2*pi*bessj0(q*r*sin(alpha))*sin(q*g*cos(alpha))*2*cos(alpha)/q
    psi = psi_sum/psi_sum[0]

    return psi

def psi_torus(q,r,R,alpha):
    return psi_elliptical_torus(q,r,r,R,alpha)

def P_elliptical_torus(q,a,b,c):
    """
    Form factor of torus with elliptical cross section
    a: major radii of elliptical cross section
    b: minor radii of elliptical cross section
    c: distance from center of torus to middle of cross section
    """
    
    N_alpha = 30
    alpha = np.linspace(0,pi/2,N_alpha)
    P_sum = np.zeros(len(q))
    for aa in alpha:
        P_sum += psi_elliptical_torus(q,a,b,c,aa)**2*sin(aa)

    P = P_sum/P_sum[0]
    return P

def P_torus(q,r,R):
    return P_elliptical_torus(q,r,r,R)

def elliptical_torus(q,a,b,c,scale,background):
    """
    Model: torus with elliptical cross section
    a: major radii of elliptical cross section
    b: minor radii of elliptical cross section
    c: distance from center of torus to middle of cross section
    """

    return scale * P_elliptical_torus(q,a,b,c) + background

def torus(q,r,R,scale,background):
    """
    Model: torus
    r: radius of cross section
    R: distance from center of torus to middle of cross section
    """

    return scale * P_torus(q,r,R) + background

def V_torus(r,R):
    """
    Volume, torus
    r: radius of cross section
    R: distance from center of torus to middle of cross section
    """
    #A = pi*r**2 # cross sectional area
    #C = 2*pi*R # circumference
    #return A*C
    return V_elliptical_torus(r,r,R)

def V_elliptical_torus(a,b,R):
    """
    Volume: elliptical torus
    a,b: semiaxes of cross section
    R: distance from center of torus to middle of cross section
    """
    A = pi*a*b
    C = 2*pi*R
    return A*C

####################################################################
# supercylinder (superegg/superellipsoid)
####################################################################

# DOI: 10.1021/acsnano.6b08089

def beta(x,y):
    return gamma(x)*gamma(y)/gamma(x+y)

def V_supercylinder(R,t,eps):
    return 4*pi/(3*t)*beta(2/t,1/t)*R**3*eps

def psi_supercylinder(q,R,t,eps,a):
    """
    Form factor amplitude: supercylinder
    """
    N_z,z_max = 20,R*eps
    dz = z_max/N_z
    zz = np.linspace(dz/2,z_max-dz/2,N_z)
    psi_sum = np.zeros(len(q))
    for z in zz:
        r = (R**t - abs(z/eps)**t)**(1.0/t)
        x = q*r*sin(a)
        y = q*z*cos(a)
        psi_sum += 2*pi*r**2*2*bessj1c(x)*cos(y)
    psi = psi_sum*dz/V_supercylinder(R,t,eps)
    return psi

def P_supercylinder(q,R,t,eps):
    """
    Form factor: supercylinder
    """
    N_a,alpha_max = 20,pi/2
    da = alpha_max/N_a
    alpha = np.linspace(da/2,alpha_max-da/2,N_a)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += psi_supercylinder(q,R,t,eps,a)**2*sin(a)
    P = P_sum*da
    return P

  
def supercylinder(q,R,t,eps,scale,background):
   return scale * P_supercylinder(q,R,t,eps) + background

def supercylinder_smooth(q,R,t,eps,sigmaR,scale,background):
   return scale * P_supercylinder(q,R,t,eps)*exp(-q**2*sigmaR**2) + background

####################################################################
# hollow supercylinder (hollow superegg/superellipsoid)
#################################################################### 

def psi_hollow_supercylinder(q,R,t,eps,r,a):
    """
    Form factor amplitude: hollow supercylinder
    """
    L = 2*R*eps
    A = V_supercylinder(R,t,eps)*psi_supercylinder(q,R,t,eps,a) - V_cyl(r,L) * psi_cyl(q,r,L,a)
    B = V_supercylinder(R,t,eps) - V_cyl(r,L)
    psi = A/B
    return psi

def P_hollow_supercylinder(q,R,t,eps,r):
    """
    Form factor: hollow supercylinder
    """
    N_a,alpha_max = 20,pi/2
    da = alpha_max/N_a
    alpha = np.linspace(da/2,alpha_max-da/2,N_a)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += psi_hollow_supercylinder(q,R,t,eps,r,a)**2*sin(a)
    P = P_sum*da
    return P

def V_hollow_supercylinder(R,t,eps,r):
   L = 2*R*eps
   return V_supercylinder(R,t,eps) - V_cyl(r,L)

def hollow_supercylinder(q,R,t,eps,r,scale,background):
   return scale * P_hollow_supercylinder(q,R,t,eps,r) + background

####################################################################
# hollow supercylinder with torus "crown"
####################################################################

def psi_hollow_supercylinder_torus(q,R,t,eps,r,r_t,R_t,a):
    """
    Form factor amplitude: hollow supercylinder with torus crown
    """

    # phase factor torus "crown"
    d_t = R*eps*0.9
    qr_t = q*d_t*cos(a)
    pf_t = exp(1j*qr_t)

    A = V_hollow_supercylinder(R,t,eps,r)*psi_hollow_supercylinder(q,R,t,eps,r,a) + V_torus(r_t,R_t) * psi_torus(q,r_t,R_t,a) * pf_t
    B = V_hollow_supercylinder(R,t,eps,r) + V_torus(r_t,R_t)
    psi = A/B
    return psi

def P_hollow_supercylinder_torus(q,R,t,eps,r,r_t,R_t):
    """
    Form factor amplitude: hollow supercylinder with torus crown
    """
    N_a,alpha_max = 20,pi/2
    da = alpha_max/N_a
    alpha = np.linspace(da/2,alpha_max-da/2,N_a)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += abs(psi_hollow_supercylinder_torus(q,R,t,eps,r,r_t,R_t,a))**2*sin(a)
    P = P_sum*da
    return P

def hollow_supercylinder_torus(q,R,t,eps,r,r_t,R_t,scale,background):
   return scale * P_hollow_supercylinder_torus(q,R,t,eps,r,r_t,R_t) + background

def hollow_supercylinder_torus_smooth(q,R,t,eps,r,r_t,R_t,sigmaR,scale,background):
   return scale * P_hollow_supercylinder_torus(q,R,t,eps,r,r_t,R_t)*exp(-q**2*sigmaR**2) + background

####################################################################
# 3 toroids
####################################################################

def psi_tri_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,a):
    """
    Form factor amplitude: three toroids, shifted along z
    """
    # phase factors
    qr2,qr3 = q*d2*cos(a),q*d3*cos(a)
    pf2,pf3 = exp(1j*qr2),exp(1j*qr3)

    A = V_torus(r1,R1)*psi_torus(q,r1,R1,a) + V_torus(r2,R2)*psi_torus(q,r2,R2,a)*pf2 + V_torus(r3,R3)*psi_torus(q,r3,R3,a)*pf3
    B = V_torus(r1,R1) + V_torus(r2,R2) + V_torus(r3,R3)

    psi = A/B
 
    return psi

def P_tri_tori(q,r1,R1,r2,R2,d2,r3,R3,d3):
    """
    Form factor: three toroids, shifted along z
    """
    N_alpha = 20
    alpha = np.linspace(0,pi/2,N_alpha)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += abs(psi_tri_tori(q,r1,R1,r2,R2,d2,r3,r3,d3,a))**2*sin(a)
    P = P_sum/P_sum[0]

    return P

def tri_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,scale,background):
    return scale * P_tri_tori(q,r1,R1,r2,R2,d2,r3,R3,d3) + background


####################################################################
# 4 toroids
####################################################################

def psi_four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4,a):
    """
    Form factor amplitude: four toroids, shifted along z
    """
    # phase factors
    qr2,qr3,qr4 = q*d2*cos(a),q*d3*cos(a),q*d4*cos(a)
    pf2,pf3,pf4 = exp(1j*qr2),exp(1j*qr3),exp(1j*qr4)
    
    # volumes
    V1,V2,V3,V4 = V_torus(r1,R1),V_torus(r2,R2),V_torus(r3,R3),V_torus(r4,R4)

    # psi
    A = V1*psi_torus(q,r1,R1,a) + V2*psi_torus(q,r2,R2,a)*pf2 + V3*psi_torus(q,r3,R3,a)*pf3 + V4*psi_torus(q,r4,R4,a)*pf4
    B = V1 + V2 + V3 + V4
    psi = A/B

    return psi

def P_four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4):
    """
    Form factor: four toroids, shifted along z
    """
    N_alpha,alpha_max = 20,pi/2
    da = alpha_max/N_alpha
    alpha = np.linspace(da/2,pi/2-da/2,N_alpha)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += abs(psi_four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4,a))**2*sin(a)
    P = P_sum*da
    
    return P

def four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4,scale,background):
    return scale * P_four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4) + background

def four_tori_constraint(q,r1,R1,r3,r4,R4,spacer,scale,background):
    r2,R2 = r1,R1
    R3 = R4-r4-r3
    d2 = 2*r1+spacer
    d3 = 3*r1+r3+2*spacer
    d4 = d3

    return scale * P_four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4) + background

####################################################################
# 4 elliptical tori
####################################################################

def psi_four_elliptical_tori(q,a1,b1,R1,a2,b2,R2,d2,a3,b3,R3,d3,a4,b4,R4,d4,a):
    """
    Form factor amplitude: four elliptical toroids, shifted along z
    """
    # phase factors
    qr2,qr3,qr4 = q*d2*cos(a),q*d3*cos(a),q*d4*cos(a)
    pf2,pf3,pf4 = exp(1j*qr2),exp(1j*qr3),exp(1j*qr4)

    # volumes
    V1,V2,V3,V4 = V_elliptical_torus(a1,b1,R1),V_elliptical_torus(a2,b2,R2),V_elliptical_torus(a3,b3,R3),V_elliptical_torus(a4,b4,R4)

    # psi
    A = V1*psi_elliptical_torus(q,a1,b1,R1,a) + V2*psi_elliptical_torus(q,a2,b2,R2,a)*pf2 + V3*psi_elliptical_torus(q,a3,b3,R3,a)*pf3 + V4*psi_elliptical_torus(q,a4,b4,R4,a)*pf4
    B = V1 + V2 + V3 + V4
    psi = A/B

    return psi

def P_four_elliptical_tori(q,a1,b1,R1,a2,b2,R2,d2,a3,b3,R3,d3,a4,b4,R4,d4):
    """
    Form factor: four elliptical toroids, shifted along z
    """
    N_alpha,alpha_max = 20,pi/2
    da = alpha_max/N_alpha
    alpha = np.linspace(da/2,pi/2-da/2,N_alpha)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += abs(psi_four_elliptical_tori(q,a1,b1,R1,a2,b2,R2,d2,a3,b3,R3,d3,a4,b4,R4,d4,a))**2*sin(a)
    P = P_sum*da

    return P

def four_elliptical_tori(q,a1,b1,R1,a2,b2,R2,d2,a3,b3,R3,d3,a4,b4,R4,d4,scale,background):
    return scale * P_four_elliptical_tori(q,a1,b1,R1,a2,b2,R2,d2,a3,b3,R3,d3,a4,b4,R4,d4) + background

def four_elliptical_tori_constraint(q,r1,eps1,R1,r3,eps3,r4,R4,eps4,scale,background):
    a1,b1 = r1,r1*eps1
    a2,b2,R2 = a1,b1,R1
    a3,b3 = r3,r3*eps3
    a4,b4 = r4,r4*eps4
    R3 = R4-a4-a3
    d2 = 2*b1
    d3 = 3*b1+b3
    d4 = d3

    return scale * P_four_elliptical_tori(q,a1,b1,R1,a2,b2,R2,d2,a3,b3,R3,d3,a4,b4,R4,d4) + background

####################################################################
# ellipsoid with cylinder channel
####################################################################

def psi_ellipsoid_channel(q,R,eps,R_c,L_c,a):
    """
    Form factor: ellipsoid with cylinder channel
    """
    A = V_ellipsoid(R,eps)*psi_ellipsoid(q,R,eps,a) - V_cyl(R_c,L_c)*psi_cyl(q,R_c,L_c,a)
    B = V_ellipsoid(R,eps) - V_cyl(R_c,L_c)

    psi = A/B

    return psi

def P_ellipsoid_channel(q,R,eps,R_c,L_c):
    
    N_alpha,alpha_max = 20,pi/2
    da = alpha_max/N_alpha
    alpha = np.linspace(da/2,alpha_max-da/2,N_alpha)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += abs(psi_ellipsoid_channel(q,R,eps,R_c,L_c,a))**2*sin(a)
    P = P_sum/P_sum[0]

    return P

def ellipsoid_channel(q,R,eps,R_c,L_c,scale,background):
    return scale * P_ellipsoid_channel(q,R,eps,R_c,L_c) + background

####################################################################
# ellipsoid with channel with torus "crown"
####################################################################


def psi_ellipsoid_channel_torus(q,R,eps,R_c,L_c,r_t,R_t,d_t,a):
    """
    Form factor: ellipsoid with cylinder channel, torus "crown", shifted along z
    """
    
    # phase factor torus "crown"
    qr_t = q*d_t*cos(a)
    pf_t = exp(1j*qr_t)

    A = V_ellipsoid(R,eps)*psi_ellipsoid_rotate(q,R,eps,a) - V_cyl(R_c,L_c)*psi_cyl(q,R_c,L_c,a) + V_torus(r_t,R_t)*psi_torus(q,r_t,R_t,a)*pf_t
    B = V_ellipsoid(R,eps) - V_cyl(R_c,L_c) + V_torus(r_t,R_t)

    psi = A/B

    return psi
    
def P_ellipsoid_channel_torus(q,R,eps,R_c,L_c,r_t,R_t,d_t):
    """
    Form factor: ellipsoid with cylinder channel, torus "crown", shifted along z
    """
    N_alpha,alpha_max = 15,pi/2
    da = alpha_max/N_alpha
    alpha = np.linspace(da/2,alpha_max-da/2,N_alpha)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += abs(psi_ellipsoid_channel_torus(q,R,eps,R_c,L_c,r_t,R_t,d_t,a))**2*sin(a)
    return P_sum*da
 
def ellipsoid_channel_torus(q,R,eps,R_c,r_t,R_t,scale,background):
    
    L_c = 2*R*eps*0.95
    d_t = L_c/2

    return scale * P_ellipsoid_channel_torus(q,R,eps,R_c,L_c,r_t,R_t,d_t) + background

####################################################################
# 4 toroids
####################################################################

#def psi_four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4,a):
#    """
#    Form factor amplitude: four toroids, shifted along z
#    """
#    # phase factoRs
#    qr2,qr3,qr4 = q*d2*cos(a),q*d3*cos(a),q*d4*cos(a)
#    pf2,pf3,pf4 = exp(1j*qr2),exp(1j*qr3),exp(1j*qr4)

#    A = V_torus(r1,R1)*psi_torus(q,r1,R1,a) + V_torus(r2,R2)*psi_torus(q,r2,R2,a)*pf2 + V_torus(r3,R3)*psi_torus(q,r3,R3,a)*pf3 + V_torus(r4,R4)*psi_torus(q,r4,R4,a)*pf4
#    B = V_torus(r1,R1) + V_torus(r2,R2) + V_torus(r3,R3) + V_torus(r4,R4)

#    psi = A/B

#    return psi

#def P_four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4):
#    """
#    Form factor: four toroids, shifted along z
#    """
#    N_alpha = 20
#    alpha = np.linspace(0,pi/2,N_alpha)
#    P_sum = np.zeros(len(q))
#    for a in alpha:
#        P_sum += abs(psi_four_tori(q,r1,R1,r2,R2,d2,r3,r3,d3,r4,R4,d4,a))**2*sin(a)
#    P = P_sum/P_sum[0]

#    return P

#def four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4,scale,background):
#    return scale * P_four_tori(q,r1,R1,r2,R2,d2,r3,R3,d3,r4,R4,d4) + background



####################################################################
# hollow cylinder
####################################################################

def psi_hollow_cylinder(q,r,R,L,a):
    """"
    Form factor amplitude: hollow cylinder
    r: inner radius
    R: outer radius
    L: length
    """
    A = V_cyl(R,L)*psi_cyl(q,R,L,a) - V_cyl(r,L)*psi_cyl(q,r,L,a)
    B = V_cyl(R,L)- V_cyl(r,L)
    psi = A/B
 
    return psi

def P_hollow_cylinder(q,r,R,L):
    """
    Form factor: hollow cylinder
    r: inner radius
    R: outer radius
    L: length
    """
    alpha = np.linspace(0,pi/2,30)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += psi_hollow_cylinder(q,r,R,L,a)**2*sin(a)
    P = P_sum/P_sum[0]

    return P

def hollow_cylinder(q,r,R,L,scale,background):
    """
    Model: hollow cylinder
    r: inner radius
    R: outer radius
    L: length    
    """
    return scale * P_hollow_cylinder(q,r,R,L) + background

def V_hollow_cylinder(r,R,L):
    return V_cyl(R,L) - V_cyl(r,L)

####################################################################
# hollow cylinders spaced
####################################################################

def psi_hollow_cylinders_spaced(q,r,R,L,d,a):
   """
   Form factor amplitude: hollow cylinders spaced by void
   """
   A = V_hollow_cylinder(r,R,L)*psi_hollow_cylinder(q,r,R,L,a) - V_hollow_cylinder(r,R,d)*psi_hollow_cylinder(q,r,R,d,a)
   B = V_hollow_cylinder(r,R,L) - V_hollow_cylinder(r,R,d)
   psi = A/B

   return psi

def P_hollow_cylinders_spaced(q,r,R,L,d):
   """
   Form factor: hollow cylinders spaced by void
   """
   alpha = np.linspace(0,pi/2,60)
   P_sum = np.zeros(len(q))
   for a in alpha:
       P_sum += psi_hollow_cylinders_spaced(q,r,R,L,d,a)**2 * sin(a)

   P = P_sum/P_sum[0]

   return P

def V_hollow_cylinders_spaced(r,R,L,d):
   return V_hollow_cylinder(r,R,L-d)
 
def hollow_cylinders_spaced(q,r,R,L,d,scale,background):
   """
   Model: hollow cylinders separated by void
   """

   return scale * P_hollow_cylinders_spaced(q,r,R,L,d) + background

####################################################################
# cylinder with torus crown
####################################################################

def psi_cylinder_torus(q,r_c,R_c,L,r_t,R_t,d,a):
    """"
    Form factor amplitude: hollow cylinder with torus "crown"
    torus shifted with distance d along z
    a: alpha, for orientational averaging
    """

    # phase factor
    qr_t = q*d*cos(a)
    pf_t = np.exp(1j*qr_t)
    
    A = V_hollow_cylinder(r_c,R_c,L)*psi_hollow_cylinder(q,r_c,R_c,L,a) + V_torus(r_t,R_t)*psi_torus(q,r_t,R_t,a)*pf_t
    B = V_hollow_cylinder(r_c,R_c,L) + V_torus(r_t,R_t)

    psi = A/B
 
    return psi

def P_cylinder_torus(q,r_c,R_c,L,r_t,R_t,d):
    """ 
    Form factor: 
    """
    N_alpha,alpha_max = 20,pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)

    P_a = np.zeros(len(q))
    for a in alpha:
        P_a += abs(psi_cylinder_torus(q,r_c,R_c,L,r_t,R_t,d,a))**2 * sin(a)
     
    P = P_a/P_a[0]
    return P

def cylinder_torus(q,r_c,R_c,L,r_t,R_t,d,scale,background):
    """
    Model:
    """
    return scale * P_cylinder_torus(q,r_c,R_c,L,r_t,R_t,d) + background

def cylinder_torus_fix(q,r_c,L,r_t,R_t,scale,background):
    """
    Model:
    """
    return scale * P_cylinder_torus(q,r_c,R_t-r_t,L,r_t,R_t,L/2) + background

####################################################################
# hollow cylinder with hollow cylinder crown
####################################################################

def psi_hollow_cylinder_crown(q,r,R,L,r_c,R_c,L_c,d_c,a):
    """"
    Form factor amplitude: hollow cylinder with hollow cylinder "crown"
    """
    
    #phase factors
    qr_c = q*d_c*cos(a)
    pf_c = np.exp(1j*qr_c)

    A = V_hollow_cylinder(r,R,L)*psi_hollow_cylinder(q,r,R,L,a) + V_hollow_cylinder(r_c,R_c,L_c)*psi_hollow_cylinder(q,r_c,R_c,L_c,a)*pf_c
    B = V_hollow_cylinder(r,R,L) + V_hollow_cylinder(r_c,R_c,L_c)
    
    psi = A/B

    return psi

def P_hollow_cylinder_crown(q,r,R,L,r_c,R_c,L_c,d_c):
    """"
    Form factor: hollow cylinder with hollow cylinder "crown"
    """
    N_alpha,alpha_max = 20,pi/2
    alpha = np.linspace(alpha_max/N_alpha,alpha_max,N_alpha)
    
    P_a = np.zeros(len(q))
    for a in alpha:
        P_a += abs(psi_hollow_cylinder_crown(q,r,R,L,r_c,R_c,L_c,d_c,a))**2 * sin(a)
    P = P_a/P_a[0]

    return P

def hollow_cylinder_crown(q,r,R,L,r_c,R_c,L_c,d_c,scale,background):
    """
    Model: hollow cylinder with hollow cylinder "crown"
    """
    return scale * P_hollow_cylinder_crown(q,r,R,L,r_c,R_c,L_c,d_c) + background     
     
####################################################################
# hollow cylinders with space between and torus crown
####################################################################

def psi_hollow_cylinders_spaced_torus(q,r_c,R_c,L,d_c,r_t,R_t,d_t,a):
    """
    Form factor amplitude: hollow cylinders spaced by void, with torus "crown"
    """

    # phase factor
    qr_t = q*d_t*cos(a)
    pf_t = np.exp(1j*qr_t)    

    A = V_hollow_cylinders_spaced(r_c,R_c,L,d_c)*psi_hollow_cylinders_spaced(q,r_c,R_c,L,d_c,a) + V_torus(r_t,R_t)*psi_torus(q,r_t,R_t,a)*pf_t
    B = V_hollow_cylinders_spaced(r_c,R_c,L,d_c) + V_torus(r_t,R_t)

    psi = A/B

    
    return psi

def P_hollow_cylinders_spaced_torus(q,r_c,R_c,L,d_c,r_t,R_t,d_t):
    """
    Form factor: hollow cylinders spaced by void, with torus "crown"
    """
    alpha = np.linspace(0,pi/2,30)
    P_a = np.zeros(len(q))
    for a in alpha:
        P_a += abs(psi_hollow_cylinders_spaced_torus(q,r_c,R_c,L,d_c,r_t,R_t,d_t,a))**2 * sin(a)
    P = P_a/P_a[0]
    return P

def hollow_cylinders_spaced_torus_fix(q,r_c,L,d_c,r_t,R_t,scale,background):
   """
   Model: hollow cylinders spaced by void, with torus "crown"
   """
   return scale * P_hollow_cylinders_spaced_torus(q,r_c,R_t-r_t,L,d_c,r_t,R_t,L/2) + background

####################################################################
# ellipsoid and torus
####################################################################

def psi_ellipsoid_torus(q,r,eps,r_t,R_t,d_t,a):
    """
    Form factor amplitude: ellipsoid and torus
    """
    # phase factor
    qr_t = q*d_t*cos(a)
    pf_t = np.exp(1j*qr_t)

    A = V_ellipsoid(r,eps)*psi_ellipsoid(q,r,eps,a) + V_torus(r_t,R_t)*psi_torus(q,r_t,R_t,a)*pf_t
    B = V_ellipsoid(r,eps) + V_torus(r_t,R_t)

    psi = A/B

    return psi

def P_ellipsoid_torus(q,r,eps,r_t,R_t,d_t):
    """
    Form factor: ellipsoid and torus
    """
    alpha = np.linspace(0,pi/2,30)
    P_sum = np.zeros(len(q))
    for a in alpha:
        P_sum += abs(psi_ellipsoid_torus(q,r,eps,r_t,R_t,d_t,a))**2 * sin(a)
    P = P_sum/P_sum[0]
    return P

def ellipsoid_torus(q,r,eps,r_t,R_t,d_t,scale,background):
    return scale * P_ellipsoid_torus(q,r,eps,r_t,R_t,d_t) + background

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
    sinA = sin(A)
    cosA = cos(A)
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
    N_agg = (r3+r4)*pi/r4
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

####################################################################
# MD
####################################################################
 
def md_saxs(q,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,scale,background):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0    
    #for logw,f in zip([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10],[71,98,95,54,42,27,14,15,20,8]):
    for logw,f in zip([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10],[3,20,30,40,50,60,70,80,90,100]):
        Ifit = np.genfromtxt('frames/frame%d-SASDNJ2.fit' % f,usecols=[3],skip_header=6,unpack=True)
        w = 10**logw 
        y += w*Ifit
        sumw += w
    y /= sumw

    return scale * y + background

def md_sans(q,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,scale,background):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """
    y,sumw = 0,0
    #for logw,f in zip([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10],[71,98,95,54,42,27,14,15,20,8]):
    for logw,f in zip([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10],[3,20,30,40,50,60,70,80,90,100]):
        Ifit = np.genfromtxt('frames/ring_frame%d-SASDNK2_deut0_d2o100.fit' % f,skip_header=6,usecols=[3],unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw

    return scale * y + background

def md_sas(qx,qn,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,scalex,backgroundx,scalen,backgroundn):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """
    y1 = md_saxs(qx,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,scalex,backgroundx)
    y2 = md_sans(qn,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,scalen,backgroundn)

    return [y1,y2]

####################################################################
# MD 4 frames
####################################################################

def md_saxs_w4(q,w1,w2,w3,w4,scale,background):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    #for logw,f in zip([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10],[71,98,95,54,42,27,14,15,20,8]):
    for logw,f in zip([w1,w2,w3,w4],[3,20,30,40]):
        Ifit = np.genfromtxt('frames/frame%d-SASDNJ2.fit' % f,usecols=[3],skip_header=6,unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw

    return scale * y + background

def md_sans_w4(q,w1,w2,w3,w4,scale,background):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """
    y,sumw = 0,0
    #for logw,f in zip([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10],[71,98,95,54,42,27,14,15,20,8]):
    for logw,f in zip([w1,w2,w3,w4],[3,20,30,40]):
        Ifit = np.genfromtxt('frames/ring_frame%d-SASDNK2_deut0_d2o100.fit' % f,skip_header=6,usecols=[3],unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw

    return scale * y + background

def md_sas_w4(qx,qn,w1,w2,w3,w4,scalex,backgroundx,scalen,backgroundn):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """
    y1 = md_saxs_w4(qx,w1,w2,w3,w4,scalex,backgroundx)
    y2 = md_sans_w4(qn,w1,w2,w3,w4,scalen,backgroundn)

    return [y1,y2]

####################################################################
# MD 3 frames
####################################################################

def md_saxs_w3(q,w1,w2,w3,scale,background):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    for logw,f in zip([w1,w2,w3],[71,42,8]):
        Ifit = np.genfromtxt('frames/frame%d-SASDNJ2.fit' % f,usecols=[3],skip_header=6,unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw

    return scale * y + background

def md_sans_w3(q,w1,w2,w3,scale,background):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """
    y,sumw = 0,0
    for logw,f in zip([w1,w2,w3],[71,42,8]):
        Ifit = np.genfromtxt('frames/ring_frame%d-SASDNK2_deut0_d2o100.fit' % f,skip_header=6,usecols=[3],unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw

    return scale * y + background

def md_sas_w3(qx,qn,w1,w2,w3,scalex,backgroundx,scalen,backgroundn):
    """
    Model: MD sim with AMBER14 and TIP3P water
    input is log10(w)
    """
    y1 = md_saxs_w3(qx,w1,w2,w3,scalex,backgroundx)
    y2 = md_sans_w3(qn,w1,w2,w3,scalen,backgroundn)

    return [y1,y2]

####################################################################
# MD 2 force fields
####################################################################

def md_saxs_w2(q,w1,w2,b1,b2):
    """
    Model: MD sim with AMBER14 OR CHARMM36-IDP and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    for logw,b,f in zip([w1,w2],[b1,b2],[4,6]):
        Ifit = np.genfromtxt('MD%d/frames/SAXS_Ifit_consensus.dat' % f,usecols=[1],skip_header=2,unpack=True)
        w = 10**logw
        y += w*Ifit+b
        sumw += w
    y /= sumw

    return y

def md_sans_w2(q,w1,w2,b1,b2):
    """
    Model: MD sim with AMBER14 OR CHARMM36-IDP and TIP3P water
    input is log10(w)
    """
    y,sumw = 0,0
    for logw,b,f in zip([w1,w2],[b1,b2],[4,6]):
        Ifit = np.genfromtxt('MD%d/frames/SANS_Ifit_consensus.dat' % f,usecols=[1],skip_header=2,unpack=True)
        w = 10**logw
        y += w*Ifit+b
        sumw += w
    y /= sumw

    return y

def md_sas_w2(qx,qn,w1,w2,bx1,bx2,bn1,bn2):
    """
    Model: MD sim with AMBER14 OR CHARMM36-IDP and TIP3P water
    input is log10(w)
    """
    y1 = md_saxs_w2(qx,w1,w2,bx1,bx2)
    y2 = md_sans_w2(qn,w1,w2,bn1,bn2)

    return [y1,y2]

####################################################################
# MD 2 force fields
####################################################################

def md_saxs_ff2_b(q,w1,w2,b1,b2):
    """
    Model: MD sim with AMBER14 and CHARMM36-IDP and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    for logw,b,f in zip([w1,w2],[b1,b2],[4,6]):
        Ifit = np.genfromtxt('MD%d/frames/fit_average_SAXS.dat' % f,unpack=True)
        w = 10**logw
        y += w*Ifit+b
        sumw += w
    y /= sumw

    return y

####################################################################
# MD 2 force fields
####################################################################

def md_saxs_ff2(q,w1,w2):
    """
    Model: MD sim with AMBER14 and CHARMM36-IDP and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    for logw,f in zip([w1,w2],[4,6]):
        Ifit = np.genfromtxt('MD%d/frames/fit_average_SAXS.dat' % f,unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw

    return y

####################################################################
# MD 2 force fields, 1 relative weight
####################################################################

def md_saxs_ff2_w1(q,w1,s):
    """
    Model: MD sim with AMBER14 and CHARMM36-IDP and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    for logw,f in zip([w1,0],[4,6]):
        Ifit = np.genfromtxt('MD%d/frames/fit_average_SAXS.dat' % f,unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw
    y *= s

    return y

def md_sans_ff2_w1(q,w1,s):
    """
    Model: MD sim with AMBER14 and CHARMM36-IDP and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    for logw,f in zip([w1,0],[4,6]):
        Ifit = np.genfromtxt('MD%d/frames/fit_average_SANS.dat' % f,unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw
    y *= s

    return y

def md_sas_ff2_w1(qx,qn,w1,sx,sn):
    """
    Model: MD sim with AMBER14 OR CHARMM36-IDP and TIP3P water
    input is log10(w)
    """
    y1 = md_saxs_ff2_w1(qx,w1,sx)
    y2 = md_sans_ff2_w1(qn,w1,sn)

    return [y1,y2]

####################################################################
# MD 2 force fields, 1 relative weight, no scaling
####################################################################

def md_saxs_ff2_w1_nos(q,w1):
    """
    Model: MD sim with AMBER14 and CHARMM36-IDP and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    for logw,f in zip([w1,0],[4,6]):
        Ifit = np.genfromtxt('MD%d/frames/fit_average_SAXS.dat' % f,unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw

    return y

def md_sans_ff2_w1_nos(q,w1):
    """
    Model: MD sim with AMBER14 and CHARMM36-IDP and TIP3P water
    input is log10(w)
    """

    y,sumw = 0,0
    for logw,f in zip([w1,0],[4,6]):
        Ifit = np.genfromtxt('MD%d/frames/fit_average_SANS.dat' % f,unpack=True)
        w = 10**logw
        y += w*Ifit
        sumw += w
    y /= sumw

    return y

def md_sas_ff2_w1_nos(qx,qn,w1):
    """
    Model: MD sim with AMBER14 OR CHARMM36-IDP and TIP3P water
    input is log10(w)
    """
    y1 = md_saxs_ff2_w1_nos(qx,w1)
    y2 = md_sans_ff2_w1_nos(qn,w1)

    return [y1,y2]

