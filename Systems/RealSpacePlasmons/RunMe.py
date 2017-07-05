from numpy import *
from matplotlib.pyplot import *
from scipy.special import j0
from scipy.interpolate import interp1d

rs=[470.8,470.8,938.6,938.6,1400.7,1400.7,1854.1,1854.1,\
    2296.1,2296.1,2723.9,2723.9,3135.0,3135.0,3526.7,3526.7,\
    3896.7,3896.7,4242.6,4242.6,4562.4,4562.4,4854.1,4854.1,\
    5115.8,5115.8,5346.0,5346.0,5543.3,5543.3,5706.3,5706.3,\
    5834.2,5834.2,5926.1,5926.1,5981.5,5981.5,6000.0,6000.0,\
    5981.5,5981.5,5926.1,5926.1,5834.2,5834.2,5706.3,5706.3,\
    5543.3,5543.3,5346.0,5346.0,5115.8,5115.8,4854.1,4854.1,\
    4562.4,4562.4,4242.6,4242.6,3896.7,3896.7,3526.7,3526.7,\
    3135.0,3135.0,2723.9,2723.9,2296.1,2296.1,1854.1,1854.1,\
    1400.7,1400.7,938.6,938.6,470.8]

qnm=logspace(-3,0,500)
#qnm=logspace(-3,2,500)*1/(25.) #1e-3 to 1e2 inverse tip radii
qr=linspace(0,2*max(rs)*max(qnm),15000000)
besselmatrix=j0(qr)
interp_besselmatrix_at=interp1d(qr,besselmatrix)

def besselcollective(rs,qnm):
    
    btot=0
    for i in range(len(rs)):
        ii=i+1
        btot+=interp_besselmatrix_at(rs[i]*qnm)*(-1)**(round(ii/2)+1)
        
    return btot

btot=1+besselcollective(rs,qnm)
figure(); plot(qnm,btot)
xlabel('qnm')
ylabel('Btot')
