from numpy import *
from matplotlib.pyplot import *
from common.plotting import *

rcParams['xtick.major.size']=10
rcParams['xtick.minor.size']=5
rcParams['ytick.major.size']=10
rcParams['ytick.minor.size']=5
tick_width=1.5

c=299792458. #in m/sec
hbar=6.582e-16 #in eV*sec

#Borders of the infrared spectrum
lambda1=.78e-6 #in m
lambda_mid=3e-6
lambda_far=50e-6
lambda2=1000e-6

lambdas_m=linspace(lambda1,lambda2,5000) #wavelengths in m
lambdas_cm=lambdas_m*100 #wavelengths in cm
lambdas_nm=lambdas_m*1e9 #wavelengths in nm

freqs_Hz=c/lambdas_m #frequencies in Hz
freqs_THz=freqs_Hz/1e12 #frequencies in THz

evs=freqs_Hz*hbar #energies in eVs

##Plot THz vs. wavelengths##
f=figure()
semilogx(lambdas_nm,freqs_THz,color='k',lw=3)
minor_ticks(y=True,x=False)
ylim(freqs_THz.min(),freqs_THz.max())
ax1=gca()
xaxis=ax1.xaxis
yaxis=ax1.yaxis

#Set y axis label
ylabel('$\omega \, [THz]$',fontsize=17)

#Set infrared range delimiters#
axvline(lambda_mid*1e9,color='k',lw=2,ls='--')
axvline(lambda_far*1e9,color='k',lw=2,ls='--')

#Set x axis color
color='b'
ax1.spines['bottom'].set_color(color)
xaxis.label.set_color(color)
grid(color='k',lw=1.5)
xlabel('$\lambda \, [nm]$',fontsize=17)

#Set x axis ticks
xaxis.tick_bottom()
for t in xaxis.get_ticklines():
    t.set_markeredgewidth(tick_width)
    t.set_color(color)
for t in xaxis.get_minorticklines():
    t.set_markeredgewidth(tick_width)
    t.set_color(color)
for t in ax1.get_xticklabels():
    t.set_color(color)

#Set x axis labels
##Doesn't work for x axis??
#xlims=xlim()
#base_lims=[floor(log(x)/log(10.)) for x in xlims]
#base_range=max(base_lims)-min(base_lims)

#sub_ticks=[1,2,5]
#ticklocs=[]
#for i in range(base_range):
#    base=min(base_lims)+i
#    ticklocs+=[sub_tick*10**base for \
#               sub_tick in sub_ticks]
#xaxis.set_minor_formatter(SelectTickDecorator(xaxis.get_major_formatter(),ticklocs=ticklocs))

#Set y axis ticks
major_ts=xaxis.get_ticklines()
for t in major_ts: t.set_markeredgewidth(tick_width)
minor_ts=xaxis.get_minorticklines()
for t in minor_ts: t.set_markeredgewidth(tick_width)

##Plot eVs vs. wavelengths##
twinx()
semilogx(lambdas_nm,evs)
ylim(evs.min(),evs.max())
lines=gca().lines
lines.remove(lines[-1])
ax2=gca()
yaxis=ax2.yaxis

#Set y axis color
color='r'
ax1.spines['right'].set_color(color)
ax2.spines['right'].set_color(color)
yaxis.label.set_color(color)
grid(color=color,lw=1.5)
ylabel('$\omega \, [eV]$',fontsize=17,rotation=270)

#Set y axis ticks
for t in yaxis.get_ticklines():
    t.set_markeredgewidth(tick_width)
    t.set_color(color)
for t in yaxis.get_minorticklines():
    t.set_markeredgewidth(tick_width)
    t.set_color(color)
for t in ax2.get_yticklabels():
    t.set_color(color)
    
#Plot eVs vs. 1/cm
#f.sca(ax1)
twiny()
semilogx(1/lambdas_cm,freqs_THz)
ylim(evs.min(),evs.max())
lines=gca().lines
lines.remove(lines[-1])
ax3=gca()
xaxis=ax3.xaxis
ax3.set_xlim(ax3.get_xlim()[::-1])

#Set x axis color
color='g'
ax1.spines['top'].set_color(color)
ax2.spines['top'].set_color(color)
xaxis.label.set_color(color)
grid(color='b',lw=1.5)
xlabel('$k \, [cm^{-1}]$',fontsize=17)

#Set x axis ticks
for t in xaxis.get_ticklines():
    t.set_markeredgewidth(tick_width)
    t.set_color(color)
for t in xaxis.get_minorticklines():
    t.set_markeredgewidth(tick_width)
    t.set_color(color)
for t in ax3.get_xticklabels():
    t.set_color(color)
    
##Add infrared range labels
bbox=dict(facecolor='wheat',boxstyle='round')
ax3.text(.1,.4,'Near-IR',transform=ax.transAxes, fontsize=17,verticalalignment='top', bbox=bbox)
ax3.text(.45,.4,'Mid-IR',transform=ax.transAxes, fontsize=17,verticalalignment='top', bbox=bbox)
ax3.text(.8,.4,'Far-IR',transform=ax.transAxes, fontsize=17,verticalalignment='top', bbox=bbox)

#Add legend label
line=ax1.lines[0]
legend([line],['IR Light Line'],shadow=True,loc='upper left',bbox_to_anchor=(.15,.85))