import numpy as np
from NearFieldOptics import Materials as M
import matplotlib
matplotlib.use('TKAgg') #This backend works with 
from matplotlib import pyplot as plt
plt.ion()

#--- Metals
print("Let's make some new materials.  Let's start with metals.")
plasma_frequencies=(1000,1200,1400)
print('The plasma frequency is the natural resonance frequency of electrons in the metal.')
print('Plasma frequencies = %s wave-numbers (cm^-1).'%repr(plasma_frequencies))
print('Wave-numbers are a measure of energy for spectroscopists.')

print()
input('Continue? [y]:  ')
print()

scattering_frequency=800
print('The scattering rate is simply the mean scattering time of an electron, '+\
      'first converted to a frequency, then to an energy.')
print('Scattering frequency = %s cm^-1.'%scattering_frequency)

print()
input('Continue? [y]:  ')
print()

eps_infinity=1
print('The `background` dielectric constant (so-called `epsilon at infinite frequency`) '+\
      'of a material reflects the polarizability of a material arising from high-energy '+\
      'resonances (e.g. ultraviolet or X-ray).')
print('eps_infinity = %s.'%eps_infinity)
metals=[M.IsotropicMaterial(drude_params=(plasma_frequency,\
                                          scattering_frequency),\
                            eps_infinity=eps_infinity) \
        for plasma_frequency in plasma_frequencies]

print()
input('Continue? [y]:  ')
print()

print("We've generated some metals.  Let's evaluate their complex dielectric constants "+\
      'and plot versus frequency.')
frequencies=np.linspace(400,2000,500) #Again, in units of cm^-1
f1=plt.figure()
for metal in metals:
    eps=metal.epsilon(frequencies)
    plt.plot(frequencies,eps.real,\
             label='$\omega_p=%s$ cm$^{-1}$'%metal.drude_params[0])
    plt.plot(frequencies,eps.imag,linestyle='--',\
             color=plt.gca().lines[-1].get_color())

ax=plt.gca()
handles, labels = ax.get_legend_handles_labels()
handles.insert(0,ax.lines[0]); labels.insert(0,r'Re[$\varepsilon$]')
handles.insert(1,ax.lines[1]); labels.insert(1,r'Im[$\varepsilon$]')
plt.legend(handles,labels,loc='best')
plt.ylabel(r'$\varepsilon$')
plt.xlabel('$\omega$  (cm$^{-1}$)')
plt.grid(alpha=.3)
plt.draw(); plt.show(block=False)

print()
input('Continue? [y]:  ')
print()

