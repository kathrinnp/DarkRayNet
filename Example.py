### Simple example to test the DarkRayNet and check if it runs smoothly.
### For a more visual example please refer to the 'Example_Notebook.ipynb'

# Step 1 - Inititate the DRN simulation tool

from simulation_tool import DRN
DRN = DRN()
#
import numpy as np
import time

# First Example

print()
print('First Example: Simulate DM components of the antiproton flux, with fixed propagation parameters and branching fractions and increasing values for the DM mass.')


# Step 2 - Define parameters for simulation inputs

N = 50

prop_params = np.array([1.8, 1.79, 2.405, 2.357, 7.92e+03, 0.37, 2.05e+28, 0.419, 8.84, 0.09, 2.60]) * np.ones((N, 11)) # N identical sets of parameters

DM_masses = np.logspace(1, 3, N) # N different masses 

DM_branching_fractions = np.ones((N,8)) * 1e-4 # set all branching fractions to 0.01% 
DM_branching_fractions[:, 2] = 1 - 7e-4  # only the branching fraction corresponding to b bbar has a dominant conribution

# Step 3 - Define the desired output spectra

Particle_List = ['DM Antiprotons']

# Step 4 - Run this predict function

t0 = time.time()
Output = DRN.predict(Particle_List, prop_params, DM_mass=DM_masses, DM_fs=DM_branching_fractions)
t1 = time.time()

# Step 5 - Do whatever you want with the output

Spectra, Energy_bins = Output[0]

print()
print("It took ", float(t1-t0), " seconds to predict ", N*len(Particle_List), " cosmic ray spectra.")
print()
print("The Output spectra have the shape ", Spectra.shape, " (N, number of energy bins)")
print()
print("An examplary spectrum for a DM mass of %.1f GeV might look like this (GeV^-1 m^-2 sr^-1 s^-1)"%DM_masses[-1])
print(Spectra[-1])
print()
print("with the corresponding energy bins (GeV)")
print(Energy_bins)

# Second example 

print()
print('Second Example: Simulate various cosmic ray spectra with one set of fixed propagation parameters.')


# Step 2 - Define parameters for simulation inputs

N = 1

prop_params = np.array([1.8, 1.79, 2.405, 2.357, 7.92e+03, 0.37, 2.05e+28, 0.419, 8.84, 0.09, 2.60]) 

# Step 3 - Define the desired output spectra

Particle_List = ['Secondary Antiprotons', 'Protons', 'Deuterium', 'Helium 3', 'Helium 4']

# Step 4 - Run this predict function

t2 = time.time()
Output = DRN.predict(Particle_List, prop_params)
t3 = time.time()

# Step 5 - Do whatever you want with the output

pbar_s, Energy_pbar_s = Output[0]
p, Energy_p = Output[1]
D, Energy_D = Output[2]
He3, Energy_He3 = Output[3]
He4, Energy_He4 = Output[4]

print()
print("It took ", float(t3-t2), " seconds to predict ", N*len(Particle_List), " cosmic ray spectra of different particle types.")
print()
print("As an example, the proton spectrum might look like this (GeV^-1 m^-2 sr^-1 s^-1)")
print(p)
print()
print("with the corresponding energy bins (GeV)")
print(Energy_p)
