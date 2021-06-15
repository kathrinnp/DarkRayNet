# Welcome to the Dark Ray Net
## A Neural Network Based Simulation Tool for Indirect Dark Matter Searches

The recurrent neural networks provided in this tool can quickly simulate antiprotons, protons and Helium cosmic ray spectra at Earth, for an extensive range of parameters. The antiproton spectra consist of both a contribution of secondary emission and a component resulting from dark matter annihilation into various standard model particles that contribute to antiproton fluxes.
The tool is designed to predict measurable cosmic ray spectra for thousands of parameter sets in few seconds and thus enables quick parameter scans for indirect dark matter searches.
 
The following provides an introduction of the tool and its functions, as well as a description of the neural network involved and (physical) assumption on which the training data is based. 

### Table of Contents
* [The Simulation Tool](#The-Simulation-Tool)
    * [Requirements](#Requirements)
	* [Functions](#Functions)
* [Recurrent Neural Networks](#Recurrent-Neural-Networks)
* [Physical Assumptions](#Physical-Assumptions)
	* [Simulation Setup](#Simulation-Setup)
	* [Allowed Parameter Ranges](#Allowed-Parameter-Ranges)
 
## The Simulation Tool

### Requirements

This tool is based entirely on **Python**. 
The packages needed to execute the functions in this tool are:

 - NumPy 
 - Tensorflow (version 2.0 or more recent)
 - h5py (version 2.10.0)
 
 ### Functions
**__init__** (prevent_extrapolation  =  True)
 - Loads the neural network
 - Options: 
*prevent_extrapolation* (default: True) If any of the input parameters are outside of the trained parameter regions this raises a warning and the corresponding cosmic ray fluxes are zero. 

**predict** (particle_list,  propagation_parameters,  DM_mass  =  None,  DM_fs  =  None,  sigma_v  =  None)

Inputs:

 - *List of desired comic ray spectra*
		 - Options:  'DM Antiprotons, 'Secondary Antiprotons', 'Protons', 'Deuterium', 'Helium 3', 'Helium 4'  
 - *Propagation Parameters*
		 - List or array of sets of the propagation parameters (i.e shape=(10,) or (N,10)). 
		 - Order and units: gamma_1,p, gamma_1, gamma_2,p, gamma_2, R_0 [MV], s_0, D_0 [cm^2/s], delta, v_Alfven [km/s], v_0,c [km/s], z_h [kpc] (see following section for more details)
 - *Dark Matter Mass*
		 - Default: None
		 - Required if list of desired comic ray spectra contains 'DM Antiprotons'
		 - Scalar or List/1D Array of length N (desired number of simulated fluxes)
		 - Input in units $\log_{10}(m_{DM}/$MeV$)$
 - *Dark Matter Branching fractions "fs"*
		 - Default: None
		 - Required if list of desired comic ray spectra contains 'DM Antiprotons'
		 - Shape = (8,) or (N,8)
		 - Please normalize your branching fractions so that for each flux the sum of the fractions are 1. 
 - *Dark Matter Annihilation Cross Section*
		 - Default: None
		 - Only relevant if list of desired comic ray spectra contains 'DM Antiprotons', if not given will be set to default ($3 \cdot 10^{-26}$ cm$^3$ s$^-1$)
		 - Scalar or List/1D Array of length N (desired number of simulated fluxes)

Outputs:

 - List of tuples (flux, energy bins) for each element in the list of desired fluxes. 
		 - The length of the energy bins (len(E)) can vary for different particle types, as they're adjusted to the available measurements (AMS-02, Voyager)
		 - Shape of each flux array: (len(E), ) or (N, len(E))

Cosmic Ray spectra of identical charge number are evaluated at the identical energy bins and can thus easily be added. 

## Recurrent Neural Networks

There is a total of six neural networks implemented in the Dark Ray Net tool. 

Here we give a brief overview. More details can be found in ***TO DO: add link to arXiv here***
### Physical Assumptions

**Cosmic Ray Propagation**

All cosmic ray spectra are simulated with GALPROP []. We assume a propagation model that can be represented by the following parameters: 

Sources of primary cosmic rays:
 - $\gamma_1, \gamma_2$ Spectral indices of the rigidity dependent source term (Index $_p$ for protons) 
 - $R_0$ Rigidity break and $s_0$ smoothing factor of the rigidity dependent source term.
 
Propagation:

 - Half height of the diffusion halo $z_h$
 - Normalization $D_0$ and slope $\delta$ of spatial diffusion
 - Alfven velocity $v_\text{alfven}$
 -  Convection velocity $v_\text{conv}$
 
  We assume only secondary emission of cosmic ray antiprotons through processes in the ISM. Further, we always include 'tertiary' antiprotons in the secondary component and 'tertiary DM antiprotons' in the DM antiproton component. 

**Dark Matter Annihilation**

We assume WIMP dark matter that is present in our Galaxy in a NFW [1] density profile relation. 
The spectra from its annihilation into standard model particles is provided by Cirelli et al. [2] in their tool PPPC4DMID. 

### Architecture and Training

### Allowed Parameter Ranges



[1] Julio F. Navarro, Carlos S. Frenk, and Simon D. M. White. “The Structure of Cold Dark Matter Halos”. In: The Astrophysical Journal 462 (1996), p. 563. ISSN : 1538-4357. DOI : 10.1086/177173.

[2] Marco Cirelli et al. “PPPC 4 DM ID: A Poor Particle Physicist Cookbook for Dark Matter Indirect Detection”. In: Journal of Cosmology and Astroparticle Physics 2011.03 (2010), pp. 051–051. DOI : 10.1088/1475-7516/2011/03/051. arXiv: 1012.4515.

