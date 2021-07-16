# Welcome to the Dark Ray Net
## A Neural Network Based Simulation Tool for Indirect Dark Matter Searches

The recurrent neural networks provided in this tool can quickly simulate antiprotons, protons and Helium cosmic ray spectra at Earth, for an extensive range of parameters. The antiproton spectra consist of both a contribution of secondary emission and a component resulting from dark matter annihilation into various standard model particles that contribute to antiproton fluxes.
The tool is designed to predict measurable cosmic ray spectra for thousands of parameter sets in few seconds and thus enables quick parameter scans for indirect dark matter searches.
 
The following provides an introduction of the tool and its functions, as well as a description of the neural network involved and (physical) assumption on which the training data is based. 

**If you choose to use this tool, please cite *TO DO: add link to arXiv here*** 

### Table of Contents
* [The Simulation Tool](#the-simulation-tool)
    * [Requirements](#requirements)
	* [Functions](#functions)
* [Further Information](#further-information)
	* [Artificial Neural Networks](#artificial-neural-networks)
	* [Physical Assumptions](#physical-assumptions)
	* [Allowed Parameter Ranges](#allowed-parameter-ranges)
 
 ________________________________________________________________

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
		 - Options:  

		 'DM Antiprotons, 
		 'Secondary Antiprotons', 
		 'Protons', 
		 'Deuterium', 
		 'Helium 3', 
		 'Helium 4'  

 - *Propagation Parameters*
		 - List or array of sets of the propagation parameters (i.e shape=(10,) or (N,10)). 
		 - Order and units: 
		 
		gamma_1,p, gamma_1, gamma_2,p, gamma_2, R_0 [MV], s_0, D_0 [cm^2/s], delta, v_Alfven [km/s], v_0,c [km/s], z_h [kpc]
		
	 (see following section for more details)
 - *Dark Matter Mass*
		 - Default: None
		 - Required if list of desired comic ray spectra contains 'DM Antiprotons'
		 - Scalar or List/1D Array of length N (desired number of simulated fluxes)
		 - Input in units GeV
 - *Dark Matter Branching fractions "fs"*
		 - Default: None
		 - Required if list of desired comic ray spectra contains 'DM Antiprotons'
		 - Shape = (8,) or (N,8)
		 - Order: 
		 
		 q qbar (q = u+d+s), c cbar, b bbar, t tbar, W+W-, ZZ, gg, hh 
	 Please normalize your branching fractions so that for each flux the sum of the fractions are 1. 
 - *Dark Matter Annihilation Cross Section*
		 - Default: None
		 - Only relevant if list of desired comic ray spectra contains 'DM Antiprotons', if not given will be set to default ($3 \cdot 10^{-26}$ cm$^3$ s$^-1$)
		 - Scalar or List/1D Array of length N (desired number of simulated fluxes)

Outputs:

 - List of tuples (flux, energy bins) for each element in the list of desired fluxes. 
		 - The length of the energy bins (len(E)) can vary for different particle types, as they're adjusted to the available measurements (AMS-02, Voyager)
		 - Shape of each flux array: 
		 
		 (N, length(energy bins)) or (length(energy bins),) if N=1

Cosmic Ray spectra of identical charge number are evaluated at the identical energy bins and can thus easily be added. 

________________________________________________________________

## Further Information

### Artificial Neural Networks

There is a total of six neural networks implemented in the Dark Ray Net tool, each corresponds to one of the particle types. Note that for the secondary antiprotons we automatically include tertiary antiprotons, tertiary DM antiprotons are included in the DM antiprotons and secondary protons are included in the proton spectra. The neural networks are implemented using the Keras API [3] and Tensorflow as backend [4]. For a detailled description of the architectures and the training process see ***TO DO: add link to arXiv here***
### Physical Assumptions

We only give a very brief overview here. Please refer to ***TO DO: add link to arXiv here*** for a detailled description. 

**Cosmic Ray Propagation**

All cosmic ray spectra used for the network training are simulated with GALPROP [5]. We assume a propagation model that can be represented by the following parameters: 

Sources of primary cosmic rays:
 - $\gamma_1, \gamma_2$ Spectral indices of the rigidity dependent source term (Index $_p$ for protons) 
 - $R_0$ Rigidity break and $s_0$ smoothing factor of the rigidity dependent source term.
 
Propagation:

 - Half height of the diffusion halo $z_h$
 - Normalization $D_0$ and slope $\delta$ of spatial diffusion
 - Alfven velocity $v_\text{alfven}$
 -  Convection velocity $v_\text{conv}$
 
  We assume only secondary emission of cosmic ray antiprotons through processes in the ISM.

**Dark Matter Annihilation**

We assume WIMP dark matter that is present in our Galaxy in a NFW [1] density profile relation. 
The spectra from its annihilation into standard model particles is provided by Cirelli et al. [2] in their tool PPPC4DMID. 

________________________________________________________________

### Allowed Parameter Ranges

The neural networks are only accurate in the parameter regions they were trained on. This limits the feasible propagation parameters to

| Parameter | Unit   | Min          | Max       |
|-----------|--------|--------------|-----------|
| gamma_1,p |        | 1.63         | 1.9       |
| gamma_1   |        | 1.6          | 1.88      |
| gamma_2,p |        | 2.38         | 2.45      |
| gamma_2   |        | 2.34         | 2.4       |
| R_0       | MV     | 5000         | 9500      |
| s_0       |        | 0.32         | 0.5       |
| D_0       | cm^2/s | 1.15 * 10^28 | 5.2*10^28 |
| delta     |        | 0.375        | 0.446     |
| v_Alfven  | km/s   | 0            | 11        |
| v_0,c     | km/s   | 0            | 12.8      |
| z_h       | kpc    | 2            | 7         |

These values follow from the data points in the training set: 

<img src="https://user-images.githubusercontent.com/55040575/125588391-83a21ff0-a8b1-4184-a71b-6a54e033ed55.png" width="600">

For the DM parameters the limitations are: 


		5 GeV <= DM Mass <= 5 TeV

and the branching fractions must be chosen to be larger than 10^-5, i.e. 0.001 % and smaller than one , minus the minimal contributions of the remaining branching fractions.
So a maximally dominant branching fraction would be 1 - 7*10^-5 = 0.99993.
For physical reasons, make sure to normalize the branching fractions in such a way that they sum up to one. 

________________________________________________________________

If you have questions or issues file an issue in the repository or contact nippel[at]physik[dot]rwth-aachen[dot]de

________________________________________________________________


[1] Julio F. Navarro, Carlos S. Frenk, and Simon D. M. White. “The Structure of Cold Dark Matter Halos”. In: The Astrophysical Journal 462 (1996), p. 563. ISSN : 1538-4357. DOI : 10.1086/177173.

[2] Marco Cirelli et al. “PPPC 4 DM ID: A Poor Particle Physicist Cookbook for Dark Matter Indirect Detection”. In: Journal of Cosmology and Astroparticle Physics 2011.03 (2010), pp. 051–051. DOI : 10.1088/1475-7516/2011/03/051. arXiv: 1012.4515.

[3] https://keras.io/api/

[4] https://www.tensorflow.org/

[5] https://galprop.stanford.edu/