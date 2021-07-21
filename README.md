# Welcome to the Dark Ray Net  

<img src="https://user-images.githubusercontent.com/55040575/125958166-4c48af03-21a2-4371-83bc-fdd346f02b3b.png" width="256">

## A Neural Network Based Simulation Tool for Indirect Dark Matter Searches

The recurrent neural networks provided in this tool can quickly simulate antiprotons, protons and Helium cosmic ray (CR) spectra at Earth, for an extensive range of parameters. 
The training of the provided networks is based on GALPROP [1] simulations.

The antiproton spectra consist of both a contribution of secondary emission and a component resulting from dark matter (DM) annihilation into various Standard Model particles that contribute to antiproton fluxes.
The tool is designed to predict measurable cosmic ray spectra for thousands of parameter sets in few seconds and thus enables quick parameter scans for indirect dark matter searches.
 
The following provides an introduction of the tool and its functions, as well as a description of the neural network involved and (physical) assumption on which the training data is based. 

**If you choose to use this tool, please cite *TO DO: add link to arXiv here*** 
 
If you have questions or problems, file an issue in this repository or contact "nippel *at* physik *dot* rwth-aachen *dot* de"

________________________________________________________________
 
### Table of Contents
* [The Simulation Tool](#the-simulation-tool)
    * [Requirements and Installation](#requirements-and-installation)
    * [Functions](#functions)
    * [Examples](#examples)
* [Further Information](#further-information)
    * [Artificial Neural Networks](#artificial-neural-networks)
    * [Physical Assumptions](#physical-assumptions)
    * [Allowed Parameter Ranges](#allowed-parameter-ranges)
	* [Performance](#performance)
 
 ________________________________________________________________

## The Simulation Tool

### Requirements and Installation

This tool is based entirely on **Python 3**. 
The packages needed to execute the functions in this tool are:

 - NumPy 
 - Tensorflow (version 2.3.1)
 - h5py (version 2.10.0)
 - Jupyter Notebook (optional, for example notebook)
 - Matplotlib (optional, for example notebook)
 
Optionally, you can run 

	source install_environment

to create and activate a virtual environment within the DarkRayNet directory where the correct dependencies are automatically installed. After running this file the environment can be deactivated with the 

	deactivate 
	
command and for the use of the tool with the corresponding dependencies reactivated with 

	source env/bin/activate
 
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
		 - The length of the energy bins (len(E)) can vary for different particle types, as they're adjusted to the available measurements (AMS-02 [2], Voyager [3])
		 - Shape of each flux array: 
		 
		 (N, length(energy bins)) or (length(energy bins),) if N=1

Cosmic Ray spectra of identical charge number are evaluated at the identical energy bins and can thus easily be added. 

**create_propagation_parameters** (gamma_1p = 1.80, gamma_1 = 1.79, gamma_2p = 2.405, gamma_2 = 2.357, R_0 = 7.92e3, s = 0.37, D_0 = 2.05e28, delta = 0.419, v_alfven = 8.84, v_0c = 0.09, z_h = 2.60, N_identical = 1)

- Helper function that provides an array of propagation parameters suitable for the input of the predict function. 
- Input parameter defaults are taken from a fit of simulated antiproton, proton and helium fluxes to AMS-02 [2] and Voyager [3] data, see table 1 in the attached paper. 
- N_identical is set to 1 but can be increased if multiple identical sets of parameters are desired, for example for evaluation multiple sets of DM parameters at once.
- Output: numpy array of shape (11,) or (N_identical,11) if N_identical > 1.

### Examples

We have set up two examples to aid further understanding of the usage of the tool.

- A python file that prints out examplary cosmic ray spectra based on arbitrarily defined input parameters. This file can also be run to check whether all requirements are installed correctly
- A jupyter notebook in which some exemplary spectra are plotted for a visualization of the outputs of the tool. 

________________________________________________________________

## Further Information

### Artificial Neural Networks

There is a total of six artificial neural networks (ANNs) implemented in the Dark Ray Net tool, each corresponds to one of the particle types. Note that for the secondary antiprotons we automatically include tertiary antiprotons, tertiary DM antiprotons are included in the DM antiprotons and secondary protons are included in the proton spectra. 
The neural networks have a build in recurrent layer and are implemented using the Keras API [4] and Tensorflow as backend [5]. For a detailled description of the architectures and the training process see ***TO DO: add link to arXiv here***
### Physical Assumptions

We only give a very brief overview here. Please refer to ***TO DO: add link to arXiv here*** for a detailled description. 

**Cosmic Ray Propagation**

All cosmic ray spectra used for the network training are simulated with GALPROP [1]. We assume a propagation model that can be represented by the following parameters: 

Sources of primary cosmic rays:
 - $\gamma_1, \gamma_2$ Spectral indices of the rigidity dependent source term (Index $_p$ for protons) 
 - $R_0$ Rigidity break and $s_0$ smoothing factor of the rigidity dependent source term.
 
Propagation:

 - Half height of the diffusion halo $z_h$
 - Normalization $D_0$ and slope $\delta$ of spatial diffusion
 - Alfven velocity $v_\text{alfven}$
 -  Convection velocity $v_\text{conv}$
 
We assume only secondary emission of cosmic ray antiprotons through processes in the ISM.

Note that we do not implement any consideration of solar modulation in this tool.

**Dark Matter Annihilation**

We assume WIMP dark matter that is present in our Galaxy in a NFW density profile relation. 
The spectra from its annihilation into Standard Model particles is provided by Cirelli et al. [6] in their tool PPPC4DMID. 

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

### Performance

The accuracy of the ANNs was tested in the development phase and we found that each cosmic ray flux predicted by the networks within the trained parameter regions differs from the simulations by a magnitude significantly below the measurement uncertainites of the AMS-02 data [2] and thus only marginally affect any likelihood evaluations. Solely the prediction of the DM antiproton flux around edges of the energy range differs more noticeable relative to the simulation but again, the effect realtive to the magnitude of the measurement is miniscule. We elaborate further on this in our paper. 

The prediction times of this tool depend on the number of selected CR particle types. You can simulate a few tousand spectra of one particle type in only one second. For multiple spectra multiple networks have to be called because of which the simulation time can increase to a couple of seconds. Regardless, this tool accelerates the evaluation of CR fluxes significantly with respect to non-ANN-based methods. 
________________________________________________________________

[1] https://galprop.stanford.edu/

[2] AMS Collaboration, M. Aguilar et al., The Alpha Magnetic Spectrometer (AMS) on the
international space station: Part II — Results from the first seven years, Phys. Rept. 894
(2021) 1–116. https://www.sciencedirect.com/science/article/pii/S0370157320303434?via%3Dihub

[3] E. C. Stone, A. C. Cummings, F. B. McDonald, B. C. Heikkila, N. Lal, et al., Voyager 1, http://dx.doi.org/10.1126/science.1236408

[4] https://keras.io/api/

[5] https://www.tensorflow.org/

[6] Marco Cirelli et al. “PPPC 4 DM ID: A Poor Particle Physicist Cookbook for Dark Matter Indirect Detection”. In: Journal of Cosmology and Astroparticle Physics 2011.03 (2010), pp. 051–051. DOI : 10.1088/1475-7516/2011/03/051. arXiv: 1012.4515.
