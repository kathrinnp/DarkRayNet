## DarkRayNet
# Quick cosmic ray antiproton simulations for dark matter annihilation models using neural networks

Requirements: Numpy, Tensorflow v2.0

## Antiproton simulation class:

- call with antiproton_sim()

- functions:

	- single_sim:
	
	inputs: propagation parameters (shape = (11,)), m_DM (python scalar), fs_i (shape = (8,))
	
	outputs: total flux, DM flux, secondary flux (each shape = (28,)), E_bins (shape = (28,))
	
	- N_sim:
	
	inputs: propagation parameters (shape = (N,11)), m_DM (shape = (N,)), fs_i (shape = (N,8))
	
	outputs: total flux, DM flux, secondary flux (each shape = (N, 28)), E_bins (shape = (28,))
	
	### More details:
	
	Inputs: 

	- Dark matter mass in log10(m_DM/MeV) 

	- Branching fractions in qq, cc, bb, tt, W+W-, ZZ, gg, hh (need to be normalized to sum(fs_i) = 1)

	- Propagation parameters gamma_1,p, gamma_1, gamma_2,p, gamma_2, R_0 [MV], s_0, D_0 [cm^2/s], delta, v_Alfven [km/s], v_0,c [km/s], z_h [kpc]

	Outputs: 

	- Total antiproton flux

	- DM antiproton flux

	- Secondary antiproton flux

	- Energy bins corresponding to the distinct flux values (GeV)

	The fluxes are provided in [GeV^-1 m^2 sr^-1 s^-1].
	
## Primary (p, He) simulation class:

- call with primary_sim()

- functions:

	- single_sim:
	
	inputs: propagation parameters (shape = (11,))
	
	outputs: proton flux, Deuterium flux, Helium 4 flux, Helium 3 flux, Energy bins (each shape = (28,))
	
	- N_sim:
	
	inputs: propagation parameters (shape = (N,11))
	
	outputs: proton flux, Deuterium flux, Helium 4 flux, Helium 3 flux (each shape = (N,28), E_bins (shape = (28,))
	
	### More details:
	
	Inputs: 

	- Propagation parameters gamma_1,p, gamma_1, gamma_2,p, gamma_2, R_0 [MV], s_0, D_0 [cm^2/s], delta, v_Alfven [km/s], v_0,c [km/s], z_h [kpc]

	Outputs: 

	- Proton flux

	- Deuterium flux

	- Helium 4 flux

	- Helium 3 flux

	- Energy bins (GeV)

	The fluxes are provided in [GeV^-1 m^2 sr^-1 s^-1].
