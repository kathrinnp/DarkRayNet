## DarkRayNet
# Quick cosmic ray antiproton simulations for dark matter annihilation models using neural networks

Requirements: Numpy, Tensorflow v2.0

Inputs: 

- Dark matter mass in log10(m_DM/GeV) 

- Branching fractions in qq, cc, bb, tt, W+W-, ZZ, gg, hh (need to be normalized to sum(fs_i) = 1)
	  
- Propagation parameters z_h [kpc], D_0 [cm2/s], delta, v_Alfven [km/s], v_0,c [km/s], R_0 [MV], s_0, gamma_1, gamma_2, gamma_1,p, gamma_2,p 
          
Outputs: 

- Total antiproton flux

- DM antiproton flux
	 
- Secondary antiproton flux
	 
- Energy bins corresponding to the distinct flux values
         
         
