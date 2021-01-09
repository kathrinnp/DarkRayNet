## DarkRayNet
# Quick cosmic ray antiproton simulations for dark matter annihilation models using neural networks

Requirements: Numpy, Tensorflow v2.0

Inputs: - Dark matter mass in $\log_{10} \left(m_\text{DM}/ \text{GeV} \right)$ 
        - Branching fractions: $\text{fs} \in \lbrace q \overline{q}, c \overline{c}, b \overline{b}, t \overline{t}, W^+ W^-, ZZ, gg, hh \rbrace \, .$
          Need to be normalized to $\text{fs}_i \in \left( 0.001, 100 \right) \%$
        - Propagation parameters:
		      $z_h$ [kpc], $D_0$ [cm$^2$ s$^{-1}$], $\delta$, $v_{\mathrm{Alfven}} [km s$^{-1}$], $v_{0,\mathrm{c}} [km s$^{-1}$], $R_0$ [MV], $s_0$, $\gamma_1$, $\gamma_2$, $\gamma_{\mathrm{1,p}}$, $\gamma_{\mathrm{2,p}}$  
          
Outputs: - Total antiproton flux
         - DM antiproton flux
         - Secondary antiproton flux
         - Energy bins corresponding to the distinct flux values
         
         
