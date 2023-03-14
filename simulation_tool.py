
from preamble_DRN import *

class DRN:
    def __init__(self, propagation_model = 'DIFF.BRK', prevent_extrapolation = True, init_particles = ["DM Antiprotons", "Secondary Antiprotons"]):
        ''' 
        Update: 
        - add default propagation model (DIFF.BRK),
        - add init_particles, per default 'DM Antiprotons' and 'Secondary Antiprotons'. 
        '''
        self.pe = prevent_extrapolation
        self.init_particles = init_particles
        model_options = ['DIFF.BRK', 'INJ.BRK']
        if propagation_model in model_options:
            self.propagation_model = propagation_model
        else:
            print()
            print('DRN Warning: The propagation model "%s" is not provided in this tool. It will be set to default (DIFF.BRK).'%propagation_model)
            self.propagation_model = 'DIFF.BRK'
        self.dep_path = c_path + '/dependencies/' + self.propagation_model + '/'
        # Check particle types
        self.particle_types = ['DM Antiprotons', 'Secondary Antiprotons', 'Protons', 'Deuterium', 'Helium', 'Helium 3', 'Helium 4']
        for ptype in self.init_particles:
            if not ptype in self.particle_types:
                raise ValueError('The particle type "%s" is not provided in this tool.'%ptype)
        self.load_deps()
        print()
        print('DRN Info: The simulation tool has been initiated.')
        print()

    def load_deps(self):
        # Energy bins
        self.E_bins = np.load(c_path + '/dependencies/E.npy')
        self.E_all = np.load(c_path + '/dependencies/E_all.npy')
        self.E_ext = np.load(c_path + '/dependencies/E_ext.npy')
        # DM antiprotons
        if 'DM Antiprotons' in self.init_particles:
            self.DM_model = tf.keras.models.load_model(self.dep_path + 'DM_model_x.h5')
            self.DM_trafos = np.load(self.dep_path + 'DM_trafos_x.npy', allow_pickle = True)
        # Secondary antiprotons
        if 'Secondary Antiprotons' in self.init_particles:
            self.S_model = tf.keras.models.load_model(self.dep_path + 'S_model.h5')
            self.S_trafos = np.load(self.dep_path + 'S_trafos.npy', allow_pickle = True)
        # Protons
        if ('Protons'in self.init_particles) or ('Deuterium' in self.init_particles):
            self.p_model = tf.keras.models.load_model(self.dep_path + 'p_model.h5')
            self.p_trafos = np.load(self.dep_path + 'p_trafos.npy', allow_pickle = True)
            self.D_model = tf.keras.models.load_model(self.dep_path + 'D_model.h5')
            self.D_trafos = np.load(self.dep_path + 'D_trafos.npy', allow_pickle = True)
        # Helium
        if ('Helium'in self.init_particles) or ('Helium 3' in self.init_particles) or ('Helium 4' in self.init_particles):
            self.He4_model = tf.keras.models.load_model(self.dep_path + 'He4_model.h5')
            self.He4_trafos = np.load(self.dep_path + 'He4_trafos.npy', allow_pickle = True)
            self.He3_model = tf.keras.models.load_model(self.dep_path + 'He3_model.h5')
            self.He3_trafos = np.load(self.dep_path + 'He3_trafos.npy', allow_pickle = True)

    def predict(self, particle_list, propagation_parameters, DM_mass = None, DM_fs = None, sigma_v = None):
        '''
        Update: Variable number of propagation parameters. Check if number aligns with propagation model.
        Ranges for parameter check updated and added for new models.
        '''
        # check number of parameters is consistent with the propagation model
        self.N_pp = {'DIFF.BRK': 10, 'INJ.BRK': 12}
        if propagation_parameters.shape[-1] != self.N_pp[self.propagation_model]:
            raise ValueError('The number of propagation parameters is not consistent with the propagation model.')
        self.pp = propagation_parameters 
        if DM_mass is not None:
            self.DM_mass = np.log10(DM_mass) + 3 
        else:
            self.DM_mass = DM_mass
        self.DM_fs = DM_fs
        self.sigma_v = sigma_v
        if self.pe:
            self.continue_all, self.continue_DM = self.check_inputs()
        else: 
            self.continue_all = True
            self.continue_DM = True
        Output = []
        if self.pp.ndim == 2:
            self.N = len(propagation_parameters)
            for ptype in particle_list:
                out = self.sim_ptype(ptype)
                Output.append(out)
        elif self.pp.ndim == 1:
            self.N = 1
            self.pp = np.repeat([self.pp], 2, axis = 0)
            if self.DM_mass != None:
                self.DM_mass = np.repeat(np.array([self.DM_mass]), 2, axis = 0)
            if not self.DM_fs is None:
                self.DM_fs = np.repeat([self.DM_fs], 2, axis = 0)
            for ptype in particle_list:
                out = self.sim_ptype(ptype)
                out[0] = out[0][0]
                Output.append(out)
        return Output

    def sim_ptype(self, ptype):
        if ptype == 'Protons':
            if not ptype in self.init_particles:
                raise ValueError('The particle type "%s" has not been initialized.'%ptype)
            out = self.p_sim()
        elif ptype == 'DM Antiprotons':
            if not ptype in self.init_particles:
                raise ValueError('The particle type "%s" has not been initialized.'%ptype)
            if self.DM_mass is None:
                print()
                print('DRN Warning: The particle type "DM Antiprotons" is skipped, because no dark matter mass was given.')
                self.continue_DM = False
            if self.DM_fs is None:
                print()
                print('DRN Warning: The particle type "DM Antiprotons" is skipped, because no dark matter branching fractions were given.')
                self.continue_DM = False
            if self.sigma_v is None:
                print()
                print('DRN Info: No value was given for the annihilation cross section. It will be set to default (<sigma v> = 3 * 10^-26 cm^3 s^-1).')
                sigma_v = 10**(-25.5228)
            out = self.DM_sim()
            out[0] = sigma_v/10**(-25.5228) * out[0]
        elif ptype == 'Secondary Antiprotons':
            if not ptype in self.init_particles:
                raise ValueError('The particle type "%s" has not been initialized.'%ptype)
            out = self.secondary_sim()
        elif ptype == 'Helium 4':
            if not 'Helium' in self.init_particles:
                raise ValueError('The particle type "%s" has not been initialized.'%ptype)
            out = self.He4_sim()
        elif ptype == 'Deuterium':
            if not 'Protons' in self.init_particles:
                raise ValueError('The particle type "%s" has not been initialized.'%ptype)
            out = self.D_sim()
        elif ptype == 'Helium 3':
            if not 'Helium' in self.init_particles:
                raise ValueError('The particle type "%s" has not been initialized.'%ptype)
            out = self.He3_sim()
        else:
            print()
            print('DRN Warning: The parameter type "%s" given in the parameter type is not provided in this tool. It will be skipped.'%ptype)
            out = 0
        return out
    
    def p_sim(self):
        if self.continue_all == False:
            p_flux = np.zeros((self.N,len(self.E_ext)))
        else:
            p_flux = 10**self.p_model.predict((self.pp - np.array(self.p_trafos[0]))/np.array(self.p_trafos[1]))/self.E_ext**2.7
        return [p_flux, self.E_ext]

    def D_sim(self):
        if self.continue_all == False:
            D_flux = np.zeros((self.N,len(self.E_ext)))
        else:
            D_flux = 10**self.D_model.predict((self.pp - np.array(self.D_trafos[0]))/np.array(self.D_trafos[1]))/self.E_ext**2.7
        return [D_flux, self.E_ext]

    def He3_sim(self):
        if self.continue_all == False:
            He3_flux = np.zeros((self.N,len(self.E_ext)))
        else:
            He3_flux = 10**self.He3_model.predict((self.pp - np.array(self.He3_trafos[0]))/np.array(self.He3_trafos[1]))/self.E_ext**2.7
        return [He3_flux, self.E_ext]

    def He4_sim(self):
        if self.continue_all == False:
            He4_flux = np.zeros((self.N,len(self.E_ext)))
        else:
            He4_flux = 10**self.He4_model.predict((self.pp - np.array(self.He4_trafos[0]))/np.array(self.He4_trafos[1]))/self.E_ext**2.7
        return [He4_flux, self.E_ext]

    def secondary_sim(self):
        if self.continue_all == False:
            s_flux = np.zeros((self.N,len(self.E_bins)))
        else:
            propagation_parameters_s = ((self.pp - np.array(self.S_trafos[0]))/np.array(self.S_trafos[1]))
            s_flux = 10**self.S_model.predict(propagation_parameters_s)/self.E_bins**2.7
        return [s_flux, self.E_bins]

    def DM_sim(self):
        def make_prediction_x(prop_param, m, fs, m0):
            min_x = -0.1 # Necessary for model without reacceleration (DM antiproton spectra diverge for E -> m_DM, x -> 0)
            logx_grid = np.linspace(-3.7, min_x, 40)
            x_grid = 10**logx_grid
            E_eval = 10**(m0[:,np.newaxis]-3) * np.repeat([x_grid], len(m0[:,np.newaxis]), axis = 0)
            final_flux = 10**(self.DM_model.predict([m, fs, prop_param])) * 1/(10**(m0[:,np.newaxis]-3))**3 * 1/np.repeat([x_grid], len(m0[:,np.newaxis]), axis = 0)
            DM_flux = np.zeros((len(m), 28))
            for i in range(len(m)):
                E_bins_sub = []
                for e in self.E_all[23:51]:
                    if e/10**(m0[i]-3) >= 10**-3.7 and e/10**(m0[i]-3) <= 1:
                        E_bins_sub.append(e)
                E_bins_sub = np.array(E_bins_sub)
                interp_flux = np.exp(np.interp(np.log(E_bins_sub), np.log(E_eval[i]), np.log(final_flux[i]))) 
                inds = np.arange(np.where(self.E_bins == E_bins_sub[0])[0] , np.where(self.E_bins == E_bins_sub[0])[0] + len(E_bins_sub))
                DM_flux[i, inds] = interp_flux
            return DM_flux
        if self.continue_DM == False or self.continue_all == False:
            DM_flux = np.zeros((self.N,len(self.E_bins)))
        else:
            propagation_parameters_DM = ((self.pp - np.array(self.DM_trafos[0,0]))/np.array(self.DM_trafos[0,1]))
            DM_mass_t = (self.DM_mass - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
            DM_fs = (np.log10(self.DM_fs) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0]))
            DM_flux = np.zeros(len(self.E_bins))
            DM_flux = make_prediction_x(propagation_parameters_DM, DM_mass_t, DM_fs, self.DM_mass)
        return [DM_flux, self.E_bins]

    def check_inputs(self):
        continue_all = True
        continue_DM = True

        if self.DM_mass is not None:
            if np.min(self.DM_mass) < (np.log10(5) + 3) or np.max(self.DM_mass) > np.log10(5000) + 3:
                print()
                print('DRN Warning: The particle type "DM Antiprotons" is skipped. At least one of the given DM masses is outside of the provided range (5 GeV to 5 TeV).')
                continue_DM = False
            if np.min(self.DM_fs) < 1e-5 or np.max(self.DM_fs) > 1 or not np.allclose(np.sum(self.DM_fs, axis = -1), np.ones_like(np.sum(self.DM_fs, axis = -1))):
                new_fs = self.norm_scale_fractions(self.DM_fs)
                self.DM_fs = new_fs
                print(new_fs)
                print()
                print('DRN Info: The selected branching fractions were not in the range of trained parameters or not normalized to one. Values below 1e-5 were mapped to 1e-5 and the remaining fractions normalized accordingly.')

        strings = {'DIFF.BRK' : ['gamma_2,p', 'gamma_2', 'D0', 'delta_l', 'delta', 'delta_h - delta', 'R_D,0', 's_D', 'R_D,1', 'v_0,c'],
                   'INJ.BRK': ['gamma_1,p', 'gamma_1', 'R_0', 's', 'gamma_2,p', 'gamma_2', 'D_0', 'delta', 'delta_h - delta', 'R_1,D', 'v_0', 'v_A']} 
        mins_pp = {'DIFF.BRK' : [2.249, 2.194, 3.411e+28, -9.66635e-01, 4.794e-01, -2.000e-01, 3.044e+03, 3.127e-01, 1.217e+0, -1e-5],
                   'INJ.BRK': [1.59786, 1.60102, 4939.44, 0.221776, 2.41369, 2.36049,3.53e+28, 0.301255, -0.171395, 125612, -1e-5, 14.3322]}
        maxs_pp = {'DIFF.BRK' : [2.37e+00, 2.314e+00, 4.454e+28, -3.677e-01, 6.048e-01, -8.330e-02, 4.928e+03, 5.142e-01, 3.154e+05, 1.447e+01],
                   'INJ.BRK': [1.84643, 1.84721, 8765.77, 0.45543, 2.49947, 2.44248, 5.49E+28, 0.41704, -0.0398135, 413544, 8.61201, 29.206]}
        for i in range(self.N_pp[self.propagation_model]):
            if self.pp.ndim == 2:
                if np.min(self.pp[:, i]) <= mins_pp[self.propagation_model][i] or np.max(self.pp[:, i]) >= maxs_pp[self.propagation_model][i]:
                    print()
                    print('DRN Warning: At least one of the inputs for %s is outside of the trained parameter ranges. No output will be given. '%strings[self.propagation_model][i])
                    continue_all = False
            else: 
                if (self.pp[i] <= mins_pp[self.propagation_model][i]) or  (self.pp[i] >= maxs_pp[self.propagation_model][i]):
                    print()
                    print('DRN Warning: At least one of the inputs for %s is outside of the trained parameter ranges. No output will be given. '%strings[self.propagation_model][i])
                    print('Min:', mins_pp[self.propagation_model][i])
                    print('Max:', maxs_pp[self.propagation_model][i])
                    print('Input:', self.pp[i])
                    continue_all = False
        return continue_all, continue_DM

    def create_INJ_BRK_parameters(self, gamma_1p = 1.72, gamma_1 = 1.73, R_0 = 6.43e3, s = 0.33, gamma_2p = 2.45, gamma_2 = 2.39, D_0 = 4.1e28, delta = 0.372, delta_h_delta = -0.09, R_1D = 2.34e5, v_0c = 0.64, v_A = 20.4, N_identical = 1):
        '''
        Only useful for 'INJ.BRK' setup.
        '''
        propagation_parameters = np.array([gamma_1p, gamma_1, R_0, s, gamma_2p, gamma_2, D_0, delta, delta_h_delta, R_1D, v_0c, v_A])
        if N_identical > 1:
            propagation_parameters = np.repeat(propagation_parameters[np.newaxis], N_identical, axis = 0)
        return propagation_parameters

    def create_DIFF_BRK_parameters(self, gamma_2p = 2.34, gamma_2 = 2.28, D0 = 3.78e28, delta_l = -0.66, delta = 0.52, delta_h_delta = -0.16, R_D0 = 3910, s_D = 0.41, R_D1 = 2.22e5, v_0c = 1.91, N_identical = 1):
        '''
        Only useful for 'DIFF.BRK' setup.
        '''
        propagation_parameters = np.array([gamma_2p, gamma_2, D0, delta_l, delta, delta_h_delta, R_D0, s_D, R_D1, v_0c])
        if N_identical > 1:
            propagation_parameters = np.repeat(propagation_parameters[np.newaxis], N_identical, axis = 0)
        return propagation_parameters

    def norm_scale_fractions(self, fs):
        if fs.ndim > 1:
            rf = fs/np.sum(fs, axis = -1)[:,None] # initial normalization
            masked_array = np.where(rf < 1e-5, 0, 1) # ones for every fs >= 1e-5
            masked_reversed = np.ones_like(masked_array) - masked_array # ones for every fs < 1e-5
            masked_rf = masked_array * rf # array with entries only >= 1e-5, else 0
            scaling = (1-np.sum(masked_reversed, axis = -1)*1e-5)/np.sum(masked_rf, axis = -1) # scaling for each >=1e-5 fs, while keeping relative fractions and normalizations
            new_fs = masked_rf * scaling[:,None] + masked_reversed*1e-5 # scale fs >=1e-5 and set other to 1e-5
        else:
            rf = fs/np.sum(fs, axis = -1) # initial normalization
            masked_array = np.where(rf < 1e-5, 0, 1) # ones for every fs >= 1e-5
            masked_reversed = np.ones_like(masked_array) - masked_array # ones for every fs < 1e-5
            masked_rf = masked_array * rf # array with entries only >= 1e-5, else 0
            scaling = (1-np.sum(masked_reversed, axis = -1)*1e-5)/np.sum(masked_rf, axis = -1) # scaling for each >=1e-5 fs, while keeping relative fractions and normalizations
            new_fs = masked_rf * scaling + masked_reversed*1e-5 # scale fs >=1e-5 and set other to 1e-5
        return new_fs