
from preamble_DRN import *

class DRN:
    def __init__(self, prevent_extrapolation = True):
        self.pe = prevent_extrapolation
        self.load_deps()
        print('The simulation tool has been initiated.')
        print()
        
    def load_deps(self):
        # Energy bins
        self.E_bins = np.load(c_path + '/dependencies/E.npy')
        self.E_all = np.load(c_path + '/dependencies/E_all.npy')
        self.E_ext = np.load(c_path + '/dependencies/E_ext.npy')
        # DM antiprotons
        self.DM_model = tf.keras.models.load_model(c_path + '/dependencies/DM_model_x.h5')
        self.DM_trafos = np.load(c_path + '/dependencies/DM_trafos_x.npy', allow_pickle = True)
        # Secondary antiprotons
        self.S_model = tf.keras.models.load_model(c_path + '/dependencies/S_model.h5')
        self.S_trafos = np.load(c_path + '/dependencies/S_trafos.npy', allow_pickle = True)
        # Protons
        self.p_model = tf.keras.models.load_model(c_path + '/dependencies/p_model.h5')
        self.p_trafos = np.load(c_path + '/dependencies/p_trafos.npy', allow_pickle = True)
        self.D_model = tf.keras.models.load_model(c_path + '/dependencies/D_model.h5')
        self.D_trafos = np.load(c_path + '/dependencies/D_trafos.npy', allow_pickle = True)
        # Helium
        self.He4_model = tf.keras.models.load_model(c_path + '/dependencies/He4_model.h5')
        self.He4_trafos = np.load(c_path + '/dependencies/He4_trafos.npy', allow_pickle = True)
        self.He3_model = tf.keras.models.load_model(c_path + '/dependencies/He3_model.h5')
        self.He3_trafos = np.load(c_path + '/dependencies/He3_trafos.npy', allow_pickle = True)

    def predict(self, particle_list, propagation_parameters, DM_mass = None, DM_fs = None, sigma_v = None):
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
        if self.continue_all == False:
            for ptype in particle_list:
                Output.append([[None], [None]])
            return Output
        else:
            if self.pp.ndim == 2:
                for ptype in particle_list:
                    out = self.sim_ptype(ptype)
                    Output.append(out)
            elif self.pp.ndim == 1:
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
            if self.continue_all:
                out = self.p_sim()
        elif ptype == 'DM Antiprotons':
            if self.DM_mass is None:
                print('The particle type "DM Antiprotons" is skipped, because no dark matter mass was given.')
                self.continue_DM = False
            if self.DM_fs is None:
                print('The particle type "DM Antiprotons" is skipped, because no dark matter branching fractions were given.')
                self.continue_DM = False
            if self.sigma_v is None:
                print('No value was given for the annihilation cross section. It will be set to default (<sigma v> = 3 * 10^-26 cm^3 s^-1).')
                sigma_v = 10**(-25.5228)
            out = self.DM_sim()
            out[0] = sigma_v/10**(-25.5228) * out[0]
        elif ptype == 'Secondary Antiprotons':
            out = self.secondary_sim()
        elif ptype == 'Helium 4':
            out = self.He4_sim()
        elif ptype == 'Deuterium':
            out = self.D_sim()
        elif ptype == 'Helium 3':
            out = self.He3_sim()
        else:
            print('The parameter type "%s" given in the parameter type is not provided in this tool. It will be skipped.'%ptype)
            out = 0
        return out
    
    def p_sim(self):
        p_flux = 10**self.p_model.predict((self.pp - np.array(self.p_trafos[0])[:11])/np.array(self.p_trafos[1])[:11])/self.E_ext**2.7
        return [p_flux, self.E_ext]

    def D_sim(self):
        D_flux = 10**self.D_model.predict((self.pp - np.array(self.D_trafos[0])[:11])/np.array(self.D_trafos[1])[:11])/self.E_ext**2.7
        return [D_flux, self.E_ext]

    def He3_sim(self):
        He3_flux = 10**self.He3_model.predict((self.pp - np.array(self.He3_trafos[0])[:11])/np.array(self.He3_trafos[1])[:11])/self.E_ext**2.7
        return [He3_flux, self.E_ext]

    def He4_sim(self):
        He4_flux = 10**self.He4_model.predict((self.pp - np.array(self.He4_trafos[0])[:11])/np.array(self.He4_trafos[1])[:11])/self.E_ext**2.7
        return [He4_flux, self.E_ext]

    def secondary_sim(self):
        propagation_parameters_s = ((self.pp - np.array(self.S_trafos[0])[:11])/np.array(self.S_trafos[1])[:11])
        s_flux = 10**self.S_model.predict(propagation_parameters_s)/self.E_bins**2.7
        return [s_flux, self.E_bins]

    def DM_sim(self):
        def make_prediction_x(prop_param, m, fs, m0):
            logx_grid = np.linspace(-3.7, 0, 40)
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
            return [np.zeros((len(self.DM_mass), len(self.E_bins))), self.E_bins]
        else:
            propagation_parameters_DM = ((self.pp - np.array(self.DM_trafos[0,0])[:11])/np.array(self.DM_trafos[0,1])[:11])
            DM_mass_t = (self.DM_mass - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
            DM_fs = (np.log10(self.DM_fs) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0]))
            DM_flux = np.zeros(len(self.E_bins))
            DM_flux = make_prediction_x(propagation_parameters_DM, DM_mass_t, DM_fs, self.DM_mass)
            return [DM_flux, self.E_bins]

    def check_inputs(self):
        continue_all = True
        continue_DM = True

        mins_pp = [1.63, 1.6, 2.38, 2.34, 5000, 0.32, 1.15e28, 0.375, 0, 0, 2]
        maxs_pp = [1.9, 1.88, 2.45, 2.4, 9500, 0.5, 5.2e28, 0.446, 11, 12.8, 7]

        if self.DM_mass is not None:
            if np.min(self.DM_mass) < (np.log10(5) + 3) or np.max(self.DM_mass) > np.log10(5000) + 3:
                print('The particle type "DM Antiprotons" is skipped. At least one of the given DM masses is outside of the provded range (5 GeV to 5 GeV).')
                continue_DM = False
            if np.min(self.DM_fs) < 1e-5 or np.max(self.DM_fs) > 1:
                print('The particle type "DM Antiprotons" is skipped. Branching fractions have to be in the range 1e-5 to 1.')
                continue_DM = False
        # elif self.DM_mass is not None:
        #     if self.DM_mass < 5 or self.DM_mass > 5000:
        #         print('The particle type "DM Antiprotons" is skipped. The given DM masses is outside of the provded range (5 GeV to 5 GeV).')
        #         continue_DM = False
        strings = ['gamma 1,p', 'gamma 1', 'gamma 2,p', 'gamma 2', 'R_0', 's_0', 'D_0', 'delta', 'v_Alfven', 'v_0,c', 'z_h']
        for i in range(11):
            if self.pp.ndim == 2:
                if np.min(self.pp[:, i]) <= mins_pp[i] or np.max(self.pp[:, i]) >= maxs_pp[i]:
                    print('A least one of the inputs for %s is outside of the trained parameter ranges. No output will be given. '%strings[i])
                    continue_all = False
            else: 
                if (self.pp[i] <= mins_pp[i]) or  (self.pp[i] >= maxs_pp[i]):
                    print('A least on of the inputs for %s is outside of the trained parameter ranges. No output will be given. '%strings[i])
                    continue_all = False
        return continue_all, continue_DM

