from preamble_DRN import *

class antiproton_sim:
    def __init__(self):
        self.load_deps()
        print('The antiproton simulation tool has been initiated.')
        
    def load_deps(self):
        self.DM_model = tf.keras.models.load_model(c_path + '/dependencies/DM_model.h5')
        self.S_model = tf.keras.models.load_model(c_path + '/dependencies/S_model.h5')
        self.DM_trafos = np.load(c_path + '/dependencies/DM_trafos.npy', allow_pickle = True)
        self.S_trafos = np.load(c_path + '/dependencies/S_trafos.npy', allow_pickle = True)
        self.E_bins = np.load(c_path + '/dependencies/E.npy')

    def single_sim(self, propagation_parameters, DM_mass, DM_fs):
        propagation_parameters_DM = ((propagation_parameters - np.array(self.DM_trafos[0,1])[:11])/np.array(self.DM_trafos[1,1])[:11])
        DM_mass = (DM_mass - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
        DM_fs = (np.log10(DM_fs) - np.array(self.DM_trafos[2,1])) / (np.array(self.DM_trafos[3,1])- np.array(self.DM_trafos[2,1]))
        propagation_parameters_s = ((propagation_parameters - np.array(self.S_trafos[0])[:11])/np.array(self.S_trafos[1])[:11])
        s_flux = 10**self.S_model.predict(np.repeat([propagation_parameters_s], 2, axis = 0))[0]/self.E_bins**2.7
        DM_flux = 10**self.DM_model.predict([np.repeat([DM_mass],2,axis = 0), np.repeat([DM_fs],2,axis = 0), np.repeat([propagation_parameters_DM], 2, axis = 0)])[0]/self.E_bins**2.7
        total_flux = s_flux + DM_flux
        return total_flux, DM_flux, s_flux, self.E_bins

    def N_sim(self, propagation_parameters, DM_mass, DM_fs):
        propagation_parameters_DM = ((propagation_parameters - np.array(self.DM_trafos[0,1])[:11])/np.array(self.DM_trafos[1,1])[:11])
        DM_mass = (DM_mass - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
        DM_fs = (np.log10(DM_fs) - np.array(self.DM_trafos[2,1])) / (np.array(self.DM_trafos[3,1])- np.array(self.DM_trafos[2,1]))
        propagation_parameters_s = ((propagation_parameters - np.array(self.S_trafos[0])[:11])/np.array(self.S_trafos[1])[:11])
        s_flux = 10**self.S_model.predict(propagation_parameters_s)/self.E_bins**2.7
        DM_flux = 10**self.DM_model.predict([DM_mass, DM_fs, propagation_parameters_DM])/self.E_bins**2.7
        total_flux = s_flux + DM_flux
        return total_flux, DM_flux, s_flux, self.E_bins

class primary_sim:
    def __init__(self):
        self.load_deps()
        print('The primary (p,He) simulation tool has been initiated.')
        
    def load_deps(self):
        self.p_model = tf.keras.models.load_model(c_path + '/dependencies/p_model.h5')
        self.He_model = tf.keras.models.load_model(c_path + '/dependencies/He_model.h5')
        self.p_He_trafos = np.load(c_path + '/dependencies/p_He_trafos.npy', allow_pickle = True)
        self.R_p = np.load(c_path + '/dependencies/R_p.npy')
        self.R_He4 = np.load(c_path + '/dependencies/R_He4.npy')

    def single_sim(self, propagation_parameters):
        propagation_parameters = ((propagation_parameters - np.array(self.p_He_trafos[0])[:11])/np.array(self.p_He_trafos[1])[:11])
        p_flux = 10**self.p_model.predict(np.repeat([propagation_parameters], 2, axis = 0))[0]/self.R_p**2.7
        He_flux = 10**self.He_model.predict(np.repeat([propagation_parameters], 2, axis = 0))[0]/self.R_He4**2.7
        return p_flux, self.R_p, He_flux, self.R_He4

    def N_sim(self, propagation_parameters):
        propagation_parameters = ((propagation_parameters - np.array(self.p_He_trafos[0])[:11])/np.array(self.p_He_trafos[1])[:11])
        p_flux = 10**self.p_model.predict(propagation_parameters)/self.R_p**2.7
        He_flux = 10**self.He_model.predict(propagation_parameters)/self.R_He4**2.7
        return p_flux, self.R_p, He_flux, self.R_He4
