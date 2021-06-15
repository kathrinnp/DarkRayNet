from preamble_DRN import *

class antiproton_sim:
    def __init__(self, x = True):
        self.x = x
        self.load_deps()
        print('The antiproton simulation tool has been initiated.')
        
    def load_deps(self):
        if self.x == True:
            self.DM_model = tf.keras.models.load_model(c_path + '/dependencies/DM_model_x.h5')
            self.DM_trafos = np.load(c_path + '/dependencies/DM_trafos_x.npy', allow_pickle = True)
            self.E_all = np.load(c_path + '/dependencies/E_all.npy')
        else:
            self.DM_model = tf.keras.models.load_model(c_path + '/dependencies/DM_model.h5')
            self.DM_trafos = np.load(c_path + '/dependencies/DM_trafos.npy', allow_pickle = True)
        #
        self.S_model = tf.keras.models.load_model(c_path + '/dependencies/S_model.h5')
        self.S_trafos = np.load(c_path + '/dependencies/S_trafos.npy', allow_pickle = True)
        self.E_bins = np.load(c_path + '/dependencies/E.npy')

    def single_sim(self, propagation_parameters, DM_mass, DM_fs):
        if self.x == False:
            propagation_parameters_DM = ((propagation_parameters - np.array(self.DM_trafos[0,0])[:11])/np.array(self.DM_trafos[0,1])[:11])
            DM_mass = (DM_mass - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
            DM_fs = (np.log10(DM_fs) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0]))
            propagation_parameters_s = ((propagation_parameters - np.array(self.S_trafos[0])[:11])/np.array(self.S_trafos[1])[:11])
            s_flux = 10**self.S_model.predict(np.repeat([propagation_parameters_s], 2, axis = 0))[0]/self.E_bins**2.7
            DM_flux = 10**self.DM_model.predict([np.repeat([DM_mass],2,axis = 0), np.repeat([DM_fs],2,axis = 0), np.repeat([propagation_parameters_DM], 2, axis = 0)])[0]/self.E_bins**2.7
            total_flux = s_flux + DM_flux
            return total_flux, DM_flux, s_flux, self.E_bins
        else: 
            def make_prediction(prop_param, m, fs, m0):
                logx_grid = np.linspace(-3.7, 0, 40)
                x_grid = 10**logx_grid
                E_eval = 10**(m0-3) * x_grid        
                final_flux = 10**(self.DM_model.predict([m, fs, prop_param])[0]) * 1/(10**(m0-3))**3 * 1/x_grid
                E_bins_sub = []
                for e in self.E_all[23:51]:
                    if e/10**(m0-3) >= 10**-3.7 and e/10**(m0-3) <= 1:
                        E_bins_sub.append(e)
                E_bins_sub = np.array(E_bins_sub)
                final_flux = np.exp(np.interp(np.log(E_bins_sub), np.log(E_eval), np.log(final_flux))) 
                return final_flux, E_bins_sub
            propagation_parameters_DM = ((propagation_parameters - np.array(self.DM_trafos[0,0]))/np.array(self.DM_trafos[0 ,1]))
            DM_mass = (DM_mass_0 - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
            DM_fs = (np.log10(DM_fs) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0]))
            propagation_parameters_s = ((propagation_parameters - np.array(self.S_trafos[0])[:11])/np.array(self.S_trafos[1])[:11])
            s_flux = 10**self.S_model.predict(propagation_parameters_s)/self.E_bins**2.7
            DM_flux = np.zeros(s_flux.shape)
            for i in range(len(DM_mass)):
                DM_flux_sub, E_bins_sub = make_prediction(propagation_parameters_DM[i:i+1, :11], DM_mass[i:i+1], DM_fs[i:i+1], DM_mass_0[i])
                inds = np.arange(np.where(self.E_bins == E_bins_sub[0])[0] , np.where(self.E_bins == E_bins_sub[0])[0] + len(E_bins_sub))
                DM_flux[i, inds] = DM_flux_sub
            total_flux = s_flux + DM_flux
            return total_flux, DM_flux, s_flux, self.E_bins

    def N_sim(self, propagation_parameters, DM_mass, DM_fs):
        if self.x == False:
            propagation_parameters_DM = ((propagation_parameters - np.array(self.DM_trafos[0,0]))/np.array(self.DM_trafos[0 ,1]))
            DM_mass = (DM_mass - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
            DM_fs = (np.log10(DM_fs) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0]))
            DM_flux = 10**self.DM_model.predict([DM_mass, DM_fs, propagation_parameters_DM])/self.E_bins**2.7
        else:
            def make_prediction_x(prop_param, m, fs, E_bins, loaded_model, m0):
                #
                # model must already be loaded
                #
                logx_grid = np.linspace(-3.7, 0, 40)
                x_grid = 10**logx_grid
                E_eval = 10**(m0[:,np.newaxis]-3) * np.repeat([x_grid], len(m0), axis = 0)
                final_flux = 10**(loaded_model.predict([m, fs, prop_param])) * 1/(10**(m0[:,np.newaxis]-3))**3 * 1/np.repeat([x_grid], len(m0), axis = 0)
                DM_flux = np.zeros((len(m), 28))
                for i in range(len(m)):
                    E_bins_sub = []
                    for e in self.E_all[23:51]:
                        if e/10**(m0[i]-3) >= 10**-3.7 and e/10**(m0[i]-3) <= 1:
                            E_bins_sub.append(e)
                    E_bins_sub = np.array(E_bins_sub)
                    interp_flux = np.exp(np.interp(np.log(E_bins_sub), np.log(E_eval[i]), np.log(final_flux[i]))) 
                    inds = np.arange(np.where(E_bins == E_bins_sub[0])[0] , np.where(E_bins == E_bins_sub[0])[0] + len(E_bins_sub))
                    if len(inds) != len(interp_flux):
                        print(m0, interp_flux, E_bins_sub)
                    DM_flux[i, inds] = interp_flux
                return DM_flux
            propagation_parameters_DM = ((propagation_parameters - np.array(self.DM_trafos[0,0]))/np.array(self.DM_trafos[0 ,1]))
            DM_mass_scaled = (DM_mass - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
            DM_fs = (np.log10(DM_fs) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0]))
            DM_flux = make_prediction_x(propagation_parameters_DM, DM_mass_scaled, DM_fs, self.E_bins, self.DM_model, DM_mass)
        propagation_parameters_s = ((propagation_parameters - np.array(self.S_trafos[0])[:11])/np.array(self.S_trafos[1])[:11])
        s_flux = 10**self.S_model.predict(propagation_parameters_s)/self.E_bins**2.7
        total_flux = s_flux + DM_flux
        return total_flux, DM_flux, s_flux, self.E_bins

    # def x_sim(self, propagation_parameters, DM_mass_0, DM_fs):
    #     def make_prediction(prop_param, m, fs, m0):
    #         logx_grid = np.linspace(-3.7, 0, 40)
    #         x_grid = 10**logx_grid
    #         E_eval = 10**(m0-3) * x_grid        
    #         final_flux = 10**(self.DM_model.predict([m, fs, prop_param])[0]) * 1/(10**(m0-3))**3 * 1/x_grid
    #         E_bins_sub = []
    #         for e in self.E_all[23:51]:
    #             if e/10**(m0-3) >= 10**-3.7 and e/10**(m0-3) <= 1:
    #                 E_bins_sub.append(e)
    #         E_bins_sub = np.array(E_bins_sub)
    #         final_flux = np.exp(np.interp(np.log(E_bins_sub), np.log(E_eval), np.log(final_flux))) 
    #         return final_flux, E_bins_sub
    #     propagation_parameters_DM = ((propagation_parameters - np.array(self.DM_trafos[0,0]))/np.array(self.DM_trafos[0 ,1]))
    #     DM_mass = (DM_mass_0 - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
    #     DM_fs = (np.log10(DM_fs) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0]))
    #     propagation_parameters_s = ((propagation_parameters - np.array(self.S_trafos[0])[:11])/np.array(self.S_trafos[1])[:11])
    #     s_flux = 10**self.S_model.predict(propagation_parameters_s)/self.E_bins**2.7
    #     DM_flux = np.zeros(s_flux.shape)
    #     for i in range(len(DM_mass)):
    #         DM_flux_sub, E_bins_sub = make_prediction(propagation_parameters_DM[i:i+1, :11], DM_mass[i:i+1], DM_fs[i:i+1], DM_mass_0[i])
    #         inds = np.arange(np.where(self.E_bins == E_bins_sub[0])[0] , np.where(self.E_bins == E_bins_sub[0])[0] + len(E_bins_sub))
    #         DM_flux[i, inds] = DM_flux_sub
    #     total_flux = s_flux + DM_flux
    #     return total_flux, DM_flux, s_flux, self.E_bins

class primary_sim:
    def __init__(self):
        self.load_deps()
        print('The primary (p,He) simulation tool has been initiated.')
        
    def load_deps(self):
        self.p_model = tf.keras.models.load_model(c_path + '/dependencies/p_model.h5')
        self.p_trafos = np.load(c_path + '/dependencies/p_trafos.npy', allow_pickle = True)
        self.He4_model = tf.keras.models.load_model(c_path + '/dependencies/He4_model.h5')
        self.He4_trafos = np.load(c_path + '/dependencies/He4_trafos.npy', allow_pickle = True)
        self.D_model = tf.keras.models.load_model(c_path + '/dependencies/D_model.h5')
        self.D_trafos = np.load(c_path + '/dependencies/D_trafos.npy', allow_pickle = True)
        self.He3_model = tf.keras.models.load_model(c_path + '/dependencies/He3_model.h5')
        self.He3_trafos = np.load(c_path + '/dependencies/He3_trafos.npy', allow_pickle = True)
        self.E_bins = np.load(c_path + '/dependencies/E_ext.npy')

    def single_sim(self, propagation_parameters):
        p_flux = 10**self.p_model.predict(np.repeat([((propagation_parameters - np.array(self.p_trafos[0])[:11])/np.array(self.p_trafos[1])[:11])], 2, axis = 0))[0]/self.E_bins**2.7
        He3_flux = 10**self.He3_model.predict(np.repeat([((propagation_parameters - np.array(self.He3_trafos[0])[:11])/np.array(self.He3_trafos[1])[:11])], 2, axis = 0))[0]/self.E_bins**2.7
        D_flux = 10**self.D_model.predict(np.repeat([((propagation_parameters - np.array(self.D_trafos[0])[:11])/np.array(self.D_trafos[1])[:11])], 2, axis = 0))[0]/self.E_bins**2.7
        He4_flux = 10**self.He4_model.predict(np.repeat([((propagation_parameters - np.array(self.He4_trafos[0])[:11])/np.array(self.He4_trafos[1])[:11])], 2, axis = 0))[0]/self.E_bins**2.7
        return p_flux, D_flux, He4_flux, He3_flux, self.E_bins

    def N_sim(self, propagation_parameters):
        p_flux = 10**self.p_model.predict((propagation_parameters - np.array(self.p_trafos[0])[:11])/np.array(self.p_trafos[1])[:11])/self.E_bins**2.7
        He3_flux = 10**self.He3_model.predict((propagation_parameters - np.array(self.He3_trafos[0])[:11])/np.array(self.He3_trafos[1])[:11])/self.E_bins**2.7
        D_flux = 10**self.D_model.predict((propagation_parameters - np.array(self.D_trafos[0])[:11])/np.array(self.D_trafos[1])[:11])/self.E_bins**2.7
        He4_flux = 10**self.He4_model.predict((propagation_parameters - np.array(self.He4_trafos[0])[:11])/np.array(self.He4_trafos[1])[:11])/self.E_bins**2.7
        return p_flux, D_flux, He4_flux, He3_flux, self.E_bins


class interface_sim:
    def __init__(self):
        self.load_deps()
        print('The simulation tool has been initiated.')
        
    def load_deps(self):
        self.DM_model = tf.keras.models.load_model(c_path + '/dependencies/DM_model_x.h5')
        self.DM_trafos = np.load(c_path + '/dependencies/DM_trafos_x.npy', allow_pickle = True)
        self.S_model = tf.keras.models.load_model(c_path + '/dependencies/S_model.h5')
        self.S_trafos = np.load(c_path + '/dependencies/S_trafos.npy', allow_pickle = True)
        self.p_model = tf.keras.models.load_model(c_path + '/dependencies/p_model.h5')
        self.p_trafos = np.load(c_path + '/dependencies/p_trafos.npy', allow_pickle = True)
        self.He4_model = tf.keras.models.load_model(c_path + '/dependencies/He4_model.h5')
        self.He4_trafos = np.load(c_path + '/dependencies/He4_trafos.npy', allow_pickle = True)
        self.D_model = tf.keras.models.load_model(c_path + '/dependencies/D_model.h5')
        self.D_trafos = np.load(c_path + '/dependencies/D_trafos.npy', allow_pickle = True)
        self.He3_model = tf.keras.models.load_model(c_path + '/dependencies/He3_model.h5')
        self.He3_trafos = np.load(c_path + '/dependencies/He3_trafos.npy', allow_pickle = True)
        self.E_bins = np.load(c_path + '/dependencies/E.npy')
        self.E_bins_ext = np.load(c_path + '/dependencies/E_ext.npy')
        self.E_all = np.load(c_path + '/dependencies/E_all.npy')

    def p_sim(self, propagation_parameters):
        p_flux = 10**self.p_model.predict(np.repeat([((propagation_parameters - np.array(self.p_trafos[0])[:11])/np.array(self.p_trafos[1])[:11])], 2, axis = 0))[0]/self.E_bins_ext**2.7
        return p_flux, self.E_bins_ext

    def D_sim(self, propagation_parameters):
        D_flux = 10**self.D_model.predict(np.repeat([((propagation_parameters - np.array(self.D_trafos[0])[:11])/np.array(self.D_trafos[1])[:11])], 2, axis = 0))[0]/self.E_bins_ext**2.7
        return D_flux, self.E_bins_ext

    def He3_sim(self, propagation_parameters):
        He3_flux = 10**self.He3_model.predict(np.repeat([((propagation_parameters - np.array(self.He3_trafos[0])[:11])/np.array(self.He3_trafos[1])[:11])], 2, axis = 0))[0]/self.E_bins_ext**2.7
        return He3_flux, self.E_bins_ext

    def He4_sim(self, propagation_parameters):
        He4_flux = 10**self.He4_model.predict(np.repeat([((propagation_parameters - np.array(self.He4_trafos[0])[:11])/np.array(self.He4_trafos[1])[:11])], 2, axis = 0))[0]/self.E_bins_ext**2.7
        return He4_flux, self.E_bins_ext

    def secondary_sim(self, propagation_parameters):
        propagation_parameters_s = ((propagation_parameters - np.array(self.S_trafos[0])[:11])/np.array(self.S_trafos[1])[:11])
        s_flux = 10**self.S_model.predict(np.repeat([propagation_parameters_s], 2, axis = 0))[0]/self.E_bins**2.7
        return s_flux, self.E_bins

    def DM_sim(self, propagation_parameters, DM_mass_0, DM_fs):
        def make_prediction(prop_param, m, fs, m0):
            logx_grid = np.linspace(-3.7, 0, 40)
            x_grid = 10**logx_grid
            E_eval = 10**(m0-3) * x_grid        
            final_flux = 10**(self.DM_model.predict([m, fs, prop_param])[0]) * 1/(10**(m0-3))**3 * 1/x_grid
            E_bins_sub = []
            for e in self.E_all[23:51]:
                if e/10**(m0-3) >= 10**-3.7 and e/10**(m0-3) <= 1:
                    E_bins_sub.append(e)
            E_bins_sub = np.array(E_bins_sub)
            final_flux = np.exp(np.interp(np.log(E_bins_sub), np.log(E_eval), np.log(final_flux))) 
            return final_flux, E_bins_sub
        propagation_parameters_DM = ((propagation_parameters - np.array(self.DM_trafos[0,0])[:11])/np.array(self.DM_trafos[0,1])[:11])
        DM_mass = (DM_mass_0 - np.log10(5e3)) / (np.log10(5e6) - np.log10(5e3))
        DM_fs = (np.log10(DM_fs) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0]))
        #DM_flux = 10**self.DM_model.predict([np.repeat([DM_mass],2,axis = 0), np.repeat([DM_fs],2,axis = 0), np.repeat([propagation_parameters_DM], 2, axis = 0)])[0]/self.E_bins**2.7
        DM_flux = np.zeros(len(self.E_bins))
        DM_flux_sub, E_bins_sub = make_prediction(np.repeat([propagation_parameters_DM], 2, axis = 0), np.repeat([DM_mass],2,axis = 0), np.repeat([DM_fs],2,axis = 0), DM_mass_0)
        inds = np.arange(np.where(self.E_bins == E_bins_sub[0])[0] , np.where(self.E_bins == E_bins_sub[0])[0] + len(E_bins_sub))
        DM_flux[inds] = DM_flux_sub
        return DM_flux, self.E_bins
