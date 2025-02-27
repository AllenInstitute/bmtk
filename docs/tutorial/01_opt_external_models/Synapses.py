from neuron import h

class Synapse_py3:
    def __init__(self,source,target,section,weight = 1):
		
        self.input = h.NetStim(0.5)
        self.input.start = -10
        self.input.number = 1
        self.input.interval = 1e9
        self.weight = weight


        self.postsyns = {}

        if (type(source) == type('s')):
            sourcetype = source
           
                
#Purkinje cell
        if sourcetype == 'pf':
            if target.whatami == 'prk':
                # Make a PF synapse onto a purkinje cell
                # Use deterministic synapses*
                self.whatami = "syn_pf2prk_det"
                self.postsyns['AMPA'] = [h.PF_syn(0.5, sec=section)]
                self.postsyns['AMPA'][0].tau_facil=10.8*5#
                self.postsyns['AMPA'][0].tau_rec=35.1 
                self.postsyns['AMPA'][0].tau_1=3*2
                self.postsyns['AMPA'][0].gmax = 2800
                self.postsyns['AMPA'][0].U=0.13

                self.nc_syn = [h.NetCon(self.input,receptor[0],0,0.1,1) for receptor in self.postsyns.values()]

        elif sourcetype == 'aa':
            if target.whatami == 'prk':
                # Make a ascending axon synapse onto a purkinje cell
                # Use deterministic synapses
                self.whatami = "syn_aa2prk_det"
                self.postsyns['AMPA'] = [h.PF_syn(0.5, sec=section)]
                self.postsyns['AMPA'][0].tau_facil=10.8*5
                self.postsyns['AMPA'][0].tau_rec=35.1*1
                self.postsyns['AMPA'][0].tau_1=3*5
                self.postsyns['AMPA'][0].gmax = 2800
                self.postsyns['AMPA'][0].U=0.13

                self.nc_syn = [h.NetCon(self.input,receptor[0],0,0.1,1) for receptor in self.postsyns.values()]
            
        elif sourcetype == 'stl':
            if target.whatami == 'prk':
                self.whatami = "syn_stl2prk_alpha1"
                self.postsyns['GABA'] = [h.PC_gaba_alpha1(0.5, sec=section)] # self.postsyns
                self.postsyns['GABA'][0].tau_facil=4
                self.postsyns['GABA'][0].tau_rec=15
                self.postsyns['GABA'][0].tau_1=1
                self.postsyns['GABA'][0].Erev = -60
                self.postsyns['GABA'][0].gmaxA1 = 2600
                self.postsyns['GABA'][0].U=0.35
                self.nc_syn = [h.NetCon(self.input,receptor[0],0,0.1,1) for receptor in self.postsyns.values()] 	
                
        else:
            print('SOURCE TYPE DOES NOT EXIST SOMETHING WRONG!!!!!!!!!')
            
            
