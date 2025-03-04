from neuron import h
import numpy as np
import random
import math
import sys
from random import choice

from Purkinje_morpho_1_number import number_ind_1

from Synapses import Synapse_py3

class Purkinje_Morpho_1:
    def __init__(self,spines_on = 0):
        
        h.load_file('stdlib.hoc')
        h.load_file('import3d.hoc')
        
        cell = h.Import3d_Neurolucida3()
        cell.input('morphology/soma_10c.asc')
            
        
        i3d = h.Import3d_GUI(cell,0)
        i3d.instantiate(self)
        
        #the file contains all the results for the population. Some are good, some are bad, some are ugly.
        conductvalues = np.genfromtxt("R_01_final_pop.txt")
        indiv_number = number_ind_1['indiv']
        print( 'indiv number', indiv_number)
        
        
        self.x = self.y = self.z = 0
        def set_position(x, y, z):
            for sec in self.all:
                for i in range(sec.n3d()):
                    sec.pt3dchange(i,
                                x - self.x + sec.x3d(i),
                                y - self.y + sec.y3d(i),
                                z - self.z + sec.z3d(i),
                                sec.diam3d(i))
            self.x, self.y, self.z = x, y, z
        
        self.soma[0].nseg = 1 + (2 * int(self.soma[0].L / 40))
        self.soma[0].cm = 2
        self.soma[0].Ra = 122

        self.soma[0].insert('Leak')
        self.soma[0].e_Leak = -61
        self.soma[0].gmax_Leak = 1E-3

        self.soma[0].insert('Nav1_6')
        self.soma[0].gbar_Nav1_6 = conductvalues[indiv_number, 14]
        self.soma[0].ena = 60
        
        self.soma[0].insert('Kv1_1')
        self.soma[0].gbar_Kv1_1 = conductvalues[indiv_number, 15]
        
        self.soma[0].insert('Kv1_5')
        self.soma[0].gKur_Kv1_5 = conductvalues[indiv_number, 35]
         
        self.soma[0].insert('Kv3_4')
        self.soma[0].gkbar_Kv3_4 = conductvalues[indiv_number, 16]
        
        self.soma[0].insert('Kir2_3')
        self.soma[0].gkbar_Kir2_3 = conductvalues[indiv_number, 17]
        
        self.soma[0].insert('Cav2_1') 
        self.soma[0].pcabar_Cav2_1 = conductvalues[indiv_number,18]
        
        self.soma[0].insert('Cav3_1') 
        self.soma[0].pcabar_Cav3_1 = conductvalues[indiv_number,19]
        
        self.soma[0].insert('Cav3_2')
        self.soma[0].gcabar_Cav3_2 =conductvalues[indiv_number,20]
        
        self.soma[0].insert('Cav3_3')
        self.soma[0].pcabar_Cav3_3 = conductvalues[indiv_number,21]

        self.soma[0].insert('Kca1_1')
        self.soma[0].gbar_Kca1_1 = conductvalues[indiv_number,22]
        
        self.soma[0].insert('Kca2_2') 
        self.soma[0].gkbar_Kca2_2 = conductvalues[indiv_number,23]
        
        self.soma[0].insert('Kca3_1')
        self.soma[0].gkbar_Kca3_1 = conductvalues[indiv_number,24]    
        self.soma[0].ek = -88
        
        self.soma[0].insert('HCN1')
        self.soma[0].gbar_HCN1 = conductvalues[indiv_number,25]
        self.soma[0].eh = -34.4        
        
        self.soma[0].insert('cdp5_CAM')
        
        self.soma[0].TotalPump_cdp5_CAM = 5e-8
        
        self.soma[0].push()
        self.soma[0].eca = 137.52625 
        
        h.define_shape()
        if spines_on == 1: #only for spines
            set_position(200, 962, 0)
        h.pop_section()
        
        self.whatami = "prk"
	
#Dend coordinate
	
           
        for d in self.dend:
            d.Ra = 122
            if spines_on == 1:
                d.cm = 2
            if spines_on == 0:    
                d.cm = (11.510294 * math.exp( - 1.376463 * d.diam) + 2.120503)
                
            d.nseg = 1 + (2 * int(d.L / 40))
            
            d.insert('Leak')
            d.e_Leak = -61
            d.gmax_Leak = 0.0003
            
            d.insert('Kv3_3')
            d.gbar_Kv3_3 = conductvalues[indiv_number,3]
            
            d.insert('Kv4_3')
            d.gkbar_Kv4_3 = conductvalues[indiv_number,4]
            
            d.insert('Cav2_1') 
            d.pcabar_Cav2_1 = conductvalues[indiv_number,6]*6
            
            d.insert('Cav3_3')
            d.pcabar_Cav3_3 = conductvalues[indiv_number,9]
        
            d.insert('Kca1_1')
            d.gbar_Kca1_1 = conductvalues[indiv_number,10]
            
            d.insert('HCN1')
            d.gbar_HCN1 = conductvalues[indiv_number,13]
            d.eh = -34.4
            
            d.insert('Kca2_2') 
            d.gkbar_Kca2_2 = conductvalues[indiv_number,11]
            
            d.insert('cdp5_CAM')
            
            d.TotalPump_cdp5_CAM = 6e-8
            
            if d.diam >= 1.6:
                d.cm = 2
                
                d.insert('Kv1_1')
                d.gbar_Kv1_1 = conductvalues[indiv_number,1]
                
                d.insert('Kv1_5')
                d.gKur_Kv1_5 = conductvalues[indiv_number,2]
                
                d.insert('Kir2_3')
                d.gkbar_Kir2_3 = conductvalues[indiv_number,5]   
                
                d.insert('Cav3_1') 
                d.pcabar_Cav3_1 = conductvalues[indiv_number,7]   
                
                d.insert('Cav3_2')
                d.gcabar_Cav3_2 = conductvalues[indiv_number,8]
                
                d.insert('Kca3_1') 
                d.gkbar_Kca3_1 = conductvalues[indiv_number,12]

                if d.diam >=3.3:
                    d.cm = 2
                    d.insert('Nav1_6')
                    d.gbar_Nav1_6 = conductvalues[indiv_number,0]    
                    d.ena = 60
            
            d.ek = -88
	    
            d.push()
            d.eca = 137.52625
            h.pop_section()
	
        
        
#Axon AIS. First section after the soma[0]
        self.axonAIS = h.Section(name='axonAIS')
        self.axonAIS.L = 17 #from Foust 2010 and clark 2005 and Somogyi76
        self.axonAIS.nseg = 1
        self.axonAIS.diam = 0.97 #ito 1983
        self.axonAIS.cm = 1
        
        self.axonAIS.Ra = 122
        
        self.axonAIS.insert('Leak')
        self.axonAIS.e_Leak = -61
        self.axonAIS.gmax_Leak = 0.0003
        
        self.axonAIS.insert('Nav1_6')
        self.axonAIS.gbar_Nav1_6 = conductvalues[indiv_number,26]*1.5
        self.axonAIS.ena = 60
        
        self.axonAIS.insert('Kv3_4')
        self.axonAIS.gkbar_Kv3_4 = conductvalues[indiv_number,27]
        self.axonAIS.ek = -88
        
        self.axonAIS.insert('Cav2_1') 
        self.axonAIS.pcabar_Cav2_1 = conductvalues[indiv_number,28]
        
        self.axonAIS.insert('Cav3_1') 
        self.axonAIS.pcabar_Cav3_1 = conductvalues[indiv_number,29]
        
        self.axonAIS.insert('cdp5_CAM')

        self.axonAIS.TotalPump_cdp5_CAM = 2e-8
	
        self.axonAIS.push()
        self.axonAIS.eca = 137.52625
        h.pop_section()
	
#AISK
	
        self.axonAISK = h.Section(name='axonAISK')
        self.axonAISK.L = 4 
        self.axonAISK.nseg = 1
        self.axonAISK.diam = 0.97 
        self.axonAISK.cm = 1
        
        self.axonAISK.Ra = 122
        
        self.axonAISK.insert('Leak')
        self.axonAISK.e_Leak = -61
        self.axonAISK.gmax_Leak = 0.0003
        
        self.axonAISK.insert('Kv1_1')
        self.axonAISK.gbar_Kv1_1 = conductvalues[indiv_number,30]
        self.axonAISK.ek = -88 #aggiunto 28 gennaio 2016.


	
#First Myelination

        self.axonmyelin = h.Section(name='axonmyelin')
        self.axonmyelin.L = 100 #da 60 a 150um
        self.axonmyelin.nseg = 1 + (2 * int(self.axonmyelin.L / 40))
        self.axonmyelin.diam = 0.73
        

        self.axonmyelin.insert('pas') #from brill77
        self.axonmyelin.e_pas = -61 #from brill77
        self.axonmyelin.g_pas = 5.60e-9 #from brill77
        self.axonmyelin.cm = 1.87e-11 #from brill77
        self.axonmyelin.Ra = 122	
	
#NEW NODES of RANVIER
		  
        secnumber = 3
	
#The IN paranode
        self.renv_node = [h.Section(name='renv_node'+str(x)) for x in range(secnumber)]

        for a in self.renv_node:
            a.L = 4
            a.nseg = 1 #+ (2 * int(a.L / 40))
            a.diam = 0.73
            a.cm = 1
            
            a.Ra = 122
            
            a.insert('Leak')
            a.e_Leak = -61
            a.gmax_Leak = 0.0003
            
            a.insert('Nav1_6')
            a.gbar_Nav1_6 = conductvalues[indiv_number,31] #0.03
            a.ena = 60
            
            a.insert('Kv3_4')
            a.gkbar_Kv3_4 = conductvalues[indiv_number,32] #0.02
            a.ek = -88

            a.insert('Cav2_1') 
            a.pcabar_Cav2_1 = conductvalues[indiv_number,33] #1e-5

            a.insert('Cav3_1') 
            a.pcabar_Cav3_1 = conductvalues[indiv_number,34] #2.2e-4

            a.insert('cdp5_CAM')
                
            a.TotalPump_cdp5_CAM = 5e-7
            
            a.push()
            a.eca = 137.52625
            h.pop_section() 
	
#Myelin sections
        self.axoninode = [h.Section(name='axoninode'+str(x)) for x in range(secnumber)]
        for d in self.axoninode:
            d.L = 100 
            d.nseg = 1 #+ (2 * int(d.L / 40))
            d.diam = 0.73
            
            
            d.insert('pas') #from brill77
            d.e_pas = -61 #from brill77
            d.g_pas = 5.60e-9 #from brill77
            d.cm = 1.87e-11 #from brill77
            d.Ra = 122


#Collateral.
        self.axoncoll = h.Section(name='axoncoll')
        self.axoncoll.nseg = 1
        self.axoncoll.diam = 0.6
        self.axoncoll.L = 100
        #self.axoncoll.Ra = 122 #critical

        self.axoncoll.insert('Leak')
        self.axoncoll.e_Leak = -61
        self.axoncoll.gmax_Leak = 0.0003
        
        self.axoncoll.insert('Nav1_6')
        self.axoncoll.gbar_Nav1_6 = 0.03 
        self.axoncoll.ena = 60

        self.axoncoll.insert('Kv3_4')
        self.axoncoll.gkbar_Kv3_4 = 0.02  
        self.axoncoll.ek = -88
        
        self.axoncoll.insert('Cav3_1') 
        self.axoncoll.pcabar_Cav3_1 = 1e-5 
        
        self.axoncoll.insert('Cav2_1') 
        self.axoncoll.pcabar_Cav2_1 = 2.2e-4 
        
        self.axoncoll.insert('cdp5_CAM')

            
        self.axoncoll.TotalPump_cdp5_CAM = 5e-7
        
        self.axoncoll.push()
        h.ion_style("ca_ion", 1, 1, 0, 1, 0) 
        self.axoncoll.eca = 137.52625
        self.axoncoll.cai = h.cai0_ca_ion
        self.axoncoll.cao = h.cao0_ca_ion
        h.pop_section()
        
#Collateral second part
        self.axoncoll2 = h.Section(name='axoncoll2')
        self.axoncoll2.nseg = 1
        self.axoncoll2.diam = 0.6
        self.axoncoll2.L = 100
        #self.axoncoll2.Ra = 122

        self.axoncoll2.insert('Leak')
        self.axoncoll2.e_Leak = -61
        self.axoncoll2.gmax_Leak = 0.0003
        
        self.axoncoll2.insert('Nav1_6')
        self.axoncoll2.gbar_Nav1_6 = 0.03 
        self.axoncoll2.ena = 60


        self.axoncoll2.insert('Kv3_4')
        self.axoncoll2.gkbar_Kv3_4 = 0.02  
        self.axoncoll2.ek = -88
        
        self.axoncoll2.insert('cdp5_CAM')
        
        self.axoncoll2.insert('Cav3_1') 
        self.axoncoll2.pcabar_Cav3_1 = 1e-5 
        
        self.axoncoll2.insert('Cav2_1') 
        self.axoncoll2.pcabar_Cav2_1 = 2.2e-4 
        
        self.axoncoll2.TotalPump_cdp5_CAM = 5e-7
        
        self.axoncoll2.push()
        h.ion_style("ca_ion", 1, 1, 0, 1, 0) 
        self.axoncoll2.eca = 137.52625
        self.axoncoll2.cai = h.cai0_ca_ion
        self.axoncoll2.cao = h.cao0_ca_ion
        h.pop_section()	  
	
        self.axonAIS.connect(self.soma[0],0,0)
        self.axonAISK.connect(self.axonAIS,1,0)
        self.axonmyelin.connect(self.axonAISK,1,0)	#myelin1
        self.renv_node[0].connect(self.axonmyelin,1,0)      #node1
        self.axoninode[0].connect(self.renv_node[0],1,0)	#myelin2
        self.renv_node[1].connect(self.axoninode[0],1,0)	#node2
        self.axoninode[1].connect(self.renv_node[1],1,0)	#myelin3
        self.renv_node[2].connect(self.axoninode[1],1,0)	#node3      
        self.axoninode[2].connect(self.renv_node[2],1,0)	#myelin4
        
        self.axoncoll.connect(self.renv_node[1],1,0)
        self.axoncoll2.connect(self.axoncoll,1,0)
        
        if spines_on == 1:
            dend_index = []
            dend_local_len = []
            dend_len = []
            spine_location = []	 
            self.dend_spine = []
            
            #number of spikes per micron
            spine_per_micron = 2
            
            for e in self.dend:
                if e.diam >= 0 and e.diam < 1.5:
                    self.dend_spine.append(e)
                    
            original_dend_index = []        
            for old_index, section_acess in enumerate(self.dend):
                if section_acess.diam >= 0 and section_acess.diam < 1.5:
                    original_dend_index.append(old_index)
           
                       
            for local_index, x_dend in enumerate(self.dend_spine):
                dend_index.append(local_index)
                dend_local_len.append(int(x_dend.L) * spine_per_micron)
                
                dend_len.append(int(x_dend.L) * spine_per_micron)
                
                to_list = np.linspace(0.05, 0.95, num = int(x_dend.L) * spine_per_micron)
                spine_location.append(to_list.tolist())          
        
            dend_total_len = sum(dend_len)
            info_dend = list(zip(dend_index,dend_local_len))
            
            info_dend_old_index = list(zip(original_dend_index,dend_local_len))
            spine_number = int(dend_total_len) 
            somma_location_spines = sum(spine_location,[])
            

            self.spine_head = [h.Section(name='spine_head_'+str(x)) for x in range(spine_number)]

            for i in self.spine_head:
                    i.nseg = 1
                    i.diam = 1 
                    i.cm = 2
                    i.L = 0.35
                    i.Ra = 122
                    
                    i.insert('Leak')
                    i.e_Leak = -61
                    i.gmax_Leak = 1e-5
                    
                    i.insert('Cav2_1') 
                    i.pcabar_Cav2_1 = 1e-5
                    
                    i.insert('Cav3_1') 
                    i.pcabar_Cav3_1 = 1e-6
                    
                    i.insert('Kca1_1')
                    i.gbar_Kca1_1 = 0.01
                    
                    i.insert('Kca2_2') 
                    i.gkbar_Kca2_2 = 1e-3
                    
                    i.insert('Kv4_3')
                    i.gkbar_Kv4_3 =  0.005
            
                
                    
                    i.insert('cdp5_CAM')
                    i.Nannuli_cdp5_CAM = 0.326 + (1.94 * (i.diam)) + (0.289*(i.diam)*(i.diam)) - ((3.33e-2)*(i.diam)*(i.diam)*(i.diam)) + ((1.55e-3)*(i.diam)*(i.diam)*(i.diam)*(i.diam)) - (2.55e-5*(i.diam)*(i.diam)*(i.diam)*(i.diam)*(i.diam))
                    i.Buffnull2_cdp5_CAM = 64.2 - 57.3* math.exp(-(i.diam)/1.4)
                    i.rf3_cdp5_CAM = 0.162 - 0.106* math.exp(-(i.diam)/2.29)
                    i.rf4_cdp5_CAM = 0.003 
                    
                    i.push()
                    h.ion_style("ca_ion", 1, 1, 0, 1, 0) 
                    i.eca = 137.52625
                    i.cai = h.cai0_ca_ion
                    i.cao = h.cao0_ca_ion
                    h.pop_section()
                    
                    i.TotalPump_cdp5_CAM = 7e-10
	    
#NECK OF THE SPINE
            self.spine_neck = [h.Section(name='spine_neck_'+str(x)) for x in range(spine_number)]
	    
            for x in self.spine_neck:
                x.nseg = 1
                x.diam = 0.2 
                x.cm = 3
                x.L = 0.7 
                x.Ra = 122 
                
                x.insert('Leak')
                x.e_Leak = -61
                x.gmax_Leak = 1e-5
                
		
            for i,j in enumerate(self.spine_head):
                j.connect(self.spine_neck[i],1,0)

            print("spine active!!")
            
            list_dend_local = []
            c,v = zip(*info_dend)
            for i in range(len(info_dend)):
                new_list = []
                new_list = [c[i]]*v[i]
                
                
                list_dend_local.append(new_list)
            
            list_dend_old_index = []
            c_old,v_old = zip(*info_dend_old_index)
            for i in range(len(info_dend_old_index)):
                new_list_oldindex = []
                
                new_list_oldindex = [c_old[i]]*v_old[i]
                
                
                list_dend_old_index.append(new_list_oldindex)
            
            list_dend_complete = sum(list_dend_local,[])
            
            list_dend_complete_oldindex = sum(list_dend_old_index,[])            
            
            
            self.total_list = list(zip(range(dend_total_len), list_dend_complete, somma_location_spines))
        
            
            print('dend_total_len', len(range(dend_total_len)))
            print('list_dend_complete', len(list_dend_complete))
            print('somma_location_spines', len(somma_location_spines))

            spine_sec_aa = []
            spine_sec_pf = []
            spine_sec_aa_SC = []           
            
            spine_print_list = range(0, len(self.total_list), 500)


            c_source,v_dest,z_step = zip(*self.total_list) # outside the for... not inside....
            for spines_created in range(len(self.total_list)):
                if spines_created in spine_print_list:
                    print('spines_placed', spines_created)
                #AA
                if self.dend_spine[v_dest[spines_created]].diam <= 0.3:
                    self.spine_neck[c_source[spines_created]].connect(self.dend_spine[v_dest[spines_created]],z_step[spines_created],0)  
                    spine_sec_aa.append(c_source[spines_created])                

                #AA/SC
                if self.dend_spine[v_dest[spines_created]].diam > 0.3 and self.dend_spine[v_dest[spines_created]].diam <= 0.75:
                    self.spine_neck[c_source[spines_created]].connect(self.dend_spine[v_dest[spines_created]],z_step[spines_created],0)  
                    spine_sec_aa_SC.append(c_source[spines_created])   
                    
                #PF/SC   
                if self.dend_spine[v_dest[spines_created]].diam > 0.75 and self.dend_spine[v_dest[spines_created]].diam <= 1.5:
                    self.spine_neck[c_source[spines_created]].connect(self.dend_spine[v_dest[spines_created]],z_step[spines_created],0)   
                    spine_sec_pf.append(c_source[spines_created])      
                    
            #AA
            self.spine_head_aa = []
            #SC
            self.spine_head_aa_SC = []               
            #PF
            self.spine_head_pf = []
            
            spine_print_list_syn = range(0, len(self.spine_head), 5000)
        
            for i_head, e_head in enumerate(self.spine_head):
                if i_head in spine_print_list_syn:
                    print('spines_syn: ', e_head)
                elif i_head in spine_sec_aa:
                    self.spine_head_aa.append(e_head)    
                elif i_head in spine_sec_aa_SC:
                    self.spine_head_aa_SC.append(e_head)
                elif i_head in spine_sec_pf:
                    self.spine_head_pf.append(e_head)
                elif i_head in spine_sec_CF:
                    self.spine_head_CF.append(e_head)  
                    
                    
                    
            print('spine_head_aa        between 0 - 0.3     ->', len(self.spine_head_aa)) 
            print('spine_head_aa_SC     between 0.3 - 0.75  ->', len(self.spine_head_aa_SC)) 
            print('spine_head_pf_SC     between 0.75 - 1.5  ->', len(self.spine_head_pf)) 
            
            print('!!!!!!!!!!!!!!!       spine_total', len(self.spine_head_aa_SC) + len(self.spine_head_aa) + len(self.spine_head_pf), '       !!!!!!!!!!!!!!!')


            self.dendAA_SC = []
            self.dendAA_SC_index = []
            
            self.dendPF_SC = []
            self.dendPF_SC_index = []
            
            for i,e in enumerate(self.dend):
                if e.diam > 0.3 and e.diam < 0.75:
                    self.dendAA_SC.append(e)
                    self.dendAA_SC_index.append(i)
                
                if e.diam > 0.75 and e.diam < 1.5:
                    self.dendPF_SC.append(e)	
                    self.dendPF_SC_index.append(i)
            
             		
#Recorder section. 	    

        
        self.time_vector = h.Vector()
        self.time_vector.record(h._ref_t)

        self.vm_soma = h.Vector()
        self.vm_soma.record(self.soma[0](0.5)._ref_v)
	

    def createsyn(self, npf, naa, numpfstl):
       		
             		

        self.dendparallel = []
        self.dendascending = []
        
        self.dendAAstl = []
        self.dendPFstl = []
                    
    #new subdivision for the synapses		
        for i in self.dend:
            if i.diam > 0.75 and i.diam < 1.5:
                self.dendparallel.append(i)
                self.dendPFstl.append(i)
            
            if i.diam <= 0.75: 
                self.dendascending.append(i)
            

        
        print("PF")
        print(len(self.dendparallel))	  
        print("AA")
        print(len(self.dendascending))

        #Parallel fiber list
        self.PF_L = []
        
        #Ascending axon list
        self.AA_L = []

        #Stellate list
        self.SC_L = []

        #Randomizer
        self.pfrand = []
        self.aarand = []
        self.stlrand = []
        self.bgrand = []
    
        
        self.pfstlrand = []
        self.aastlrand = []
        
        #New lists for the new stellate setup
        
        self.pfSC_L = []
        self.aaSC_L = []
        self.cfSC_L = []

    
	
        n = len(self.dendparallel)
        n2 = len(self.dendascending)
        npfstl = len(self.dendPFstl)

	  
	
#PF random
        for j in range(npf):
            n -= 1
            i = random.randint(0, n)	    
            self.dendparallel[i], self.dendparallel[n] = self.dendparallel[n], self.dendparallel[i]
            self.pfrand.append(self.dendparallel[n])
            self.PF_L.append(Synapse_py3('pf',self,self.pfrand[j]))
	  
	   	  
#AA random
        for j in range(naa):
            n2 -= 1
            i = random.randint(0, n2)
            self.dendascending[i], self.dendascending[n2] = self.dendascending[n2], self.dendascending[i]
            self.aarand.append(self.dendascending[n2])
            self.AA_L.append(Synapse_py3('aa',self,self.aarand[j]))

#STl_PF random
        for j in range(numpfstl):
            npfstl -= 1
            i = random.randint(0, npfstl)
            self.dendPFstl[i], self.dendPFstl[npfstl] = self.dendPFstl[npfstl], self.dendPFstl[i]
            self.stlrand.append(self.dendPFstl[npfstl])
            self.SC_L.append(Synapse_py3('stl',self,self.stlrand[j]))	    

#revised code 2020 python3
    def dendrites_xy_nospine(self,min_x, max_x, min_y,max_y,syntype):#,option_random,number_random_syn):
      
        self.dend_cood_x = {}
        self.dend_cood_y = {}	    
        for i, j in enumerate(self.dend):
            j.push()
            self.dend_cood_x["x0_"+str(i)] = h.x3d(0)
            self.dend_cood_y["y0_"+str(i)] = h.y3d(0)  
            h.pop_section()
	    
    #only of X    
        dend_x_values = sorted(self.dend_cood_x.values()) #sorting of the values
        dend_x_kyes = sorted(self.dend_cood_x, key=self.dend_cood_x.get) #sorting of the keys based on the values
        dend_list_zip_x = zip(dend_x_kyes, dend_x_values)
        
        #print 'dimensions X'
        #print min_x
        #print max_x

    #only of Y   
        dend_y_values = sorted(self.dend_cood_y.values()) #sorting of the values
        dend_y_kyes = sorted(self.dend_cood_y, key=self.dend_cood_y.get) #sorting of the keys based on the values
        dend_list_zip_y = zip(dend_y_kyes, dend_y_values)
        
        #print 'dimensions Y'
        #print min_y
        #print max_y	
        
        self.x_coord_dend = []
        xlist = []
        for i in dend_list_zip_x:
            if i[1] > min_x and i[1] < max_x:
                xlist.append(i[1])
                self.x_coord_dend.append(int(i[0][3:]))   
        
        
        self.y_coord_dend = []
        ylist = []
        for i in dend_list_zip_y:
            if i[1] > min_y and i[1] < max_y:
                ylist.append(i[1])
                self.y_coord_dend.append(int(i[0][3:]))   
            
        #print "dend number"	
        #print len(self.x_coord_dend)   
        #print len(self.y_coord_dend)  
        
        self.total_dend = []
        self.total_dend = set(self.x_coord_dend).intersection(self.y_coord_dend)
        
        #print 'number syn'
        #print self.total_dend
        
        print('dend set', self.total_dend)
        print('Dend synapses ON.')
        
        self.diam = []
        for x in self.total_dend:
            self.diam.append(self.dend[x].diam)
        
        self.dend_diam = zip(self.total_dend, self.diam)

    #new list of synapses	

        self.AAtotal = []
        self.PFtotal = []
        self.SCtotal = []
        
        if syntype == 0:
            self.AAdendminmax = []
        if syntype == 1:        
            self.PFdendminmax = []
        if syntype == 2:    
            self.SCdendminmax = []
            
        for x in self.dend_diam:
            if syntype == 0:
                if x[1] <= 0.75:
                    #if option_random == 1:
                        #self.AAtotal.append(self.dend[x[0]])
                    #else:
                        self.AAdendminmax.append(Synapse_py3('aa',self,self.dend[x[0]]))  
                        #print('AA')
                
            if syntype == 1:    
                if x[1] > 0.75 and x[1] < 1.6:
                    #if option_random == 1:
                        #self.PFtotal.append(self.dend[x[0]])
                        ##print('x[0]', x[0])
                    #else:
                        self.PFdendminmax.append(Synapse_py3('pf',self,self.dend[x[0]]))  
                        #print('x[0]', x[0])
                        #print('nope')
        
            if syntype == 2:    
                if x[1] >= 0.3 and x[1] < 1.6:
                    #if option_random == 1:
                        #self.SCtotal.append(self.dend[x[0]])
                    #else:
                        self.SCdendminmax.append(Synapse_py3('stl',self,self.dend[x[0]])) 

        if syntype == 0:	          
            print('AA syn number!!!!!!!!!!!!!!!!!!!!!!!!!!!', len(self.AAdendminmax))
            
            #self.AAdendminmax = self.AAdendminmax + self.AAdendminmax + self.AAdendminmax
        elif syntype == 1:  
            print('PF syn number!!!!!!!!!!!!!!!!!!!!!!!!!!!', len(self.PFdendminmax))
            
            #self.PFdendminmax = self.PFdendminmax + self.PFdendminmax + self.PFdendminmax
            
        elif syntype == 2:  
            print('STL syn number!!!!!!!!!!!!!!!!!!!!!!!!!!', len(self.SCdendminmax))

#new spine coordinator

    def spine_space_limit(self):
        h.define_shape() 
        self.spinehead_coord_x = {}
        self.spinehead_coord_y = {}	    
        
        print('checking the size of the morphology')
        for i, j in enumerate(self.spine_head):
            j.push()
            self.spinehead_coord_x["x0_"+str(i)] = h.x3d(0)
            self.spinehead_coord_y["y0_"+str(i)] = h.y3d(0)  
            h.pop_section()
            
    ##only of X    
        self.spine_space_x = sorted(self.spinehead_coord_x.values())

    ##only of Y   
        self.spine_space_y = sorted(self.spinehead_coord_y.values()) 

    def dendrites_xy(self,min_x, max_x, min_y,max_y):
        self.dend_cood_x = {}
        self.dend_cood_y = {}	    
        for i, j in enumerate(self.dend):
            j.push()
            self.dend_cood_x["x0_"+str(i)] = h.x3d(0)
            self.dend_cood_y["y0_"+str(i)] = h.y3d(0)  
            h.pop_section()
	    
    #only of X    
        dend_x_values = sorted(self.dend_cood_x.values())
        dend_x_kyes = sorted(self.dend_cood_x, key=self.dend_cood_x.get) 
        dend_list_zip_x = zip(dend_x_kyes, dend_x_values)


    #only of Y   
        dend_y_values = sorted(self.dend_cood_y.values()) 
        dend_y_kyes = sorted(self.dend_cood_y, key=self.dend_cood_y.get) 
        dend_list_zip_y = zip(dend_y_kyes, dend_y_values)
        
        
        self.x_coord_dend = []
        xlist = []
        for i in dend_list_zip_x:
            if i[1] > min_x and i[1] < max_x:
                xlist.append(i[1])
                self.x_coord_dend.append(int(i[0][3:]))   
        
        
        self.y_coord_dend = []
        ylist = []
        for i in dend_list_zip_y:
            if i[1] > min_y and i[1] < max_y:
                ylist.append(i[1])
                self.y_coord_dend.append(int(i[0][3:]))   
        
        self.total_spines = []
        self.total_spines = set(self.x_coord_dend).intersection(self.y_coord_dend)
        
        print('total_dend_SC', len(self.total_spines))
        
        
    #New code dend only on X and Y at the same time to gnerate spots of activation	    	    
    def spine_heads_x_y(self,min_x, max_x, min_y,max_y):
        h.define_shape() #gives pt3d x,v,z to all the spines       
        self.spinehead_coord_x = {}
        self.spinehead_coord_y = {}	    
        for i, j in enumerate(self.spine_head):
            j.push()
            self.spinehead_coord_x["x0_"+str(i)] = h.x3d(0)
            self.spinehead_coord_y["y0_"+str(i)] = h.y3d(0)  
            h.pop_section()
            
    #only of X    
        spine_x_values = sorted(self.spinehead_coord_x.values()) #sorting of the values
        spine_x_kyes = sorted(self.spinehead_coord_x, key=self.spinehead_coord_x.get) #sorting of the keys based on the values
        spine_list_zip_x = zip(spine_x_kyes, spine_x_values)
        

    #only of Y   
        spine_y_values = sorted(self.spinehead_coord_y.values()) #sorting of the values
        spine_y_kyes = sorted(self.spinehead_coord_y, key=self.spinehead_coord_y.get) #sorting of the keys based on the values
        spine_list_zip_y = zip(spine_y_kyes, spine_y_values)

        
        self.x_coord_spinehead = []
        xlist = []
        for i in spine_list_zip_x:
            if i[1] > min_x and i[1] < max_x:
                xlist.append(i[1])
                self.x_coord_spinehead.append(int(i[0][3:]))   
        
        
        self.y_coord_spinehead = []
        ylist = []
        for i in spine_list_zip_y:
            if i[1] > min_y and i[1] < max_y:
                ylist.append(i[1])
                self.y_coord_spinehead.append(int(i[0][3:]))   
        
        self.total_spines = []
        self.total_spines = set(self.x_coord_spinehead).intersection(self.y_coord_spinehead)

        print('total_spines', len(self.total_spines))
        



    def activator(self, syntype, option_random, n_aa_set, n_pf_set, n_aa_sc_set, n_pf_sc_set, n_aa_aa_set):
        print('spine synapses ON.')

#AA only with random and fixed  
        self.AAdendminmax = []
        self.aarand = []
        self.aarand_spine_num = []
        self.AA_total = []
        
        if syntype == 0:
            for sec_name_value_AA in self.spine_head_aa:
                for y in self.total_spines: 
                    if 'spine_head_' + str(y) == str(sec_name_value_AA):
                        if option_random == 0:
                            self.AAdendminmax.append(Synapse_py3('aa',self,sec_name_value_AA))  
                            self.aarand_spine_num.append(self.sec_name_value_AA.name())
                        else:    
                            self.AA_total.append(self.sec_name_value_AA)
        
            if option_random == 1:
                
                naa = len(self.AA_total)
                for z in range(int((naa/100)*n_aa_set)):
                    naa -= 1
                    i = random.randint(0, naa)
                    self.AA_total[i], self.AA_total[naa] = self.AA_total[naa], self.AA_total[i]
                    self.aarand.append(self.AA_total[naa])
                    self.AAdendminmax.append(Synapse_py3('aa',self,self.self.aarand[z])) 
                    self.aarand_spine_num.append(self.aarand[z].name())
    
            

#PF only with random and fixed
        self.PFdendminmax = []
        self.pfrand = []
        self.pfrand_spine_num = []
        self.PF_total = []
        
        if syntype == 1:  
            for sec_name_value_PF in self.spine_head_pf:
                for y in self.total_spines: 
                    if 'spine_head_' + str(y) == str(sec_name_value_PF):
                        if option_random == 0:
                            self.PFdendminmax.append(Synapse_py3('pf',self,sec_name_value_PF))
                            self.pfrand_spine_num.append(sec_name_value_PF.name())
                            
                        else:    
                            self.PF_total.append(sec_name_value_PF)

            if option_random == 1:
                npf = len(self.PF_total)
                for z in range(int((npf/100)*n_pf_set)):
                    npf -= 1
                    i = random.randint(0, npf)
                    self.PF_total[i], self.PF_total[npf] = self.PF_total[npf], self.PF_total[i]
                    self.pfrand.append(self.PF_total[npf])
                    self.PFdendminmax.append(Synapse_py3('pf',self,self.pfrand[z])) 
                    self.pfrand_spine_num.append(self.pfrand[z].name())

#SC on AA only with random and fixed
        self.SC_AAdendminmax = []
        self.SC_AA_rand = []
        self.SC_AA_spine_num = []
        self.SC_AA_total = []
        if syntype == 2:   
            for sec_name_value_SC_AA in self.dendAA_SC_index:
                for y in self.total_spines:
                    if str(y) == str(sec_name_value_SC_AA):
                        if option_random == 0:
                            self.SC_AAdendminmax.append(Synapse_py3('stl',self,self.dend[sec_name_value_SC_AA])) 
                            self.SC_AA_spine_num.append(sec_name_value_SC_AA)
                        else:
                            self.SC_AA_total.append(self.dend[sec_name_value_SC_PF])
                            
            if option_random == 1:
                naasc = len(self.SC_AA_total)
                for z in range(int((naasc/100)*n_aa_sc_set)):
                    naasc -= 1
                    i = random.randint(0, naasc)
                    self.SC_AA_total[i], self.SC_AA_total[naasc] = self.SC_AA_total[naasc], self.SC_AA_total[i]
                    self.SC_AA_rand.append(self.SC_AA_total[naasc])
                    self.SC_AAdendminmax.append(Synapse_py3('stl',self,self.SC_AA_rand[z])) 
                    self.SC_AA_spine_num.append(self.SC_AA_rand[z].name())

#SC on PF only with random and fixed
        self.SC_PFdendminmax = [] 
        self.SC_PF_rand = []
        self.SC_PF_spine_num = []
        self.SC_PF_total = []

        if syntype == 3:
            for sec_name_value_SC_PF in self.dendPF_SC_index:
                for y in self.total_spines:
                    if y == sec_name_value_SC_PF:
                        if option_random == 0:
                            #print('no random SC_PF')
                            self.SC_PFdendminmax.append(Synapse_py3('stl',self,self.dend[sec_name_value_SC_PF]))
                            self.SC_PF_spine_num.append(sec_name_value_SC_PF)
                        else:
                            self.SC_PF_total.append(self.dend[sec_name_value_SC_PF])
                     
            if option_random == 1:
                naapf = len(self.SC_PF_total)
                for z in range(int((naapf/100)*n_pf_sc_set)):
                    naapf -= 1
                    i = random.randint(0, naapf)
                    self.SC_PF_total[i], self.SC_PF_total[naapf] = self.SC_PF_total[naapf], self.SC_PF_total[i]
                    self.SC_PF_rand.append(self.SC_PF_total[naapf])
                    self.SC_PFdendminmax.append(Synapse_py3('stl',self,self.SC_PF_rand[z])) 
                    self.SC_PF_spine_num.append(self.SC_PF_rand[z].name())
                    
                          
        
#AA only in SC location        
        self.AA_AAdendminmax = []
        self.AA_AA_rand = []
        self.AA_AA_spine_num = []
        self.AA_AA_total = []
        
        if syntype == 4:  
            for sec_name_value_AA_AA in self.spine_head_aa_SC:
                for y in self.total_spines: 
                    if 'spine_head_' + str(y) == str(sec_name_value_AA_AA):
                        if option_random == 0:
                            self.AA_AAdendminmax.append(Synapse_py3('aa',self,sec_name_value_AA_AA))
                            self.AA_AA_spine_num.append(sec_name_value_AA_AA.name())
                        else:    
                            self.AA_AA_total.append(sec_name_value_AA_AA)

            if option_random == 1:
                naa_aasc = len(self.AA_AA_total)
                for z in range(int((naa_aasc/100)*n_aa_aa_set)):
                    naa_aasc -= 1
                    i = random.randint(0, naa_aasc)
                    self.AA_AA_total[i], self.AA_AA_total[naa_aasc] = self.AA_AA_total[naa_aasc], self.AA_AA_total[i]
                    self.AA_AA_rand.append(self.AA_AA_total[naa_aasc])
                    self.AA_AAdendminmax.append(Synapse_py3('aa',self,self.AA_AA_rand[z])) 
                    self.AA_AA_spine_num.append(self.AA_AA_rand[z].name())
          
