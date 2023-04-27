import os
import json
import argparse
import xml.etree.ElementTree as gfg


def get_cadynamics(params_dict):
    ca_props = {
        'minCai': '1e-4 mM',
        'depth': '0.1 um'
    }
    for cprops in params_dict['genome']:
        if cprops['name'] == 'decay_CaDynamics' and cprops['section'] == 'soma':
            ca_props['decay'] = '{} ms'.format(cprops['value'])
        
        elif cprops['name'] == 'gamma_CaDynamics' and cprops['section'] == 'soma':
            ca_props['gamma'] = '{} ms'.format(cprops['value'])

    return ca_props


mech2ion = {
    'Im': 'k',
    'Ih': 'hcn',
    'NaTs': 'na',
    'Nap': 'na',
    'NaV': 'na',
    'K_P': 'k',
    'K_T': 'k',
    'SK': 'k',
    'Kv3_1': 'k',
    'Kd': 'k',
    'Kv2like': 'k',
    'Im_v2': 'k',
    'Ca_HVA': 'ca',
    'Ca_LVA': 'ca'
}
def find_ion(name, mechanism):
    if not mechanism and name == 'g_pas':
        return 'non_specific'
    return mech2ion[mechanism]


def find_erev(ion, param_dict):
    conditions = param_dict['conditions'][0]
    if ion == 'k':
        val = conditions['erev'][0]['ek']
    elif ion == 'na':
        val = conditions['erev'][0]['ena']
    elif ion == 'non_specific':
        val = conditions['v_init']
    elif ion == 'hcn':
        val = -45
    else:
        raise ValueError()
    
    return '{} mV'.format(val)


def GenerateXML(param_dict, cell_id=None, save_as=None) :
    get_cadynamics(param_dict)

    root = gfg.Element(
        'neuroml', 
        id="NeuroML2_file_exported_from_NEURON" 
    )
    root.set('xmlns', 'http://www.neuroml.org/schema/neuroml2')
    root.set('xmlns:xs', 'http://www.w3.org/2001/XMLSchema')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance') 
    root.set('xsi:schemaLocation', 'http://www.neuroml.org/schema/neuroml2 https://raw.github.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_v2beta4.xsd')

    # Add Ca concentration information
    conc_model = gfg.Element('concentrationModel', id='CaDynamics')
    conc_model.set('type', 'CaDynamics')
    conc_model.set('segmentGroup', 'soma')
    conc_model.set('ion', 'ca')
    for k, v in get_cadynamics(param_dict).items():
        conc_model.set(k, v)
    root.append(conc_model)
    
    # Cell id information
    cell_id = gfg.Element('cell', id=str(cell_id))
    root.append(cell_id)

    biophys_props = gfg.Element('biophysicalProperties', id='biophys')
    root.append(biophys_props)
    
    membrane_props = gfg.Element('membraneProperties')
    biophys_props.append(membrane_props)

    for cprops in param_dict['genome']:
        name = cprops['name']
        
        if name in ['gamma_CaDynamics', 'decay_CaDynamics']:
            continue

        elif name in ['gbar_Ca_HVA', 'gbar_Ca_LVA']:
            ion = find_ion(cprops['name'], cprops['mechanism'])
            cd = gfg.Element('channelDensityNernst')
            cd.set('id', cprops['name'])
            cd.set('segmentGroup', cprops['section'])
            cd.set('ion', ion)
            cd.set('ionChannel', cprops['mechanism'])
            cd.set('condDensity', '{} S_per_cm2'.format(cprops['value']))
            membrane_props.append(cd)

        elif name == 'g_pas':
            cd = gfg.Element('channelDensity')
            cd.set('id', '{}_{}'.format(cprops['name'], cprops['section']))
            cd.set('segmentGroup', cprops['section'])
            cd.set('ion', 'non_specific')
            cd.set('ionChannel', 'pas')
            cd.set('erev', find_erev('non_specific', param_dict))
            cd.set('condDensity', '{} S_per_cm2'.format(cprops['value']))
            membrane_props.append(cd)
        
        else:
            # channel_type = 'channelDensityNernst' if cprops['name'] in ['gbar_Ca_HVA', 'gbar_Ca_LVA'] else 'channelDensity'
            ion = find_ion(cprops['name'], cprops['mechanism'])
            erev = find_erev(ion, param_dict)
            cd = gfg.Element('channelDensity')
            cd.set('id', cprops['name'])
            cd.set('segmentGroup', cprops['section'])
            cd.set('ion', ion)
            cd.set('ionChannel', cprops['mechanism'])
            cd.set('erev', erev)
            cd.set('condDensity', '{} S_per_cm2'.format(cprops['value']))
            membrane_props.append(cd)

    passive_props = param_dict['passive'][0]
    for cm in passive_props['cm']:
        sc = gfg.Element('specificCapacitance')
        sc.set('segmentGroup', cm['section'])
        sc.set('value', '{} uF_per_cm2'.format(cm['cm']))
        membrane_props.append(sc)


    intracell_props = gfg.Element('intracellularProperties')
    biophys_props.append(intracell_props)

    species = gfg.Element(
        'species',
        segmentGroup='soma', 
        ion='ca', 
        initialExtConcentration='2 mM',
        concentrationModel='CaDynamics', 
        id='ca', 
        initialConcentration='0.0001 mM'
    )
    intracell_props.append(species)

    resistivity = gfg.Element(
        'resistivity',
        segmentGroup='all',
        value='{} ohm_cm'.format(passive_props['ra'])
    )
    intracell_props.append(resistivity)
     
    tree = gfg.ElementTree(root)
      
    if save_as:
        with open (save_as, "wb") as files :
            tree.write(files)

    gfg.indent(root)
    print(gfg.tostring(root, encoding='unicode'))
  

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', nargs=1, type=str)
    parser.add_argument('--cell-id', nargs='?', type=int)
    parser.add_argument('--save-as', nargs='?', type=str)
    args = parser.parse_args()
    
    cell_id = args.cell_id
    json_path = args.json_path[0]
    if cell_id is None:
        print(args)
        cell_id = int(os.path.basename(json_path).replace('_fit.json', ''))

    params_dict = json.load(open(json_path, 'r'))
    GenerateXML(params_dict, cell_id=cell_id, save_as=args.save_as)
