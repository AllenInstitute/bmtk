import os
import pandas as pd
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


pd.set_option('display.max_columns', None)
output_dir = './ecephys_cache_dir'
manifest_path = os.path.join(output_dir, 'neuropixels.manifest.json')

region_dict = {
    'cortex' : ['VISp', 'VISl', 'VISrl', 'VISam', 'VISpm', 'VIS', 'VISal','VISmma','VISmmp','VISli'],
    'thalamus' : ['LGd','LD', 'LP', 'VPM', 'TH', 'MGm','MGv','MGd','PO','LGv','VL', 'VPL','POL','Eth','PoT','PP','PIL','IntG','IGL','SGN','VPL','PF','RT'],
    'hippocampus' : ['CA1', 'CA2','CA3', 'DG', 'SUB', 'POST','PRE','ProS','HPF'],
    'midbrain': ['MB','SCig','SCiw','SCsg','SCzo','PPT','APN','NOT','MRN','OP','LT','RPF','CP']
}

def find_region(row):
    for reg, areas in region_dict.items():
        if row['ecephys_structure_acronym'] in areas:
            return reg
    return 'unknown'


def get_visl_session():
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    # sessions = cache.get_session_table()
    # mask = (sessions['session_type'] == 'brain_observatory_1.1') & (['VISl' in esa for esa in sessions['ecephys_structure_acronyms']])
    # visl_sessions = sessions[mask]
    # print(visl_sessions)

    units = cache.get_units(
        # isi_violations_maximum=np.inf,
        # amplitude_cutoff_maximum=np.inf,
        # presence_ratio_minimum=-np.inf
    )
    units_mask = (units['has_lfp_data'] == True)
    units_mask &= (units['session_type'] == 'brain_observatory_1.1') 
    units_mask &= (units['ecephys_structure_acronym'] == 'VISl')
    
    visl_units = units[units_mask]
    visl_session = visl_units[['ecephys_session_id']].value_counts()
    #print(visl_session)
    print(units)
    
    # print(units.head())
    # print()


def get_visl_cells():
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    units = cache.get_units(
        # isi_violations_maximum=np.inf,
        # amplitude_cutoff_maximum=np.inf,
        # presence_ratio_minimum=-np.inf
    )
    units_mask = True
    units_mask = (units['has_lfp_data'] == True)
    units_mask &= (units['session_type'] == 'brain_observatory_1.1') 
    units_mask &= (units['ecephys_structure_acronym'] == 'VISl')
    visl_units = units[units_mask]

    visl_session = visl_units[['ecephys_session_id']].value_counts()
    session_id = visl_session.index[0][0]
    cache.get_session_data(session_id)

    filtered_units = units[(units['ecephys_session_id'] == session_id) & (units['ecephys_structure_acronym'] == 'VISl')]
    print(filtered_units)
    pd.DataFrame({
        'node_ids': range(len(filtered_units)),
        'unit_ids': filtered_units.index.values,
        'session_id': session_id,
        'ecephys_structure_acronym': filtered_units['ecephys_structure_acronym'].values
    }).to_csv('visl_unit_ids.csv', index=False, sep=' ')



def get_hippocampal_cells():
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    units = cache.get_units(
        # isi_violations_maximum=np.inf,
        # amplitude_cutoff_maximum=np.inf,
        # presence_ratio_minimum=-np.inf
    )
    units['area'] = units.apply(find_region, axis=1)

    units_mask = True
    units_mask = (units['has_lfp_data'] == True)
    units_mask &= (units['session_type'] == 'brain_observatory_1.1') 
    # units_mask &= (units['ecephys_structure_acronym'] == 'LGN')
    units_mask &= (units['area'] == 'hippocampus')

    hippo_units = units[units_mask]
    visl_session = hippo_units[['ecephys_session_id']].value_counts()
    session_id = visl_session.index[0][0]
    # print(session_id)
    cache.get_session_data(session_id)

    # unit_ids = units[(units['ecephys_session_id'] == session_id) & (units['area'] == 'hippocampus')].index.values
    filtered_units = units[(units['ecephys_session_id'] == session_id) & (units['area'] == 'hippocampus')]
    pd.DataFrame({
        'node_ids': range(len(filtered_units)),
        'unit_ids': filtered_units.index.values,
        'session_id': session_id,
        'ecephys_structure_acronym': filtered_units['ecephys_structure_acronym'].values
    }).to_csv('hippocampal_unit_ids.csv', index=False, sep=' ')
    

def get_area_map(area, valid_units=False):
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    filter_params = {} if valid_units else {'isi_violations_maximum': np.inf, 'amplitude_cutoff_maximum': np.inf, 'presence_ratio_minimum': -np.inf}
    units = cache.get_units(**filter_params)
    units['area'] = units.apply(find_region, axis=1)

    area_units = units[(units['has_lfp_data'] == True) & (units['session_type'] == 'brain_observatory_1.1') & (units['area'] == area)]
    area_session = area_units[['ecephys_session_id']].value_counts()
    session_id = area_session.index[0][0]
    cache.get_session_data(session_id)

    filtered_units = units[(units['ecephys_session_id'] == session_id) & (units['area'] == area)]
    pd.DataFrame({
        'node_ids': range(len(filtered_units)),
        'unit_ids': filtered_units.index.values,
        'session_id': session_id,
        'ecephys_structure_acronym': filtered_units['ecephys_structure_acronym'].values
    }).to_csv('unit_ids.{}.{}.csv'.format(area, 'valid_units' if valid_units else 'all_units'), index=False, sep=' ')


def get_structure_map(structure, valid_units=False, with_timestamps=False):
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    filter_params = {} if valid_units else {'isi_violations_maximum': np.inf, 'amplitude_cutoff_maximum': np.inf, 'presence_ratio_minimum': -np.inf}
    units = cache.get_units(**filter_params)
    
    struct_units = units[(units['has_lfp_data'] == True) & (units['session_type'] == 'brain_observatory_1.1') & (units['ecephys_structure_acronym'] == structure)]
    struct_session = struct_units[['ecephys_session_id']].value_counts()
    session_id = struct_session.index[0][0]
    cache.get_session_data(session_id)

    output_filename = 'unit_ids.{}.{}{}.csv'.format(
        structure, 
        'valid_units' if valid_units else 'all_units',
        '.with_timestamps' if with_timestamps else ''
    ) 

    filtered_units = units[(units['ecephys_session_id'] == session_id) & (units['ecephys_structure_acronym'] == structure)]
    map_df = pd.DataFrame({
        'node_ids': range(len(filtered_units)),
        'unit_ids': filtered_units.index.values,
        'session_id': session_id,
        'ecephys_structure_acronym': filtered_units['ecephys_structure_acronym'].values
    }) # .to_csv(output_filename, index=False, sep=' ')

    if with_timestamps:
        start_times = np.random.uniform(0.0, 7000.0, size=len(map_df)).astype(int)
        map_df['start_times'] = start_times.astype(float)
        map_df['stop_times'] = start_times.astype(float) + 3000.0
        map_df = map_df[['node_ids', 'unit_ids', 'start_times', 'stop_times', 'session_id', 'ecephys_structure_acronym']]
        
    map_df.to_csv(output_filename, index=False, sep=' ')


def download_neuropixel_session(session_ids=[715093703, 798911424, 754829445]):
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    
    for session_id in session_ids:
        cache.get_session_data(session_id)
    


if __name__ == '__main__':
    # get_area_map('thalamus', valid_units=True)
    # get_area_map('thalamus', valid_units=False)
    # get_area_map('hippocampus', valid_units=True)
    # get_area_map('hippocampus', valid_units=False)

    # get_structure_map('VISl', valid_units=True)
    # get_structure_map('VISl', valid_units=False)
    # get_structure_map('VISl', valid_units=True, with_timestamps=True)
    download_neuropixel_session()

