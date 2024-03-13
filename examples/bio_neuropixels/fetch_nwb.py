import os
import numpy as np
import pandas as pd
from pathlib import Path

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache


def download_neuropixels():
    # Get the manifest for all the sessions
    output_dir = './ecephys_cache_dir'
    manifest_path = os.path.join(output_dir, 'neuropixels.manifest.json')
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    print('{} total sessions'.format(len(sessions)))
    print(sessions.head())

    # Find sessions with units in VISl, download the first session nwb with VISl data
    mask = (sessions['session_type'] == 'brain_observatory_1.1') & (['VISl' in esa for esa in sessions['ecephys_structure_acronyms']])
    surround_sessions = sessions[mask]
    print('{} filtered sessions'.format(len(surround_sessions)))
    print(surround_sessions.head())
    session = cache.get_session_data(
        surround_sessions.index.values[0],
        isi_violations_maximum=np.inf,
        amplitude_cutoff_maximum=np.inf,
        presence_ratio_minimum=-np.inf
    )

    # Get units table
    units = cache.get_units()
    print('{} total units'.format(len(units)))
    print(units.head())


    # Get sessions_ids by number of surround sessions they have, return the session nwb with the most VISl units
    # print(units.columns)
    units_mask = (units['has_lfp_data'] == True) & (units['session_type'] == 'brain_observatory_1.1') & (units['ecephys_structure_acronym'] == 'VISl')
    visl_units = units[units_mask]
    visl_sessions = visl_units[['ecephys_session_id']].value_counts()# .rename_axis('ecephys_session_id')# .reset_index('num_units')
    session_id = visl_sessions.idxmax()[0]
    cache.get_session_data(session_id)

    """
    # Download nwb with most units in thalamus area
    units_by_region = units[['ecephys_session_id', 'ecephys_structure_acronym']]
    units_by_region = units_by_region[units_by_region['ecephys_structure_acronym'] != 'grey']
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
    units_by_region['area'] = units_by_region.apply(find_region, axis=1)
    pd.set_option('display.max_rows', None)
    units_by_area = units_by_region[['ecephys_session_id', 'area']].value_counts().rename_axis(['ecephys_session_id', 'area']).reset_index(name='counts')
    thalamus_sessions = units_by_area[units_by_area['area'] == 'thalamus']
    print(thalamus_sessions.head())
    session_id = thalamus_sessions.loc[thalamus_sessions['counts'].idxmax()]['ecephys_session_id']
    cache.get_session_data(session_id)
    """

    print(units['ecephys_structure_acronym'].value_counts())


def download_BOb():
    output_dir = './brainobs_cache_dir'
    manifest_path = str(Path(output_dir) / 'brain_observatory_manifest.json')
    boc =  BrainObservatoryCache(manifest_file=manifest_path)
    targeted_structures = boc.get_all_targeted_structures()
    print(targeted_structures)

    visp_ecs = boc.get_experiment_containers()
    experiment_id = visp_ecs[0]['id']
    exps = boc.get_ophys_experiments(experiment_container_ids=[experiment_id])
    print(exps[0]['id'])
    data_set = boc.get_ophys_experiment_data(exps[0]['id'])


def download_visbehavior():
    output_dir = "./visual_behavior_ophys_cache_dir"
    output_dir = Path(output_dir)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=output_dir)
    print(cache.list_manifest_file_names())

    behavior_sessions = cache.get_behavior_session_table()
    print(behavior_sessions.head())

    behavior_ophys_sessions = cache.get_ophys_session_table()
    print(behavior_ophys_sessions.head())

    # behavior_session = cache.get_behavior_session(behavior_session_id=870987812)
    # ophys_experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id=951980471)
    bc = bpc.VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=output_dir)
    experiment_table = bc.get_ophys_experiment_table() 
    # print(experiment_table.head())
    ophys_experiment_id = experiment_table.index[0]
    dataset = bc.get_behavior_ophys_experiment(ophys_experiment_id)

def download_visbehavior_neuropixels():
    output_dir = "./visual_behavior_neuropixels_cache_dir"
    output_dir = Path(output_dir)
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_dir)
    ecephys_session = cache.get_ecephys_session(ecephys_session_id=1052533639)


if __name__ == '__main__':
    # download_BOb()
    # download_visbehavior()
    download_visbehavior_neuropixels()
