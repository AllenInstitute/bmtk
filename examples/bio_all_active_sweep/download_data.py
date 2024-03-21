from pathlib import Path
import os
import io

import xml.etree.ElementTree as ET 
import requests
import zipfile


def download_model(specimen_id, model_type='Biophysical - all active', base_dir='models', overwrite=False):
    """Uses the Allen REST API to find and download the model files for a given cell. All relevant files (parameters json and morphology swc)
    will be saved in a separate folder, the path to which will be returned.

    Notes: Using the REST API because at the moment I don't think the AllenSDK is capable of downloading parameters_fit.json files, see 
    https://community.brain-map.org/t/cell-types-database-api/3016 for more info on how to use the API.
    """
    # Set a request to get fetch model data available for a given speciment_id. It will return a xml string that needs to be parsed so
    # that we can find the correct model_id.
    api_url = f"http://api.brain-map.org/api/v2/data/query.xml?criteria=model::NeuronalModel,rma::critera,[specimen_id$eq{specimen_id}]"
    response = requests.get(api_url)
    xml_root = ET.fromstring(response.content)
    model_id = None
    for (model_name, model_id) in zip(xml_root.iter('name'), xml_root.iter('id')):
        if 'Biophysical - all active' in model_name.text:
            model_id = int(model_id.text)
            break

    if model_id is None:
        raise ValueError(f'Could not find a "{model_type}" model for cell {specimen_id}')
    
    # Now that we have the model_id for the given cell we can download and unzip the files into the correct directory. To prevent downloading
    # the zip everytime we'll check to see if the directory already exists.
    model_dir = Path(f'{base_dir}/neuronal_model_{model_id}')
    if model_dir.exists() and not overwrite:
        print(f'> {model_dir} already exits, skipping donwloadng data')
        return model_dir

    zip_uri = f'http://api.brain-map.org/neuronal_model/download/{model_id}'
    zip_response = requests.get(zip_uri)
    zip_file = zipfile.ZipFile(io.BytesIO(zip_response.content))
    zip_file.extractall(model_dir)
    return model_dir


def download_ephys_data(specimen_id, base_dir='ephys_inputs'):
    """Download nwb file containing sweeps."""
    api_url = f'http://api.brain-map.org/api/v2/data/query.xml?criteria=model::Specimen,rma::criteria,[id$eq{specimen_id}],rma::include,ephys_result(well_known_files(well_known_file_type[name$eqNWBDownload]))'
    response = requests.get(api_url)
    # print(response.content)
    download_uri = None
    xml_root = ET.fromstring(response.content)
    for dl_link in xml_root.iter('download-link'):
        download_uri = dl_link.text
        break

    ephys_id = None
    for erid in xml_root.iter('ephys-result-id'):
        ephys_id = erid.text
        break

    nwb_path = Path(f'{base_dir}/{ephys_id}_ephys.nwb')
    if nwb_path.exists():
        print(f'> {nwb_path} already exits, skipping donwload.')
        return

    if not Path(base_dir).exists():
        os.makedirs(base_dir)

    url_req = f'https://celltypes.brain-map.org/{download_uri}'
    nwb_req = requests.get(url_req, stream=True)
    with open(nwb_path, 'wb') as fh:
        for chunk in nwb_req.iter_content():
            fh.write(chunk)


def compile_mechanisms(model_dir, modfiles_dir='modfiles', overwrite=False):
    from subprocess import call
    import platform
    
    print(model_dir)
    if not Path(f'{model_dir}/{modfiles_dir}').is_dir():
        print(f'> Could not find directory {model_dir}/{modfiles_dir}, skipping compiling.')

    if Path(platform.machine()).is_dir():
        print(f'> {Path(platform.machine())} already existing, skipping compiling')

    cwd = os.getcwd()
    os.chdir(model_dir)
    call(['nrnivmodl', 'modfiles'])
    os.chdir(cwd)


if __name__ == '__main__':
    model_dir = download_model(specimen_id=488683425)
    download_ephys_data(specimen_id=488683425)
    # compile_mechanisms(model_dir)
