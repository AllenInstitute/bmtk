import os
import h5py
import numpy as np
import pandas as pd
import six
import dateutil
import re
from pkg_resources import parse_version

from .cache import Cache, get_default_manifest_file
from .rma_template import RmaTemplate
from . import stimulus_info as si
from .manifest import ManifestBuilder

class NoEyeTrackingException(Exception): 
    pass


class BrainObservatoryNwbDataSet(object):
    PIPELINE_DATASET = 'brain_observatory_pipeline'
    SUPPORTED_PIPELINE_VERSION = "3.0"

    FILE_METADATA_MAPPING = {
        'age': 'general/subject/age',
        'sex': 'general/subject/sex',
        'imaging_depth': 'general/optophysiology/imaging_plane_1/imaging depth',
        'targeted_structure': 'general/optophysiology/imaging_plane_1/location',
        'ophys_experiment_id': 'general/session_id',
        'experiment_container_id': 'general/experiment_container_id',
        'device_string': 'general/devices/2-photon microscope',
        'excitation_lambda': 'general/optophysiology/imaging_plane_1/excitation_lambda',
        'indicator': 'general/optophysiology/imaging_plane_1/indicator',
        'fov': 'general/fov',
        'genotype': 'general/subject/genotype',
        'session_start_time': 'session_start_time',
        'session_type': 'general/session_type',
        'specimen_name': 'general/specimen_name',
        'generated_by': 'general/generated_by'
    }

    STIMULUS_TABLE_TYPES = {
        'abstract_feature_series': [si.DRIFTING_GRATINGS, si.STATIC_GRATINGS],
        'indexed_time_series': [si.NATURAL_SCENES, si.LOCALLY_SPARSE_NOISE,
                                si.LOCALLY_SPARSE_NOISE_4DEG, si.LOCALLY_SPARSE_NOISE_8DEG],
        'repeated_indexed_time_series':[si.NATURAL_MOVIE_ONE, si.NATURAL_MOVIE_TWO, si.NATURAL_MOVIE_THREE]

    }

    # this array was moved before file versioning was in place
    MOTION_CORRECTION_DATASETS = [ "MotionCorrection/2p_image_series/xy_translations",
                                   "MotionCorrection/2p_image_series/xy_translation" ]

    def __init__(self, nwb_file):

        self.nwb_file = nwb_file
        self.pipeline_version = None

        if os.path.exists(self.nwb_file):
            meta = self.get_metadata()
            if meta and 'pipeline_version' in meta:
                pipeline_version_str = meta['pipeline_version']
                self.pipeline_version = parse_version(pipeline_version_str)

        self._stimulus_search = None

    def get_stimulus_epoch_table(self):
        '''Returns a pandas dataframe that summarizes the stimulus epoch duration for each acquisition time index in
        the experiment

        Parameters
        ----------
        None

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        traces: 2D numpy array
            Fluorescence traces for each cell
        '''


        # These are thresholds used by get_epoch_mask_list to set a maximum limit on the delta aqusistion frames to
        #  count as different trials (rows in the stim table).  This helps account for dropped frames, so that they dont
        #  cause the cutting of an entire experiment into too many stimulus epochs.  If these thresholds are too low,
        #  the assert statment in get_epoch_mask_list will halt execution.  In that case, make a bug report!.
        threshold_dict = {si.THREE_SESSION_A:32+7,
                          si.THREE_SESSION_B:15,
                          si.THREE_SESSION_C:7,
                          si.THREE_SESSION_C2:7}

        stimulus_table_dict = {}
        for stimulus in self.list_stimuli():

            stimulus_table_dict[stimulus] = self.get_stimulus_table(stimulus)

            if stimulus == si.SPONTANEOUS_ACTIVITY:
                stimulus_table_dict[stimulus]['frame'] = 0

        interval_list = []
        interval_stimulus_dict = {}
        for stimulus in self.list_stimuli():
            stimulus_interval_list = get_epoch_mask_list(stimulus_table_dict[stimulus], threshold=threshold_dict.get(self.get_session_type(), None))
            for stimulus_interval in stimulus_interval_list:
                interval_stimulus_dict[stimulus_interval] = stimulus
            interval_list += stimulus_interval_list
        interval_list.sort(key=lambda x: x[0])

        stimulus_signature_list = ['gap']
        duration_signature_list = [int(interval_list[0][0])]
        interval_signature_list = [(0,int(interval_list[0][0]))]
        for ii, interval in enumerate(interval_list):
            stimulus_signature_list.append(interval_stimulus_dict[interval])
            duration_signature_list.append(int(interval[1] - interval[0]))
            interval_signature_list.append((int(interval[0]), int(interval[1])))

            if ii != len(interval_list)-1:
                stimulus_signature_list.append('gap')
                duration_signature_list.append((int(interval_list[ii+1][0] - interval_list[ii][1])))
                interval_signature_list.append((int(interval_list[ii][1]), int(interval_list[ii+1][0])))

        stimulus_signature_list.append('gap')
        interval_signature_list.append((int(interval_list[-1][1]), len(self.get_fluorescence_timestamps())))
        duration_signature_list.append(interval_signature_list[-1][1]-interval_signature_list[-1][0])

        interval_df = pd.DataFrame({'stimulus':stimulus_signature_list,
                                    'duration':duration_signature_list,
                                    'interval':interval_signature_list})

        # Gaps are uninformative; remove them:
        interval_df = interval_df[interval_df.stimulus != 'gap']
        interval_df['start'] = [x[0] for x in interval_df['interval'].values]
        interval_df['end'] = [x[1] for x in interval_df['interval'].values]

        interval_df.reset_index(inplace=True, drop=True)
        interval_df.drop(['interval', 'duration'], axis=1, inplace=True)
        return interval_df


    def get_fluorescence_traces(self, cell_specimen_ids=None):
        ''' Returns an array of fluorescence traces for all ROI and
        the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        traces: 2D numpy array
            Fluorescence traces for each cell
        '''
        timestamps = self.get_fluorescence_timestamps()
        with h5py.File(self.nwb_file, 'r') as f:
            ds = f['processing'][self.PIPELINE_DATASET][
                'Fluorescence']['imaging_plane_1']['data']

            if cell_specimen_ids is None:
                cell_traces = ds[()]
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                cell_traces = ds[inds, :]

        return timestamps, cell_traces

    def get_fluorescence_timestamps(self):
        ''' Returns an array of timestamps in seconds for the fluorescence traces '''

        with h5py.File(self.nwb_file, 'r') as f:
            timestamps = f['processing'][self.PIPELINE_DATASET][
                'Fluorescence']['imaging_plane_1']['timestamps'][()]
        return timestamps

    def get_neuropil_traces(self, cell_specimen_ids=None):
        ''' Returns an array of neuropil fluorescence traces for all ROIs
        and the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        traces: 2D numpy array
            Neuropil fluorescence traces for each cell
        '''

        timestamps = self.get_fluorescence_timestamps()

        with h5py.File(self.nwb_file, 'r') as f:
            if self.pipeline_version >= parse_version("2.0"):
                ds = f['processing'][self.PIPELINE_DATASET][
                    'Fluorescence']['imaging_plane_1_neuropil_response']['data']
            else:
                ds = f['processing'][self.PIPELINE_DATASET][
                    'Fluorescence']['imaging_plane_1']['neuropil_traces']

            if cell_specimen_ids is None:
                np_traces = ds[()]
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                np_traces = ds[inds, :]

        return timestamps, np_traces


    def get_neuropil_r(self, cell_specimen_ids=None):
        ''' Returns a scalar value of r for neuropil correction of flourescence traces

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then results for all are returned

        Returns
        -------
        r: 1D numpy array, len(r)=len(cell_specimen_ids)
            Scalar for neuropil subtraction for each cell
        '''

        with h5py.File(self.nwb_file, 'r') as f:
            if self.pipeline_version >= parse_version("2.0"):
                r_ds = f['processing'][self.PIPELINE_DATASET][
                    'Fluorescence']['imaging_plane_1_neuropil_response']['r']
            else:
                r_ds = f['processing'][self.PIPELINE_DATASET][
                    'Fluorescence']['imaging_plane_1']['r']

            if cell_specimen_ids is None:
                r = r_ds[()]
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                r = r_ds[inds]

        return r

    def get_demixed_traces(self, cell_specimen_ids=None):
        ''' Returns an array of demixed fluorescence traces for all ROIs
        and the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        traces: 2D numpy array
            Demixed fluorescence traces for each cell
        '''

        timestamps = self.get_fluorescence_timestamps()

        with h5py.File(self.nwb_file, 'r') as f:
            ds = f['processing'][self.PIPELINE_DATASET][
                'Fluorescence']['imaging_plane_1_demixed_signal']['data']
            if cell_specimen_ids is None:
                traces = ds[()]
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                traces = ds[inds, :]

        return timestamps, traces

    def get_corrected_fluorescence_traces(self, cell_specimen_ids=None):
        ''' Returns an array of demixed and neuropil-corrected fluorescence traces
        for all ROIs and the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        traces: 2D numpy array
            Corrected fluorescence traces for each cell
        '''

        # starting in version 2.0, neuropil correction follows trace demixing
        if self.pipeline_version >= parse_version("2.0"):
            timestamps, cell_traces = self.get_demixed_traces(cell_specimen_ids)
        else:
            timestamps, cell_traces = self.get_fluorescence_traces(cell_specimen_ids)

        r = self.get_neuropil_r(cell_specimen_ids)

        _, neuropil_traces = self.get_neuropil_traces(cell_specimen_ids)

        fc = cell_traces - neuropil_traces * r[:, np.newaxis]

        return timestamps, fc

    def get_cell_specimen_indices(self, cell_specimen_ids):
        ''' Given a list of cell specimen ids, return their index based on their order in this file.

        Parameters
        ----------
        cell_specimen_ids: list of cell specimen ids

        '''

        all_cell_specimen_ids = list(self.get_cell_specimen_ids())
        
        try:
            inds = [list(all_cell_specimen_ids).index(i)
                    for i in cell_specimen_ids]
        except ValueError as e:
            raise ValueError("Cell specimen not found (%s)" % str(e))

        return inds

    def get_dff_traces(self, cell_specimen_ids=None):
        ''' Returns an array of dF/F traces for all ROIs and
        the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return data for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        dF/F: 2D numpy array
            dF/F values for each cell
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            dff_ds = f['processing'][self.PIPELINE_DATASET][
                'DfOverF']['imaging_plane_1']

            timestamps = dff_ds['timestamps'][()]

            if cell_specimen_ids is None:
                cell_traces = dff_ds['data'][()]
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                cell_traces = dff_ds['data'][inds, :]

        return timestamps, cell_traces

    def get_roi_ids(self):
        ''' Returns an array of IDs for all ROIs in the file

        Returns
        -------
        ROI IDs: list
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            roi_id = f['processing'][self.PIPELINE_DATASET][
                'ImageSegmentation']['roi_ids'][()]
        return roi_id

    def get_cell_specimen_ids(self):
        ''' Returns an array of cell IDs for all cells in the file

        Returns
        -------
        cell specimen IDs: list
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            cell_id = f['processing'][self.PIPELINE_DATASET][
                'ImageSegmentation']['cell_specimen_ids'][()]
        return cell_id

    def get_session_type(self):
        ''' Returns the type of experimental session, presently one of the
        following: three_session_A, three_session_B, three_session_C

        Returns
        -------
        session type: string
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            session_type = f['general/session_type'][()]
        return session_type.decode('utf-8')

    def get_max_projection(self):
        '''Returns the maximum projection image for the 2P movie.

        Returns
        -------
        max projection: np.ndarray
        '''

        with h5py.File(self.nwb_file, 'r') as f:
            max_projection = f['processing'][self.PIPELINE_DATASET]['ImageSegmentation'][
                'imaging_plane_1']['reference_images']['maximum_intensity_projection_image']['data'][()]
        return max_projection

    def list_stimuli(self):
        ''' Return a list of the stimuli presented in the experiment.

        Returns
        -------
        stimuli: list of strings
        '''

        with h5py.File(self.nwb_file, 'r') as f:
            keys = list(f["stimulus/presentation/"].keys())
        return [ k.replace('_stimulus', '') for k in keys ]


    def _get_master_stimulus_table(self):
        ''' Builds a table for all stimuli by concatenating (vertically) the 
        sub-tables describing presentation of each stimulus
        '''

        epoch_table = self.get_stimulus_epoch_table()

        stimulus_table_dict = {}
        for stimulus in self.list_stimuli():
            stimulus_table_dict[stimulus] = self.get_stimulus_table(stimulus)

        table_list = []
        for stimulus in self.list_stimuli():
            curr_stimtable = stimulus_table_dict[stimulus]

            for _, row in epoch_table[epoch_table['stimulus'] == stimulus].iterrows():

                epoch_start_ind, epoch_end_ind = row['start'], row['end']
                curr_subtable = curr_stimtable[(epoch_start_ind <= curr_stimtable['start']) &
                                                (curr_stimtable['end'] <= epoch_end_ind)].copy()
                curr_subtable['stimulus'] = stimulus
                table_list.append(curr_subtable)

        new_table = pd.concat(table_list, sort=True)
        new_table.reset_index(drop=True, inplace=True)

        return new_table


    def get_stimulus_table(self, stimulus_name):
        ''' Return a stimulus table given a stimulus name 
        
        Notes
        -----
        For more information, see:
        http://help.brain-map.org/display/observatory/Documentation?preview=/10616846/10813485/VisualCoding_VisualStimuli.pdf 

        '''

        if stimulus_name == 'master':
            return self._get_master_stimulus_table()

        with h5py.File(self.nwb_file, 'r') as nwb_file:

            stimulus_group = _find_stimulus_presentation_group(nwb_file, stimulus_name)

            if stimulus_name in self.STIMULUS_TABLE_TYPES['abstract_feature_series']:
                datasets = h5_utilities.load_datasets_by_relnames(
                    ['data', 'features', 'frame_duration'], nwb_file, stimulus_group)
                return _make_abstract_feature_series_stimulus_table(
                    datasets['data'], h5_utilities.decode_bytes(datasets['features']), datasets['frame_duration'])

            if stimulus_name in self.STIMULUS_TABLE_TYPES['indexed_time_series']:
                datasets = h5_utilities.load_datasets_by_relnames(['data', 'frame_duration'], nwb_file, stimulus_group)
                return _make_indexed_time_series_stimulus_table(datasets['data'], datasets['frame_duration'])

            if stimulus_name in self.STIMULUS_TABLE_TYPES['repeated_indexed_time_series']:
                datasets = h5_utilities.load_datasets_by_relnames(['data', 'frame_duration'], nwb_file, stimulus_group)
                return _make_repeated_indexed_time_series_stimulus_table(datasets['data'], datasets['frame_duration'])

            if stimulus_name == 'spontaneous':
                datasets = h5_utilities.load_datasets_by_relnames(['data', 'frame_duration'], nwb_file, stimulus_group)
                return _make_spontaneous_activity_stimulus_table(datasets['data'], datasets['frame_duration'])

        raise IOError("Could not find a stimulus table named '%s'" % stimulus_name)
                

    # @memoize
    def get_stimulus_template(self, stimulus_name):
        ''' Return an array of the stimulus template for the specified stimulus.

        Parameters
        ----------
        stimulus_name: string
            Must be one of the strings returned by list_stimuli().

        Returns
        -------
        stimulus table: pd.DataFrame
        '''
        stim_name = stimulus_name + "_image_stack"
        with h5py.File(self.nwb_file, 'r') as f:
            image_stack = f['stimulus']['templates'][stim_name]['data'][()]
        return image_stack

    def get_locally_sparse_noise_stimulus_template(self,
                                                   stimulus,
                                                   mask_off_screen=True):
        ''' Return an array of the stimulus template for the specified stimulus.

        Parameters
        ----------
        stimulus: string
           Which locally sparse noise stimulus to retrieve.  Must be one of:
               stimulus_info.LOCALLY_SPARSE_NOISE
               stimulus_info.LOCALLY_SPARSE_NOISE_4DEG
               stimulus_info.LOCALLY_SPARSE_NOISE_8DEG

        mask_off_screen: boolean
           Set off-screen regions of the stimulus to LocallySparseNoise.LSN_OFF_SCREEN.

        Returns
        -------
        tuple: (template, off-screen mask)
        '''

        if stimulus not in si.LOCALLY_SPARSE_NOISE_DIMENSIONS:
            raise KeyError("%s is not a known locally sparse noise stimulus" % stimulus)

        template = self.get_stimulus_template(stimulus)

        # build mapping from template coordinates to display coordinates
        template_shape = si.LOCALLY_SPARSE_NOISE_DIMENSIONS[stimulus]
        template_shape = [ template_shape[1], template_shape[0] ]

        template_display_shape = (1260, 720)
        display_shape = (1920, 1200)

        scale = [
            float(template_shape[0]) / float(template_display_shape[0]),
            float(template_shape[1]) / float(template_display_shape[1])
        ]
        offset = [
            -(display_shape[0] - template_display_shape[0]) * 0.5,
            -(display_shape[1] - template_display_shape[1]) * 0.5
        ]

        x, y = np.meshgrid(np.arange(display_shape[0]), np.arange(
            display_shape[1]), indexing='ij')
        template_display_coords = np.array([(x + offset[0]) * scale[0] - 0.5,
                                            (y + offset[1]) * scale[1] - 0.5],
                                           dtype=float)
        template_display_coords = np.rint(template_display_coords).astype(int)

        # build mask
        template_mask, template_frac = si_mask_stimulus_template(
            template_display_coords, template_shape)

        if mask_off_screen:
            template[:, ~template_mask.T] = LocallySparseNoise.LSN_OFF_SCREEN

        return template, template_mask.T

    def get_roi_mask_array(self, cell_specimen_ids=None):
        ''' Return a numpy array containing all of the ROI masks for requested cells.
        If cell_specimen_ids is omitted, return all masks.

        Parameters
        ----------
        cell_specimen_ids: list
            List of cell specimen ids.  Default None.

        Returns
        -------
        np.ndarray: NxWxH array, where N is number of cells
        '''

        roi_masks = self.get_roi_mask(cell_specimen_ids)

        if len(roi_masks) == 0:
            raise IOError("no masks found for given cell specimen ids")

        roi_arr = roi.create_roi_mask_array(roi_masks)

        return roi_arr

    def get_roi_mask(self, cell_specimen_ids=None):
        ''' Returns an array of all the ROI masks

        Parameters
        ----------
        cell specimen IDs: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
            List of ROI_Mask objects
        '''

        with h5py.File(self.nwb_file, 'r') as f:
            mask_loc = f['processing'][self.PIPELINE_DATASET][
                'ImageSegmentation']['imaging_plane_1']
            roi_list = f['processing'][self.PIPELINE_DATASET][
                'ImageSegmentation']['imaging_plane_1']['roi_list'][()]

            inds = None
            if cell_specimen_ids is None:
                inds = range(self.number_of_cells)
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)

            roi_array = []
            for i in inds:
                v = roi_list[i]
                roi_mask = mask_loc[v]["img_mask"][()]
                m = roi.create_roi_mask(roi_mask.shape[1], roi_mask.shape[0],
                                        [0, 0, 0, 0], roi_mask=roi_mask, label=v)
                roi_array.append(m)

        return roi_array

    @property
    def number_of_cells(self):
        '''Number of cells in the experiment'''

        # Replace here is there is a better way to get this info:
        return len(self.get_cell_specimen_ids())


    def get_metadata(self):
        ''' Returns a dictionary of meta data associated with each
        experiment, including Cre line, specimen number,
        visual area imaged, imaging depth

        Returns
        -------
        metadata: dictionary
        '''

        meta = {}

        with h5py.File(self.nwb_file, 'r') as f:
            for memory_key, disk_key in BrainObservatoryNwbDataSet.FILE_METADATA_MAPPING.items():
                try:
                    v = f[disk_key][()]

                    # convert numpy strings to python strings
                    if v.dtype.type is np.bytes_:
                        if len(v.shape) == 0:
                            v = v.decode('UTF-8')
                        elif len(v.shape) == 1:
                            v = [ s.decode('UTF-8') for s in v ]
                        else:
                            raise Exception("Unrecognized metadata formatting for field %s" % disk_key)

                    meta[memory_key] = v
                except KeyError as e:
                    logging.warning("could not find key %s", disk_key)

        # extract cre line from genotype string
        genotype = meta.get('genotype')
        meta['cre_line'] = meta['genotype'].split(';')[0] if genotype else None

        imaging_depth = meta.pop('imaging_depth', None)
        meta['imaging_depth_um'] = int(imaging_depth.split()[0]) if imaging_depth else None

        ophys_experiment_id = meta.get('ophys_experiment_id')
        meta['ophys_experiment_id'] = int(ophys_experiment_id) if ophys_experiment_id else None

        experiment_container_id = meta.get('experiment_container_id')
        meta['experiment_container_id'] = int(experiment_container_id) if experiment_container_id else None

        # convert start time to a date object
        session_start_time = meta.get('session_start_time')
        if isinstance( session_start_time, six.string_types ):
            meta['session_start_time'] = dateutil.parser.parse(session_start_time)

        age = meta.pop('age', None)
        if age:
            # parse the age in days
            m = re.match("(.*?) days", age)
            if m:
                meta['age_days'] = int(m.groups()[0])
            else:
                raise IOError("Could not parse age.")


        # parse the device string (ugly, sorry)
        device_string = meta.pop('device_string', None)
        if device_string:
            m = re.match("(.*?)\.\s(.*?)\sPlease*", device_string)
            if m:
                device, device_name = m.groups()
                meta['device'] = device
                meta['device_name'] = device_name
            else:
                raise IOError("Could not parse device string.")

        # file version
        generated_by = meta.pop('generated_by', None)
        version = generated_by[-1] if generated_by else "0.9"
        meta["pipeline_version"] = version

        return meta

    def get_running_speed(self):
        ''' Returns the mouse running speed in cm/s
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            dx_ds = f['processing'][self.PIPELINE_DATASET][
                'BehavioralTimeSeries']['running_speed']
            dxcm = dx_ds['data'][()]
            dxtime = dx_ds['timestamps'][()]

        timestamps = self.get_fluorescence_timestamps()

        # v0.9 stored this as an Nx1 array instead of a flat 1-d array
        if len(dxcm.shape) == 2:
            dxcm = dxcm[:, 0]

        dxcm, dxtime = align_running_speed(dxcm, dxtime, timestamps)

        return dxcm, dxtime

    def get_pupil_location(self, as_spherical=True):
        '''Returns the x, y pupil location.

        Parameters
        ----------
        as_spherical : bool
            Whether to return the location as spherical (default) or
            not. If true, the result is altitude and azimuth in
            degrees, otherwise it is x, y in centimeters. (0,0) is
            the center of the monitor.

        Returns
        -------
        (timestamps, location)
            Timestamps is an (Nx1) array of timestamps in seconds.
            Location is an (Nx2) array of spatial location.
        '''
        if as_spherical:
            location_key = "pupil_location_spherical"
        else:
            location_key = "pupil_location"
        try:
            with h5py.File(self.nwb_file, 'r') as f:
                eye_tracking = f['processing'][self.PIPELINE_DATASET][
                    'EyeTracking'][location_key]
                pupil_location = eye_tracking['data'][()]
                pupil_times = eye_tracking['timestamps'][()]
        except KeyError:
            raise NoEyeTrackingException("No eye tracking for this experiment.")

        return pupil_times, pupil_location

    def get_pupil_size(self):
        '''Returns the pupil area in pixels.

        Returns
        -------
        (timestamps, areas)
            Timestamps is an (Nx1) array of timestamps in seconds.
            Areas is an (Nx1) array of pupil areas in pixels.
        '''
        try:
            with h5py.File(self.nwb_file, 'r') as f:
                pupil_tracking = f['processing'][self.PIPELINE_DATASET][
                    'PupilTracking']['pupil_size']
                pupil_size = pupil_tracking['data'][()]
                pupil_times = pupil_tracking['timestamps'][()]
        except KeyError:
            raise NoEyeTrackingException("No pupil tracking for this experiment.")

        return pupil_times, pupil_size

    def get_motion_correction(self):
        ''' Returns a Panda DataFrame containing the x- and y- translation of each image used for image alignment
        '''

        motion_correction = None
        with h5py.File(self.nwb_file, 'r') as f:
            pipeline_ds = f['processing'][self.PIPELINE_DATASET]

            # pipeline 0.9 stores this in xy_translations
            # pipeline 1.0 stores this in xy_translation
            for mc_ds_name in self.MOTION_CORRECTION_DATASETS:
                try:
                    mc_ds = pipeline_ds[mc_ds_name]

                    motion_log = mc_ds['data'][()]
                    motion_time = mc_ds['timestamps'][()]
                    motion_names = mc_ds['feature_description'][()]

                    motion_correction = pd.DataFrame(motion_log, columns=motion_names)
                    motion_correction['timestamp'] = motion_time

                    # break out if we found it
                    break
                except KeyError as e:
                    pass

        if motion_correction is None:
            raise KeyError("Could not find motion correction data.")

        # Python3 compatibility:
        rename_dict = {}
        for c in motion_correction.columns:
            if not isinstance(c, str):
                rename_dict[c] = c.decode("utf-8")
        motion_correction.rename(columns=rename_dict, inplace=True)

        return motion_correction

    def save_analysis_dataframes(self, *tables):
        store = pd.HDFStore(self.nwb_file, mode='a')
        for k, v in tables:
            store.put('analysis/%s' % (k), v)
        store.close()

    def save_analysis_arrays(self, *datasets):
        with h5py.File(self.nwb_file, 'a') as f:
            for k, v in datasets:
                if k in f['analysis']:
                    del f['analysis'][k]
                f.create_dataset('analysis/%s' % k, data=v)

    @property
    def stimulus_search(self):

        if self._stimulus_search is None:
            self._stimulus_search = si.StimulusSearch(self)
        return self._stimulus_search

    def get_stimulus(self, frame_ind):

        search_result = self.stimulus_search.search(frame_ind)

        if search_result is None or search_result[2]['stimulus'] == si.SPONTANEOUS_ACTIVITY:
            return None, None

        else:

            curr_stimulus = search_result[2]['stimulus']
            if curr_stimulus in si.LOCALLY_SPARSE_NOISE_STIMULUS_TYPES + si.NATURAL_MOVIE_STIMULUS_TYPES + [si.NATURAL_SCENES]:
                curr_frame = search_result[2]['frame']
                return search_result, self.get_stimulus_template(curr_stimulus)[int(curr_frame), :, :]
            elif curr_stimulus == si.STATIC_GRATINGS or curr_stimulus == si.DRIFTING_GRATINGS:
                return search_result, None




class BrainObservatoryApi(RmaTemplate):

    OPHYS_EVENTS_FILE_TYPE = "ObservatoryEventsFile"
    NWB_FILE_TYPE = "NWBOphys"
    OPHYS_ANALYSIS_FILE_TYPE = "OphysExperimentCellRoiMetricsFile"

    rma_templates = {
        "brain_observatory_queries": [
            {
                "name": "list_isi_experiments",
                "description": "see name",
                "model": "IsiExperiment",
                "num_rows": "all",
                "count": False,
                "criteria_params": [],
            },
            {
                "name": "isi_experiment_by_ids",
                "description": "see name",
                "model": "IsiExperiment",
                "criteria": "[id$in{{ isi_experiment_ids }}]",
                "include": "experiment_container(ophys_experiments,targeted_structure)",    # noqa e501
                "num_rows": "all",
                "count": False,
                "criteria_params": ["isi_experiment_ids"],
            },
            {
                "name": "ophys_experiment_by_ids",
                "description": "see name",
                "model": "OphysExperiment",
                "criteria": "{% if ophys_experiment_ids is defined %}[id$in{{ ophys_experiment_ids }}]{%endif%}",   # noqa e501
                "include": "experiment_container,well_known_files(well_known_file_type),targeted_structure,specimen(donor(age,transgenic_lines))",  # noqa e501
                "num_rows": "all",
                "count": False,
                "criteria_params": ["ophys_experiment_ids"],
            },
            {
                "name": "ophys_experiment_data",
                "description": "see name",
                "model": "WellKnownFile",
                "criteria": "[attachable_id$eq{{ ophys_experiment_id }}],well_known_file_type[name$eq%s]"   # noqa e501
                % NWB_FILE_TYPE,
                "num_rows": "all",
                "count": False,
                "criteria_params": ["ophys_experiment_id"],
            },
            {
                "name": "ophys_analysis_file",
                "description": "see name",
                "model": "WellKnownFile",
                "criteria": "[attachable_id$eq{{ ophys_experiment_id }}],well_known_file_type[name$eq%s]"   # noqa e501
                % OPHYS_ANALYSIS_FILE_TYPE,
                "num_rows": "all",
                "count": False,
                "criteria_params": ["ophys_experiment_id"],
            },
            {
                "name": "ophys_events_file",
                "description": "see name",
                "model": "WellKnownFile",
                "criteria": "[attachable_id$eq{{ ophys_experiment_id }}],well_known_file_type[name$eq%s]"   # noqa e501
                % OPHYS_EVENTS_FILE_TYPE,
                "num_rows": "all",
                "count": False,
                "criteria_params": ["ophys_experiment_id"],
            },
            {
                "name": "column_definitions",
                "description": "see name",
                "model": "ApiColumnDefinition",
                "criteria": "[api_class_name$eq{{ api_class_name }}]",
                "num_rows": "all",
                "count": False,
                "criteria_params": ["api_class_name"],
            },
            {
                "name": "column_definition_class_names",
                "description": "see name",
                "model": "ApiColumnDefinition",
                "only": ["api_class_name"],
                "num_rows": "all",
                "count": False,
            },
            {
                "name": "stimulus_mapping",
                "description": "see name",
                "model": "ApiCamStimulusMapping",
                "criteria": "{% if stimulus_mapping_ids is defined %}[id$in{{ stimulus_mapping_ids }}]{%endif%}",   # noqa e501
                "num_rows": "all",
                "count": False,
                "criteria_params": ["stimulus_mapping_ids"],
            },
            {
                "name": "experiment_container",
                "description": "see name",
                "model": "ExperimentContainer",
                "criteria": "{% if experiment_container_ids is defined %}[id$in{{ experiment_container_ids }}]{%endif%}",   # noqa e501
                "include": "ophys_experiments,isi_experiment,specimen(donor(conditions,age,transgenic_lines)),targeted_structure",  # noqa e501
                "num_rows": "all",
                "count": False,
                "criteria_params": ["experiment_container_ids"],
            },
            {
                "name": "experiment_container_metric",
                "description": "see name",
                "model": "ApiCamExperimentContainerMetric",
                "criteria": "{% if experiment_container_metric_ids is defined %}[id$in{{ experiment_container_metric_ids }}]{%endif%}", # noqa e501
                "num_rows": "all",
                "count": False,
                "criteria_params": ["experiment_container_metric_ids"],
            },
            {
                "name": "cell_metric",
                "description": "see name",
                "model": "ApiCamCellMetric",
                "criteria": "{% if cell_specimen_ids is defined %}[cell_specimen_id$in{{ cell_specimen_ids }}]{%endif%}",   # noqa e501
                "criteria_params": ["cell_specimen_ids"],
            },
            {
                "name": "cell_specimen_id_mapping_table",
                "description": "see name",
                "model": "WellKnownFile",
                "criteria": "[id$eq{{ mapping_table_id }}],well_known_file_type[name$eqOphysCellSpecimenIdMapping]",    # noqa e501
                "num_rows": "all",
                "count": False,
                "criteria_params": ["mapping_table_id"],
            },
            {
                "name": "eye_gaze_mapping_file",
                "description": "h5 file containing mouse eye gaze mapped onto screen coordinates (as well as pupil and eye sizes)", # noqa e501
                "model": "WellKnownFile",
                "criteria": "[attachable_id$eq{{ ophys_session_id }}],well_known_file_type[name$eqEyeDlcScreenMapping]",    # noqa e501
                "num_rows": "all",
                "count": False,
                "criteria_params": ["ophys_session_id"],
            },
            # NOTE: 'all_eye_mapping_files' query is for facilitating an ugly
            # hack to get around lack of relationship between experiment id
            # and session id in current warehouse. This should be removed when
            # the relationship is added.
            {
                "name": "all_eye_mapping_files",
                "description": "Get a list of dictionaries for all eye mapping wkfs",   # noqa e501
                "model": "WellKnownFile",
                "criteria": "well_known_file_type[name$eqEyeDlcScreenMapping]",
                "num_rows": "all",
                "count": False,
            },
        ]
    }

    def __init__(self, base_uri=None, datacube_uri=None):
        super(BrainObservatoryApi, self).__init__(
            base_uri, query_manifest=BrainObservatoryApi.rma_templates
        )

        self.datacube_uri = datacube_uri

    def save_ophys_experiment_data(self, ophys_experiment_id, file_name):
        data = self.template_query(
            "brain_observatory_queries",
            "ophys_experiment_data",
            ophys_experiment_id=ophys_experiment_id,
        )

        try:
            file_url = data[0]["download_link"]
        except Exception:
            raise Exception(
                "ophys experiment %d has no data file" % ophys_experiment_id
            )

        # self._log.warning(
        #     "Downloading ophys_experiment %d NWB. This can take some time."
        #     % ophys_experiment_id
        # )

        self.retrieve_file_over_http(self.api_url + file_url, file_name)



class BrainObservatoryCache(Cache):
    EXPERIMENT_CONTAINERS_KEY = "EXPERIMENT_CONTAINERS"
    EXPERIMENTS_KEY = "EXPERIMENTS"
    CELL_SPECIMENS_KEY = "CELL_SPECIMENS"
    EXPERIMENT_DATA_KEY = "EXPERIMENT_DATA"
    ANALYSIS_DATA_KEY = "ANALYSIS_DATA"
    EVENTS_DATA_KEY = "EVENTS_DATA"
    STIMULUS_MAPPINGS_KEY = "STIMULUS_MAPPINGS"
    EYE_GAZE_DATA_KEY = "EYE_GAZE_DATA"
    MANIFEST_VERSION = "1.3"

    def __init__(self, cache=True, manifest_file=None, base_uri=None, api=None):

        if manifest_file is None:
            manifest_file = get_default_manifest_file("brain_observatory")

        super(BrainObservatoryCache, self).__init__(
            manifest=manifest_file, cache=cache, version=self.MANIFEST_VERSION
        )

        if api is None:
            self.api = BrainObservatoryApi(base_uri=base_uri)
        else:
            self.api = api


    def get_ophys_experiment_data(self, ophys_experiment_id, file_name=None):
        """Download the NWB file for an ophys_experiment (if it hasn't
        already been
        downloaded) and return a data accessor object.

        Parameters
        ----------
        file_name: string
            File name to save/read the data set.  If file_name is None,
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.

        ophys_experiment_id: integer
            id of the ophys_experiment to retrieve

        Returns
        -------
        BrainObservatoryNwbDataSet
        """
        file_name = self.get_cache_path(
            file_name, self.EXPERIMENT_DATA_KEY, ophys_experiment_id
        )

        self.api.save_ophys_experiment_data(
            ophys_experiment_id, file_name
        )

        return BrainObservatoryNwbDataSet(file_name)

    def build_manifest(self, file_name):
        """
        Construct a manifest for this Cache class and save it in a file.

        Parameters
        ----------

        file_name: string
            File location to save the manifest.

        """

        mb = ManifestBuilder()
        mb.set_version(self.MANIFEST_VERSION)
        mb.add_path("BASEDIR", ".")
        mb.add_path(
            self.EXPERIMENT_CONTAINERS_KEY,
            "experiment_containers.json",
            typename="file",
            parent_key="BASEDIR",
        )
        mb.add_path(
            self.EXPERIMENTS_KEY,
            "ophys_experiments.json",
            typename="file",
            parent_key="BASEDIR",
        )
        mb.add_path(
            self.EXPERIMENT_DATA_KEY,
            "ophys_experiment_data/%d.nwb",
            typename="file",
            parent_key="BASEDIR",
        )
        mb.add_path(
            self.ANALYSIS_DATA_KEY,
            "ophys_experiment_analysis/%d_%s_analysis.h5",
            typename="file",
            parent_key="BASEDIR",
        )
        mb.add_path(
            self.EVENTS_DATA_KEY,
            "ophys_experiment_events/%d_events.npz",
            typename="file",
            parent_key="BASEDIR",
        )
        mb.add_path(
            self.CELL_SPECIMENS_KEY,
            "cell_specimens.json",
            typename="file",
            parent_key="BASEDIR",
        )
        mb.add_path(
            self.STIMULUS_MAPPINGS_KEY,
            "stimulus_mappings.json",
            typename="file",
            parent_key="BASEDIR",
        )
        mb.add_path(
            self.EYE_GAZE_DATA_KEY,
            "ophys_eye_gaze_mapping/%d_eyetracking_dlc_to_screen_mapping.h5",
            typename="file",
            parent_key="BASEDIR",
        )

        mb.write_json_file(file_name)
