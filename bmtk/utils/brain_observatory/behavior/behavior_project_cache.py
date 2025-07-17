import pandas as pd
import pynwb

from .cloud_cache import S3CloudCache


INT_NULL = -99

INTEGER_COLUMNS = [
    "prior_exposures_to_image_set",
    "ecephys_session_id",
    "unit_count",
    "probe_count",
    "channel_count",
]


class ProjectCacheBase:
    BUCKET_NAME=None
    PROJECT_NAME=None

    def __init__(self, fetch_api, fetch_tries=2):
        self.fetch_api = fetch_api
        self.cache = None

        self.fetch_tries = fetch_tries
        # self.logger = logging.getLogger(self.__class__.__name__)


    @classmethod
    def from_s3_cache(cls, cache_dir, bucket_name_override=None):
        fetch_api = cls.cloud_api_class().from_s3_cache(
            cache_dir,
            bucket_name=(
                bucket_name_override if bucket_name_override is not None
                else cls.BUCKET_NAME),
            project_name=cls.PROJECT_NAME,
            ui_class_name=cls.__name__)

        return cls(fetch_api=fetch_api)

    @classmethod
    def cloud_api_class(cls):
        return VisualBehaviorNeuropixelsProjectCloudApi





class ProjectCloudApiBase:
    def __init__(self, cache, skip_version_check=False, local=False):
        self.cache = cache
        self.skip_version_check = skip_version_check
        self._local = local
        self.load_manifest()

    @classmethod
    def from_s3_cache(cls, cache_dir, bucket_name, project_name, ui_class_name):
        cache = S3CloudCache(cache_dir,
                             bucket_name,
                             project_name,
                             ui_class_name=ui_class_name)
        return cls(cache)


    def load_manifest(self, manifest_name=None):
        """
        Load the specified manifest file into the CloudCache

        Parameters
        ----------
        manifest_name: Optional[str]
            Name of manifest file to load. If None, load latest
            (default: None)
        """
        if manifest_name is None:
            self.cache.load_last_manifest()
        else:
            self.cache.load_manifest(manifest_name)

        if self.cache._manifest.metadata_file_names is None:
            raise RuntimeError(f"{type(self.cache)} object has no metadata "
                               f"file names. Check contents of the loaded "
                               f"manifest file: {self.cache._manifest_name}")

        if not self.skip_version_check:
            data_sdk_version = [i for i in self.cache._manifest._data_pipeline
                                if i['name'] == "AllenSDK"][0]["version"]
            version_check(
                self.cache._manifest.version,
                data_sdk_version,
                cmin=self.MANIFEST_COMPATIBILITY[0],
                cmax=self.MANIFEST_COMPATIBILITY[1])

        # self.logger = logging.getLogger(self.__class__.__name__)

        self._load_manifest_tables()

    def _get_metadata_path(self, fname: str):
        if self._local:
            path = self._get_local_path(fname=fname)
        else:
            path = self.cache.download_metadata(fname=fname)
        return path

    def _get_data_path(self, file_id: str):
        if self._local:
            data_path = self._get_local_path(file_id=file_id)
        else:
            data_path = self.cache.download_data(file_id=file_id)
        return data_path




class BehaviorEcephysSession:
    pass

    # @classmethod
    # def from_nwb(
    #     cls,
    #     nwbfile: NWBFile,
    #     add_is_change_to_stimulus_presentations_table=True,
    #     eye_tracking_z_threshold: float = 3.0,
    #     eye_tracking_dilation_frames: int = 2,
    # ) -> "BehaviorSession":
    #     """

    #     Parameters
    #     ----------
    #     nwbfile
    #     add_is_change_to_stimulus_presentations_table: Whether to add a column
    #         denoting whether the stimulus presentation represented a change
    #         event. May not be needed in case this column is precomputed
    #     eye_tracking_z_threshold : float, optional
    #         The z-threshold when determining which frames likely contain
    #         outliers for eye or pupil areas. Influences which frames
    #         are considered 'likely blinks'. By default 3.0
    #     eye_tracking_dilation_frames : int, optional
    #         Determines the number of adjacent frames that will be marked
    #         as 'likely_blink' when performing blink detection for
    #         `eye_tracking` data, by default 2

    #     Returns
    #     -------

    #     """
    #     behavior_session_id = BehaviorSessionId.from_nwb(nwbfile)
    #     stimulus_timestamps = StimulusTimestamps.from_nwb(nwbfile)
    #     running_acquisition = RunningAcquisition.from_nwb(nwbfile)
    #     raw_running_speed = RunningSpeed.from_nwb(nwbfile, filtered=False)
    #     running_speed = RunningSpeed.from_nwb(nwbfile)
    #     metadata = BehaviorMetadata.from_nwb(nwbfile)
    #     licks = Licks.from_nwb(nwbfile=nwbfile)
    #     rewards = Rewards.from_nwb(nwbfile=nwbfile)
    #     stimuli = Stimuli.from_nwb(
    #         nwbfile=nwbfile,
    #         add_is_change_to_presentations_table=(
    #             add_is_change_to_stimulus_presentations_table
    #         ),
    #     )
    #     task_parameters = TaskParameters.from_nwb(nwbfile=nwbfile)
    #     trials = cls._trials_class().from_nwb(nwbfile=nwbfile)
    #     date_of_acquisition = DateOfAcquisition.from_nwb(nwbfile=nwbfile)

    #     with warnings.catch_warnings():
    #         warnings.filterwarnings(
    #             action="ignore",
    #             message="This nwb file with identifier ",
    #             category=UserWarning,
    #         )
    #         eye_tracking_rig_geometry = EyeTrackingRigGeometry.from_nwb(
    #             nwbfile=nwbfile
    #         )
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings(
    #             action="ignore",
    #             message="This nwb file with identifier ",
    #             category=UserWarning,
    #         )
    #         eye_tracking_table = EyeTrackingTable.from_nwb(
    #             nwbfile=nwbfile,
    #             z_threshold=eye_tracking_z_threshold,
    #             dilation_frames=eye_tracking_dilation_frames,
    #         )

    #     return cls(
    #         behavior_session_id=behavior_session_id,
    #         stimulus_timestamps=stimulus_timestamps,
    #         running_acquisition=running_acquisition,
    #         raw_running_speed=raw_running_speed,
    #         running_speed=running_speed,
    #         metadata=metadata,
    #         licks=licks,
    #         rewards=rewards,
    #         stimuli=stimuli,
    #         task_parameters=task_parameters,
    #         trials=trials,
    #         date_of_acquisition=date_of_acquisition,
    #         eye_tracking_table=eye_tracking_table,
    #         eye_tracking_rig_geometry=eye_tracking_rig_geometry,
    #     )


    # @classmethod
    # def from_nwb_path(cls, nwb_path, **kwargs):
    #     """

    #     Parameters
    #     ----------
    #     nwb_path
    #         Path to nwb file
    #     kwargs
    #         Kwargs to be passed to `from_nwb`

    #     Returns
    #     -------
    #     An instantiation of a `BehaviorSession`
    #     """
    #     nwb_path = str(nwb_path)
    #     with pynwb.NWBHDF5IO(nwb_path, "r", load_namespaces=True) as read_io:
    #         nwbfile = read_io.read()
    #         return cls.from_nwb(nwbfile=nwbfile, **kwargs)



class VisualBehaviorNeuropixelsProjectCloudApi(ProjectCloudApiBase):
    MANIFEST_COMPATIBILITY = ["0.1.0", "10.0.0"]

    def _load_manifest_tables(self):
        self._get_ecephys_session_table()
        self._get_behavior_session_table()
        self._get_unit_table()
        self._get_probe_table()
        self._get_channel_table()


    def get_ecephys_session(self, ecephys_session_id):
        session_meta = return_one_dataframe_row_only(
            input_table=self._ecephys_session_table,
            index_value=ecephys_session_id,
            table_name="ecephys_session_table",
        )
        probes_meta = self._probe_table[
            (self._probe_table["ecephys_session_id"] == ecephys_session_id)
            & (self._probe_table["has_lfp_data"])
        ]
        session_file_id = str(int(session_meta[self.cache.file_id_column]))
        session_data_path = self._get_data_path(file_id=session_file_id)

        def make_lazy_load_filepath_function(file_id):
            """Due to late binding closure. See:
            https://docs.python-guide.org/writing/gotchas/
            #late-binding-closures"""

            def f():
                return self._get_data_path(file_id=file_id)

            return f

        # Backwards compatibility check for VBN data that doesn't contain
        # the LFP dataset.
        has_probe_file = self.cache.file_id_column in probes_meta.columns

        # if not probes_meta.empty and has_probe_file:
        #     probe_meta = {
        #         p.name: ProbeWithLFPMeta(
        #             lfp_csd_filepath=make_lazy_load_filepath_function(
        #                 file_id=str(int(getattr(p, self.cache.file_id_column)))
        #             ),
        #             lfp_sampling_rate=p.lfp_sampling_rate,
        #         )
        #         for p in probes_meta.itertuples(index=False)
        #     }
        # else:
        # probe_meta = None
        # return BehaviorEcephysSession.from_nwb_path(
        #     str(session_data_path), probe_meta=probe_meta
        # )


    def _get_ecephys_session_table(self):
        session_table_path = self._get_metadata_path(fname="ecephys_sessions")
        df = pd.read_csv(session_table_path)
        df = enforce_df_int_typing(df, INTEGER_COLUMNS, use_pandas_type=True)
        self._ecephys_session_table = df.set_index("ecephys_session_id")

    def _get_behavior_session_table(self):
        session_table_path = self._get_metadata_path(fname="behavior_sessions")
        df = pd.read_csv(session_table_path)
        df = enforce_df_int_typing(df, INTEGER_COLUMNS, use_pandas_type=True)
        self._behavior_session_table = df.set_index("behavior_session_id")

    def _get_probe_table(self):
        probe_table_path = self._get_metadata_path(fname="probes")
        df = pd.read_csv(probe_table_path)
        self._probe_table = df.set_index("ecephys_probe_id")

    def _get_unit_table(self):
        unit_table_path = self._get_metadata_path(fname="units")
        df = pd.read_csv(unit_table_path)
        self._unit_table = df.set_index("unit_id")

    def _get_channel_table(self):
        channel_table_path = self._get_metadata_path(fname="channels")
        df = pd.read_csv(channel_table_path)
        self._channel_table = df.set_index("ecephys_channel_id")


class VisualBehaviorNeuropixelsProjectCache(ProjectCacheBase):
    PROJECT_NAME = "visual-behavior-neuropixels"
    BUCKET_NAME = "visual-behavior-neuropixels-data"

    def __init__(self, fetch_api, fetch_tries=2):
        super().__init__(fetch_api=fetch_api, fetch_tries=fetch_tries)

    @classmethod
    def cloud_api_class(cls):
        return VisualBehaviorNeuropixelsProjectCloudApi


    def get_ecephys_session(self, ecephys_session_id):
        """
        Loads all data for `ecephys_session_id` into an
        `allensdk.ecephys.behavior_ecephys_session.BehaviorEcephysSession`
        instance

        Parameters
        ----------
        ecephys_session_id: int
            The ecephys session id

        Returns
        -------
        `allensdk.ecephys.behavior_ecephys_session.BehaviorEcephysSession`
        instance

        """
        return self.fetch_api.get_ecephys_session(ecephys_session_id)



class BehaviorCloudCacheVersionException(Exception):
    pass

def version_check(manifest_version: str,
                  data_pipeline_version: str,
                  cmin: str,
                  cmax: str):
    import semver

    mver_parsed = semver.VersionInfo.parse(manifest_version)
    cmin_parsed = semver.VersionInfo.parse(cmin)
    cmax_parsed = semver.VersionInfo.parse(cmax)

    if (mver_parsed < cmin_parsed) | (mver_parsed >= cmax_parsed):
        estr = (f"the manifest has manifest_version {manifest_version} but "
                "this version of AllenSDK is compatible only with manifest "
                f"versions {cmin} <= X < {cmax}. \n"
                "Consider using a version of AllenSDK closer to the version "
                f"used to release the data: {data_pipeline_version}")
        raise BehaviorCloudCacheVersionException(estr)


def enforce_df_int_typing(input_df, int_columns, use_pandas_type=False):
    """Enforce integer typing for columns that may have lost int typing when
    combined into the final DataFrame.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame with typing to enforce.
    int_columns : list of str
        Columns to enforce int typing and fill any NaN/None values with the
        value set in INT_NULL in this file. Requested columns not in the
        dataframe are ignored.
    use_pandas_type : bool
        Instead of filling with the value INT_NULL to enforce integer typing,
        use the pandas type Int64. This type can have issues converting to
        numpy/array type values.

    Returns
    -------
    output_df : pandas.DataFrame
        DataFrame specific columns hard typed to Int64 to allow NA values
        without resorting to float type.
    """
    for col in int_columns:
        if col in input_df.columns:
            if use_pandas_type:
                input_df[col] = input_df[col].astype("Int64")
            else:
                input_df[col] = input_df[col].fillna(INT_NULL).astype(int)
    return input_df


def return_one_dataframe_row_only(
    input_table: pd.DataFrame, index_value: int, table_name: str
) -> pd.Series:
    """Lookup and return one and only one row from the DataFrame returning
    an informative error if no or multiple rows are returned for a given
    index.

    This method is used mainly to return a more informative error when
    attempting to retrieve metadata from the values behavior cache metadata
    tables.

    Parameters
    ----------
    input_table : pandas.DataFrame
        Input dataframe to retrieve row from.
    index_value : int
        Index of the row to return. Must match an index in the input
         dataframe/table. i.e. in the case of ecephys_session_table or
        behavior_session_table.
    table_name : str
        Name of the table being returned. Used to output the table name
        in case of error.

    Returns
    -------
    row : pandas.Series
        Row corresponding to the input index.
    """
    try:
        row = input_table.loc[index_value]
    except KeyError:
        raise RuntimeError(
            f"The {table_name} should have "
            "1 and only 1 entry for a given "
            f"{input_table.index.name}. No indexed rows found for "
            f"id={index_value}"
        )
    if not isinstance(row, pd.Series):
        raise RuntimeError(
            f"The {table_name} should have "
            "1 and only 1 entry for a given "
            f"{input_table.index.name}. For "
            f"{index_value} "
            f" there are {len(row)} entries."
        )
    return row
