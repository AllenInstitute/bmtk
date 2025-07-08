import pandas as pd
import functools
from pathlib import Path

from ..cache import Cache
from ..utils import write_from_stream
from ..rma_engine import RmaEngine


class EcephysSession:
    def __init__(self, nwb_path):
        self.nwb_path = nwb_path


class EcephysProjectWarehouseApi:
    def __init__(self, rma_engine=None):
        if rma_engine is None:
            rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")
        self.rma_engine = rma_engine

    @classmethod
    def default(cls, asynchronous=False, **rma_kwargs):
        _rma_kwargs = {"scheme": "http", "host": "api.brain-map.org"}
        _rma_kwargs.update(rma_kwargs)

        engine_cls = AsyncRmaEngine if asynchronous else RmaEngine
        return cls(engine_cls(**_rma_kwargs))

    def get_session_data(self, session_id, **kwargs):
        query = "criteria=model::WellKnownFile" \
                ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']" \
                "[attachable_type$eq'EcephysSession']" \
                f"[attachable_id$eq{session_id}]"
        well_known_files = self.rma_engine.get_rma_tabular(query)

        if well_known_files.shape[0] != 1:
            raise ValueError(
                f"expected exactly 1 nwb file for session {session_id}, found: {well_known_files}"  # noqa: E501
            )

        download_link = well_known_files.iloc[0]["download_link"]
        return self.rma_engine.stream(download_link)


    def get_sessions(
        self, session_ids=None, has_eye_tracking=None, stimulus_names=None
    ):
        response = build_and_execute(
            (
                "{% import 'rma_macros' as rm %}"
                "{% import 'macros' as m %}"
                "criteria=model::EcephysSession"
                r"{{rm.optional_contains('id',session_ids)}}"
                r"{%if has_eye_tracking is not none%}[fail_eye_tracking$eq{{m.str(not has_eye_tracking).lower()}}]{%endif%}"  # noqa: E501
                r"{{rm.optional_contains('stimulus_name',stimulus_names,True)}}"  # noqa: E501
                ",rma::include,specimen(donor(age))"
                ",well_known_files(well_known_file_type)"
            ),
            base=rma_macros(),
            engine=self.rma_engine.get_rma_tabular,
            session_ids=session_ids,
            has_eye_tracking=has_eye_tracking,
            stimulus_names=stimulus_names,
        )

        response.set_index("id", inplace=True)

        age_in_days = []
        sex = []
        genotype = []
        has_nwb = []

        for idx, row in response.iterrows():
            age_in_days.append(row["specimen"]["donor"]["age"]["days"])
            sex.append(row["specimen"]["donor"]["sex"])

            gt = row["specimen"]["donor"]["full_genotype"]
            if gt is None:
                gt = "wt"
            genotype.append(gt)

            current_has_nwb = False
            for wkf in row["well_known_files"]:
                if wkf["well_known_file_type"]["name"] == "EcephysNwb":
                    current_has_nwb = True
            has_nwb.append(current_has_nwb)

        response["age_in_days"] = age_in_days
        response["sex"] = sex
        response["genotype"] = genotype
        response["has_nwb"] = has_nwb

        response.drop(
            columns=["specimen", "fail_eye_tracking", "well_known_files"],
            inplace=True,
        )
        response.rename(
            columns={"stimulus_name": "session_type"}, inplace=True
        )

        return response

    def get_probes(self, probe_ids=None, session_ids=None):
        raise NotImplementedError()


    def get_channels(self, channel_ids=None, probe_ids=None):
        raise NotImplementedError()


    def get_rig_metadata(self):
        raise NotImplementedError()


    def get_units(self, unit_ids=None, channel_ids=None, probe_ids=None, session_ids=None, *a, **k):
        raise NotImplementedError()


    def get_unit_analysis_metrics(self, unit_ids=None, ecephys_session_ids=None, session_types=None):
        raise NotImplementedError()


    def get_probe_lfp_data(self, probe_id):
        raise NotImplementedError()





class EcephysProjectCache(Cache):
    SESSIONS_KEY = 'sessions'
    PROBES_KEY = 'probes'
    CHANNELS_KEY = 'channels'
    UNITS_KEY = 'units'

    SESSION_DIR_KEY = 'session_data'
    SESSION_NWB_KEY = 'session_nwb'
    PROBE_LFP_NWB_KEY = "probe_lfp_nwb"

    NATURAL_MOVIE_DIR_KEY = "movie_dir"
    NATURAL_MOVIE_KEY = "natural_movie"

    NATURAL_SCENE_DIR_KEY = "natural_scene_dir"
    NATURAL_SCENE_KEY = "natural_scene"

    SESSION_ANALYSIS_METRICS_KEY = "session_analysis_metrics"
    TYPEWISE_ANALYSIS_METRICS_KEY = "typewise_analysis_metrics"

    MANIFEST_VERSION = '0.3.0'

    SUPPRESS_FROM_PROBES = (
        "air_channel_index", "surface_channel_index",
        "date_of_acquisition", "published_at", "specimen_id", "session_type", "isi_experiment_id", "age_in_days",
        "sex", "genotype", "has_nwb", "lfp_temporal_subsampling_factor"
    )

    def __init__(
            self,
            fetch_api=None,
            fetch_tries=2,
            stream_writer=None,
            manifest=None,
            version=None,
            cache=True):

        manifest_ = manifest or "ecephys_project_manifest.json"
        version_ = version or self.MANIFEST_VERSION

        super(EcephysProjectCache, self).__init__(manifest=manifest_,
                                                  version=version_,
                                                  cache=cache)
        self.fetch_api = (EcephysProjectWarehouseApi.default()
                          if fetch_api is None else fetch_api)
        self.fetch_tries = fetch_tries
        self.stream_writer = (stream_writer
                              or self.fetch_api.rma_engine.write_bytes)
        if stream_writer is not None:
            self.stream_writer = stream_writer
        else:
            if hasattr(self.fetch_api, "rma_engine"):    # EcephysProjectWarehouseApi    # noqa
                self.stream_writer = self.fetch_api.rma_engine.write_bytes
            # TODO: Make these names consistent in the different fetch apis
            elif hasattr(self.fetch_api, "app_engine"):    # EcephysProjectLimsApi    # noqa
                self.stream_writer = self.fetch_api.app_engine.write_bytes
            else:
                raise ValueError(
                    "Must either set value for `stream_writer`, or use a "
                    "`fetch_api` with an rma_engine or app_engine attribute "
                    "that implements `write_bytes`. See `HttpEngine` and "
                    "`AsyncHttpEngine` from "
                    "allensdk.brain_observatory.ecephys.ecephys_project_api."
                    "http_engine for examples.")


    def add_manifest_paths(self, manifest_builder):
        manifest_builder = super(EcephysProjectCache, self).add_manifest_paths(manifest_builder)

        manifest_builder.add_path(
            self.SESSIONS_KEY, 'sessions.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.PROBES_KEY, 'probes.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.CHANNELS_KEY, 'channels.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.UNITS_KEY, 'units.csv', parent_key='BASEDIR', typename='file'
        )

        manifest_builder.add_path(
            self.SESSION_DIR_KEY, 'session_%d', parent_key='BASEDIR', typename='dir'
        )

        manifest_builder.add_path(
            self.SESSION_NWB_KEY, 'session_%d.nwb', parent_key=self.SESSION_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.SESSION_ANALYSIS_METRICS_KEY, 'session_%d_analysis_metrics.csv', parent_key=self.SESSION_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.PROBE_LFP_NWB_KEY, 'probe_%d_lfp.nwb', parent_key=self.SESSION_DIR_KEY, typename='file'
        )

        manifest_builder.add_path(
            self.NATURAL_MOVIE_DIR_KEY, "natural_movie_templates", parent_key="BASEDIR", typename="dir"
        )

        manifest_builder.add_path(
            self.TYPEWISE_ANALYSIS_METRICS_KEY, "%s_analysis_metrics.csv", parent_key='BASEDIR', typename="file"
        )

        manifest_builder.add_path(
            self.NATURAL_MOVIE_KEY, "natural_movie_%d.h5", parent_key=self.NATURAL_MOVIE_DIR_KEY, typename="file"
        )

        manifest_builder.add_path(
            self.NATURAL_SCENE_DIR_KEY, "natural_scene_templates", parent_key="BASEDIR", typename="dir"
        )

        manifest_builder.add_path(
            self.NATURAL_SCENE_KEY, "natural_scene_%d.tiff", parent_key=self.NATURAL_SCENE_DIR_KEY, typename="file"
        )

        return manifest_builder

    @classmethod
    def from_warehouse(cls,
                       scheme=None,
                       host=None,
                       asynchronous=False,
                       manifest=None,
                       version=None,
                       cache=True,
                       fetch_tries=2,
                       timeout=1200):
        if scheme and host:
            app_kwargs = {"scheme": scheme, "host": host,
                          "asynchronous": asynchronous}
        else:
            app_kwargs = {"asynchronous": asynchronous}
        app_kwargs['timeout'] = timeout
        return cls._from_http_source_default(
            EcephysProjectWarehouseApi, app_kwargs, manifest=manifest,
            version=version, cache=cache, fetch_tries=fetch_tries
        )

    @classmethod
    def _from_http_source_default(cls, fetch_api_cls, fetch_api_kwargs, **kwargs):
        fetch_api_kwargs = {
            "asynchronous": True
        } if fetch_api_kwargs is None else fetch_api_kwargs

        if kwargs.get("stream_writer") is None:
            if fetch_api_kwargs.get("asynchronous", True):
                kwargs["stream_writer"] = write_bytes_from_coroutine
            else:
                kwargs["stream_writer"] = write_from_stream

        return cls(
            fetch_api=fetch_api_cls.default(**fetch_api_kwargs),
            **kwargs
        )

    def get_session_data(self, session_id, force_overwrite=False):
        """ Obtain an EcephysSession object containing detailed data for a single session
        """

        path = self.get_cache_path(None, self.SESSION_NWB_KEY, session_id, session_id)
        fetch = functools.partial(self.fetch_api.get_session_data, session_id)
        write = self.stream_writer
        self._fetch_cached_session(path, fetch, write, force_overwrite)
        return EcephysSession(path)
        
    
    def _fetch_cached_session(self, path, fetch, write, force_overwrite=False):
        path = path if isinstance(path, Path) else Path(path)
        if not force_overwrite and path.exists():
            return
        
        path.parent.mkdir(parents=True, exist_ok=True)

        data = fetch()
        write(path, data)


    def get_channels(self, suppress=None):
        raise NotImplementedError()


    def get_probes(self, suppress=None):
        raise NotImplementedError()


    def get_unit_analysis_metrics_for_session(self, session_id, annotate: bool = True, filter_by_validity: bool = True, **unit_filter_kwargs):
        raise NotImplementedError()


    def get_unit_analysis_metrics_for_session(self, session_id, annotate: bool = True, filter_by_validity: bool = True, **unit_filter_kwargs):
        raise NotImplementedError()
