import os
from abc import ABC, abstractmethod
import warnings
import pathlib
import json
import pandas as pd
import re
import semver
import urllib.parse as url_parse
import platform

import boto3
from botocore import UNSIGNED
from botocore.client import Config

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    from ..utils import FakeTqdm as tqdm


from ..utils import file_hash_from_path



class CacheFileAttributes(object):
    """
    This class will contain the attributes of a remotely stored file
    so that they can easily and consistently be passed around between
    the methods making up the remote file cache and manifest classes

    Parameters
    ----------
    url: str
        The full URL of the remote file
    version_id: str
        A string specifying the version of the file (probably calculated
        by S3)
    file_hash: str
        The (hexadecimal) file hash of the file
    local_path: pathlib.Path
        The path to the location where the file's local copy should be stored
        (probably computed by the Manifest class)
    """

    def __init__(self,
                 url: str,
                 version_id: str,
                 file_hash: str,
                 local_path: pathlib.Path):

        if not isinstance(url, str):
            raise ValueError(f"url must be str; got {type(url)}")
        if not isinstance(version_id, str):
            raise ValueError(f"version_id must be str; got {type(version_id)}")
        if not isinstance(file_hash, str):
            raise ValueError(f"file_hash must be str; "
                             f"got {type(file_hash)}")
        if not isinstance(local_path, pathlib.Path):
            raise ValueError(f"local_path must be pathlib.Path; "
                             f"got {type(local_path)}")

        self._url = url
        self._version_id = version_id
        self._file_hash = file_hash
        self._local_path = local_path

    @property
    def url(self) -> str:
        return self._url

    @property
    def version_id(self) -> str:
        return self._version_id

    @property
    def file_hash(self) -> str:
        return self._file_hash

    @property
    def local_path(self) -> pathlib.Path:
        return self._local_path

    def __str__(self):
        output = {'url': self.url,
                  'version_id': self.version_id,
                  'file_hash': self.file_hash,
                  'local_path': str(self.local_path)}
        output = json.dumps(output, indent=2, sort_keys=True)
        return f'CacheFileParameters{output}'




class Manifest(object):
    """
    A class for loading and manipulating the online manifest.json associated
    with a dataset release

    Each Manifest instance should represent the data for 1 and only 1
    manifest.json file.

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        The path to the directory where local copies of files will be stored
    json_input:
        A ''.read()''-supporting file-like object containing
        a JSON document to be deserialized (i.e. same as the
        first argument to json.load)
    use_static_project_dir: bool
        When determining what the local path of a remote resource
        (data or metadata file) should be, the Manifest class will typically
        create a versioned project subdirectory under the user provided
        `cache_dir` (e.g. f"{cache_dir}/{project_name}-{manifest_version}")
        to allow the possibility of multiple manifest (and data) versions to be
        used. In certain cases, like when using a project's s3 bucket
        directly as the cache_dir, the project directory name needs to be
        static (e.g. f"{cache_dir}/{project_name}"). When set to True,
        the Manifest class will use a static project directory to determine
        local paths for remote resources. Defaults to False.
    """

    def __init__(self, cache_dir, json_input, use_static_project_dir=False):
        if isinstance(cache_dir, str):
            self._cache_dir = pathlib.Path(cache_dir).resolve()
        elif isinstance(cache_dir, pathlib.Path):
            self._cache_dir = cache_dir.resolve()
        else:
            raise ValueError("cache_dir must be either a str "
                             "or a pathlib.Path; "
                             f"got {type(cache_dir)}")

        self._use_static_project_dir = use_static_project_dir

        self._data = json.load(json_input)
        if not isinstance(self._data, dict):
            raise ValueError("Expected to deserialize manifest into a dict; "
                             f"instead got {type(self._data)}")
        self._project_name: str = self._data["project_name"]
        self._version: str = self._data['manifest_version']
        self._file_id_column: str = self._data['metadata_file_id_column_name']
        self._data_pipeline: str = self._data["data_pipeline"]

        self._metadata_file_names = [
            file_name for file_name in self._data['metadata_files']
        ]
        self._metadata_file_names.sort()

        self._file_id_values = [ii for ii in self._data['data_files'].keys()]
        self._file_id_values.sort()

    @property
    def project_name(self):
        """
        The name of the project whose data and metadata files this
        manifest tracks.
        """
        return self._project_name

    @property
    def version(self):
        """
        The version of the dataset currently loaded
        """
        return self._version

    @property
    def file_id_column(self):
        """
        The column in the metadata files used to uniquely
        identify data files
        """
        return self._file_id_column

    @property
    def metadata_file_names(self):
        """
        List of metadata file names associated with this dataset
        """
        return self._metadata_file_names

    @property
    def file_id_values(self):
        """
        List of valid file_id values
        """
        return self._file_id_values

    def _create_file_attributes(self, remote_path, version_id, file_hash):
        """
        Create the cache_file_attributes describing a file.
        This method does the work of assigning a local_path for a remote file.

        Parameters
        ----------
        remote_path: str
            The full URL to a file
        version_id: str
            The string specifying the version of the file
        file_hash: str
            The (hexadecimal) file hash of the file

        Returns
        -------
        CacheFileAttributes
        """

        if self._use_static_project_dir:
            # If we only want to support 1 version of the project on disk
            # like when mounting the project S3 bucket as a file system
            project_dir_name = f"{self._project_name}"
        else:
            # If we want to support multiple versions of the project on disk
            # paths should be built like:
            # {cache_dir} / {project_name}-{manifest_version} / relative_path
            # Example:
            # my_cache_dir/visual-behavior-ophys-1.0.0/behavior_sessions/etc...
            project_dir_name = f"{self._project_name}-{self._version}"

        project_dir = self._cache_dir / project_dir_name

        # The convention of the data release tool is to have all
        # relative_paths from remote start with the project name which
        # we want to remove since we already specified a project_dir_name
        relative_path = relative_path_from_url(remote_path)
        shaved_rel_path = "/".join(relative_path.split("/")[1:])

        local_path = project_dir / shaved_rel_path

        obj = CacheFileAttributes(
            remote_path,
            version_id,
            file_hash,
            local_path
        )

        return obj

    def metadata_file_attributes(self, metadata_file_name):
        """
        Return the CacheFileAttributes associated with a metadata file

        Parameters
        ----------
        metadata_file_name: str
            Name of the metadata file. Must be in self.metadata_file_names

        Return
        ------
        CacheFileAttributes

        Raises
        ------
        RuntimeError
            If you try to run this method when self._data is None (meaning
            you haven't yet loaded a manifest.json)

        ValueError
            If the metadata_file_name is not a valid option
        """
        if self._data is None:
            raise RuntimeError("You cannot retrieve "
                               "metadata_file_attributes;\n"
                               "you have not yet loaded a manifest.json file")

        if metadata_file_name not in self._metadata_file_names:
            raise ValueError(f"{metadata_file_name}\n"
                             "is not in self.metadata_file_names:\n"
                             f"{self._metadata_file_names}")

        file_data = self._data['metadata_files'][metadata_file_name]
        return self._create_file_attributes(file_data['url'],
                                            file_data['version_id'],
                                            file_data['file_hash'])

    def data_file_attributes(self, file_id):
        """
        Return the CacheFileAttributes associated with a data file

        Parameters
        ----------
        file_id:
            The identifier of the data file whose attributes are to be
            returned. Must be a key in self._data['data_files']

        Return
        ------
        CacheFileAttributes

        Raises
        ------
        RuntimeError
            If you try to run this method when self._data is None (meaning
            you haven't yet loaded a manifest.json file)

        ValueError
            If the file_id is not a valid option
        """
        if self._data is None:
            raise RuntimeError("You cannot retrieve data_file_attributes;\n"
                               "you have not yet loaded a manifest.json file")

        if file_id not in self._data['data_files']:
            valid_keys = list(self._data['data_files'].keys())
            valid_keys.sort()
            raise ValueError(f"file_id: {file_id}\n"
                             "Is not a data file listed in manifest:\n"
                             f"{valid_keys}")

        file_data = self._data['data_files'][file_id]
        return self._create_file_attributes(file_data['url'],
                                            file_data['version_id'],
                                            file_data['file_hash'])


class OutdatedManifestWarning(UserWarning):
    pass


class MissingLocalManifestWarning(UserWarning):
    pass


class BasicLocalCache(ABC):
    """
    A class to handle the loading and accessing a project's data and
    metadata from a local cache directory. Does NOT include any 'smart'
    features like:
    1. Keeping track of last loaded manifest
    2. Constructing symlinks for valid data from previous dataset versions
    3. Warning of outdated manifests

    For those features (and more) see the CloudCacheBase class

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        Path to the directory where data and metadata are stored on the
        local system

    project_name: str
        the name of the project this cache is supposed to access. This will
        be the root directory for all files stored in the bucket.

    ui_class_name: Optional[str]
        Name of the class users are actually using to manipulate this
        functionality (used to populate helpful error messages)
    """

    def __init__(self, cache_dir, project_name, ui_class_name=None):
        os.makedirs(cache_dir, exist_ok=True)

        # the class users are actually interacting with
        # (for warning message purposes)
        if ui_class_name is None:
            self._user_interface_class = type(self).__name__
        else:
            self._user_interface_class = ui_class_name

        self._manifest = None
        self._manifest_name = None

        self._cache_dir = cache_dir
        self._project_name = project_name

        self._manifest_file_names = self._list_all_manifests()

    # ====================== BasicLocalCache properties =======================

    @property
    def ui(self):
        return self._user_interface_class

    @property
    def current_manifest(self):
        """The name of the currently loaded manifest"""
        return self._manifest_name

    @property
    def project_name(self) -> str:
        """The name of the project that this cache is accessing"""
        return self._project_name

    @property
    def manifest_prefix(self) -> str:
        """On-line prefix for manifest files"""
        return f'{self.project_name}/manifests/'

    @property
    def file_id_column(self) -> str:
        """The col name in metadata files used to uniquely identify data files
        """
        return self._manifest.file_id_column

    @property
    def version(self) -> str:
        """The version of the dataset currently loaded"""
        return self._manifest.version

    @property
    def metadata_file_names(self) -> list:
        """List of metadata file names associated with this dataset"""
        return self._manifest.metadata_file_names

    @property
    def manifest_file_names(self) -> list:
        """Sorted list of manifest file names associated with this dataset
        """
        return self._manifest_file_names

    @property
    def latest_manifest_file(self) -> str:
        """parses on-line available manifest files for semver string
        and returns the latest one
        self.manifest_file_names are assumed to be of the form
        '<anything>_v<semver_str>.json'

        Returns
        -------
        str
            the filename whose semver string is the latest one
        """
        return self._find_latest_file(self.manifest_file_names)

    @property
    def cache_dir(self) -> str:
        """Return the cache directory path.

        Returns
        -------
        str
            Full cache directory path
        """
        return self._cache_dir

    # ====================== BasicLocalCache methods ==========================

    @abstractmethod
    def _list_all_manifests(self):
        """
        Return a list of all of the file names of the manifests associated
        with this dataset
        """
        raise NotImplementedError()

    def list_all_downloaded_manifests(self):
        """
        Return a list of all of the manifest files that have been
        downloaded for this dataset
        """
        output = [x for x in os.listdir(self._cache_dir)
                  if re.fullmatch(".*_manifest_v.*.json", x)]
        output.sort()
        return output

    def _find_latest_file(self, file_name_list):
        vstrs = [s.split(".json")[0].split("_v")[-1]
                 for s in file_name_list]
        versions = [semver.VersionInfo.parse(v) for v in vstrs]
        imax = versions.index(max(versions))
        return file_name_list[imax]

    def _load_manifest(
        self,
        manifest_name: str,
        use_static_project_dir: bool = False
    ) -> Manifest:
        """
        Load and return a manifest from this dataset.

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        use_static_project_dir: bool
            When determining what the local path of a remote resource
            (data or metadata file) should be, the Manifest class will
            typically create a versioned project subdirectory under the user
            provided `cache_dir`
            (e.g. f"{cache_dir}/{project_name}-{manifest_version}")
            to allow the possibility of multiple manifest (and data) versions
            to be used. In certain cases, like when using a project's s3 bucket
            directly as the cache_dir, the project directory name needs to be
            static (e.g. f"{cache_dir}/{project_name}"). When set to True,
            the Manifest class will use a static project directory to determine
            local paths for remote resources. Defaults to False.

        Returns
        -------
        Manifest
        """
        if manifest_name not in self.manifest_file_names:
            raise ValueError(
                f"Manifest to load ({manifest_name}) is not one of the "
                "valid manifest names for this dataset. Valid names include:\n"
                f"{self.manifest_file_names}"
            )

        if use_static_project_dir:
            manifest_path = os.path.join(
                self._cache_dir, self.project_name, "manifests", manifest_name
            )
        else:
            manifest_path = os.path.join(self._cache_dir, manifest_name)

        with open(manifest_path, "r") as f:
            local_manifest = Manifest(
                cache_dir=self._cache_dir,
                json_input=f,
                use_static_project_dir=use_static_project_dir
            )

        return local_manifest

    def load_manifest(self, manifest_name: str):
        """
        Load a manifest from this dataset.

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """
        self._manifest = self._load_manifest(manifest_name)
        self._manifest_name = manifest_name

    def _file_exists(self, file_attributes: CacheFileAttributes) -> bool:
        """
        Given a CacheFileAttributes describing a file, assess whether or
        not that file exists locally.

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            Description of the file to look for

        Returns
        -------
        bool
            True if the file exists and is valid; False otherwise

        Raises
        -----
        RuntimeError
            If file_attributes.local_path exists but is not a file.
            It would be unclear how the cache should proceed in this case.
        """
        file_exists = False

        if file_attributes.local_path.exists():
            if not file_attributes.local_path.is_file():
                raise RuntimeError(f"{file_attributes.local_path}\n"
                                   "exists, but is not a file;\n"
                                   "unsure how to proceed")

            file_exists = True

        return file_exists

    def metadata_path(self, fname: str) -> dict:
        """
        Return the local path to a metadata file, and test for the
        file's existence

        Parameters
        ----------
        fname: str
            The name of the metadata file to be accessed

        Returns
        -------
        dict

            'path' will be a pathlib.Path pointing to the file's location

            'exists' will be a boolean indicating if the file
            exists in a valid state

            'file_attributes' is a CacheFileAttributes describing the file
            in more detail

        Raises
        ------
        RuntimeError
            If the file cannot be downloaded
        """
        file_attributes = self._manifest.metadata_file_attributes(fname)
        exists = self._file_exists(file_attributes)
        local_path = file_attributes.local_path
        output = {'local_path': local_path,
                  'exists': exists,
                  'file_attributes': file_attributes}

        return output

    def data_path(self, file_id) -> dict:
        """
        Return the local path to a data file, and test for the
        file's existence

        Parameters
        ----------
        file_id:
            The unique identifier of the file to be accessed

        Returns
        -------
        dict

            'local_path' will be a pathlib.Path pointing to the file's location

            'exists' will be a boolean indicating if the file
            exists in a valid state

            'file_attributes' is a CacheFileAttributes describing the file
            in more detail

        Raises
        ------
        RuntimeError
            If the file cannot be downloaded
        """
        file_attributes = self.get_file_attributes(file_id)
        exists = self._file_exists(file_attributes)
        local_path = file_attributes.local_path
        output = {'local_path': local_path,
                  'exists': exists,
                  'file_attributes': file_attributes}

        return output

    def get_file_attributes(self, file_id):
        """
        Retrieve file attributes for a given file_id from the meatadata.

        Parameters
        ----------
        file_id: str or int
            The unique identifier of the file to be accessed

        Returns
        -------
        CacheFileAttributes
        """
        return self._manifest.data_file_attributes(file_id)



class CloudCacheBase(BasicLocalCache):
    """
    A class to handle the downloading and accessing of data served from a cloud
    storage system

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        Path to the directory where data will be stored on the local system

    project_name: str
        the name of the project this cache is supposed to access. This will
        be the root directory for all files stored in the bucket.

    ui_class_name: Optional[str]
        Name of the class users are actually using to manipulate this
        functionality (used to populate helpful error messages)
    """

    _bucket_name = None

    def __init__(self, cache_dir, project_name, ui_class_name=None):
        super().__init__(cache_dir=cache_dir, project_name=project_name,
                         ui_class_name=ui_class_name)

        # what latest_manifest was the last time an OutdatedManifestWarning
        # was emitted
        self._manifest_last_warned_on = None

        c_path = pathlib.Path(self._cache_dir)

        # self._manifest_last_used contains the name of the manifest
        # last loaded from this cache dir (if applicable)
        self._manifest_last_used = c_path / '_manifest_last_used.txt'

        # self._downloaded_data_path is where we will keep a JSONized
        # dict mapping paths to downloaded files to their file_hashes;
        # this will be used when determining if a downloaded file
        # can instead be a symlink
        self._downloaded_data_path = c_path / '_downloaded_data.json'

        # if the local manifest is missing but there are
        # data files in cache_dir, emit a warning
        # suggesting that the user run
        # self.construct_local_manifest
        if not self._downloaded_data_path.exists():
            file_list = c_path.glob('**/*')
            has_files = False
            for fname in file_list:
                if fname.is_file():
                    if 'json' not in fname.name:
                        has_files = True
                        break
            if has_files:
                msg = 'This cache directory appears to '
                msg += 'contain data files, but it has no '
                msg += 'record of what those files are. '
                msg += 'You might want to consider running\n\n'
                msg += f'{self.ui}.construct_local_manifest()\n\n'
                msg += 'to avoid needlessly downloading duplicates '
                msg += 'of data files that did not change between '
                msg += 'data releases. NOTE: running this method '
                msg += 'will require hashing every data file you '
                msg += 'have currently downloaded and could be '
                msg += 'very time consuming.\n\n'
                msg += 'To avoid this warning in the future, make '
                msg += 'sure that\n\n'
                msg += f'{str(self._downloaded_data_path.resolve())}\n\n'
                msg += 'is not deleted between instantiations of this '
                msg += 'cache'
                warnings.warn(msg, MissingLocalManifestWarning)

    def construct_local_manifest(self) -> None:
        """
        Construct the dict that maps between file_hash and
        absolute local path. Save it to self._downloaded_data_path
        """
        lookup = {}
        files_to_hash = set()
        c_dir = pathlib.Path(self._cache_dir)
        file_iterator = c_dir.glob('**/*')
        for file_name in file_iterator:
            if file_name.is_file():
                if 'json' not in file_name.name:
                    if file_name != self._manifest_last_used:
                        files_to_hash.add(file_name.resolve())

        with tqdm.tqdm(files_to_hash,
                       total=len(files_to_hash),
                       unit='(files hashed)') as pbar:

            for local_path in pbar:
                hsh = file_hash_from_path(local_path)
                lookup[str(local_path.absolute())] = hsh

        with open(self._downloaded_data_path, 'w') as out_file:
            out_file.write(json.dumps(lookup, indent=2, sort_keys=True))

    def _warn_of_outdated_manifest(self, manifest_name: str) -> None:
        """
        Warn that manifest_name is not the latest manifest available
        """
        if self._manifest_last_warned_on is not None:
            if self.latest_manifest_file == self._manifest_last_warned_on:
                return None

        self._manifest_last_warned_on = self.latest_manifest_file

        msg = '\n\n'
        msg += 'The manifest file you are loading is not the '
        msg += 'most up to date manifest file available for '
        msg += 'this dataset. The most up to data manifest file '
        msg += 'available for this dataset is \n\n'
        msg += f'{self.latest_manifest_file}\n\n'
        msg += 'To see the differences between these manifests,'
        msg += 'run\n\n'
        msg += f"{self.ui}.compare_manifests('{manifest_name}', "
        msg += f"'{self.latest_manifest_file}')\n\n"
        msg += "To see all of the manifest files currently downloaded "
        msg += "onto your local system, run\n\n"
        msg += "self.list_all_downloaded_manifests()\n\n"
        msg += "If you just want to load the latest manifest, run\n\n"
        msg += "self.load_latest_manifest()\n\n"
        warnings.warn(msg, OutdatedManifestWarning)
        return None

    @property
    def latest_downloaded_manifest_file(self) -> str:
        """parses downloaded available manifest files for semver string
        and returns the latest one
        self.manifest_file_names are assumed to be of the form
        '<anything>_v<semver_str>.json'

        Returns
        -------
        str
            the filename whose semver string is the latest one
        """
        file_list = self.list_all_downloaded_manifests()
        if len(file_list) == 0:
            return ''
        return self._find_latest_file(self.list_all_downloaded_manifests())

    def load_last_manifest(self):
        """
        If this Cache was used previously, load the last manifest
        used in this cache. If this cache has never been used, load
        the latest manifest.
        """
        if not self._manifest_last_used.exists():
            self.load_latest_manifest()
            return None

        with open(self._manifest_last_used, 'r') as in_file:
            to_load = in_file.read()

        latest = self.latest_manifest_file

        if to_load not in self.manifest_file_names:
            msg = 'The manifest version recorded as last used '
            msg += f'for this cache -- {to_load}-- '
            msg += 'is not a valid manifest for this dataset. '
            msg += f'Loading latest version -- {latest} -- '
            msg += 'instead.'
            warnings.warn(msg, UserWarning)
            self.load_latest_manifest()
            return None

        if latest != to_load:
            self._manifest_last_warned_on = self.latest_manifest_file
            msg = f"You are loading {to_load}. A more up to date "
            msg += f"version of the dataset -- {latest} -- exists "
            msg += "online. To see the changes between the two "
            msg += "versions of the dataset, run\n"
            msg += f"{self.ui}.compare_manifests('{to_load}',"
            msg += f" '{latest}')\n"
            msg += "To load another version of the dataset, run\n"
            msg += f"{self.ui}.load_manifest('{latest}')"
            warnings.warn(msg, OutdatedManifestWarning)
        self.load_manifest(to_load)
        return None

    def load_latest_manifest(self):
        latest_downloaded = self.latest_downloaded_manifest_file
        latest = self.latest_manifest_file
        if latest != latest_downloaded:
            if latest_downloaded != '':
                msg = f'You are loading\n{self.latest_manifest_file}\n'
                msg += 'which is newer than the most recent manifest '
                msg += 'file you have previously been working with\n'
                msg += f'{latest_downloaded}\n'
                msg += 'It is possible that some data files have changed '
                msg += 'between these two data releases, which will '
                msg += 'force you to re-download those data files '
                msg += '(currently downloaded files will not be overwritten).'
                msg += f' To continue using {latest_downloaded}, run\n'
                msg += f"{self.ui}.load_manifest('{latest_downloaded}')"
                warnings.warn(msg, OutdatedManifestWarning)
        self.load_manifest(self.latest_manifest_file)

    @abstractmethod
    def _download_manifest(self,
                           manifest_name: str):
        """
        Download a manifest from the dataset

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """
        raise NotImplementedError()

    @abstractmethod
    def _download_file(self, file_attributes: CacheFileAttributes) -> bool:
        """
        Check if a file exists locally. If it does not, download it and
        return True. Return False otherwise.

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            Describes the file to download

        Returns
        -------
        bool
            True if the file was downloaded; False otherwise

        Raises
        ------
        RuntimeError
            If the path to the directory where the file is to be saved
            points to something that is not a directory.

        RuntimeError
            If it is not able to successfully download the file after
            10 iterations
        """
        raise NotImplementedError()

    def load_manifest(self, manifest_name: str):
        """
        Load a manifest from this dataset.

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """
        if manifest_name not in self.manifest_file_names:
            raise ValueError(
                f"Manifest to load ({manifest_name}) is not one of the "
                "valid manifest names for this dataset. Valid names include:\n"
                f"{self.manifest_file_names}"
            )

        if manifest_name != self.latest_manifest_file:
            self._warn_of_outdated_manifest(manifest_name)

        # If desired manifest does not exist, try to download it
        manifest_path = os.path.join(self._cache_dir, manifest_name)
        if not os.path.exists(manifest_path):
            self._download_manifest(manifest_name)

        self._manifest = self._load_manifest(manifest_name)

        # Keep track of the newly loaded manifest
        with open(self._manifest_last_used, 'w') as out_file:
            out_file.write(manifest_name)

        self._manifest_name = manifest_name

    def _update_list_of_downloads(self,
                                  file_attributes: CacheFileAttributes
                                  ) -> None:
        """
        Update the local file that keeps track of files that have actually
        been downloaded to reflect a newly downloaded file.

        Parameters
        ----------
        file_attributes: CacheFileAttributes

        Returns
        -------
        None
        """
        if not file_attributes.local_path.exists():
            # This file does not exist; there is nothing to do
            return None

        if self._downloaded_data_path.exists():
            with open(self._downloaded_data_path, 'rb') as in_file:
                downloaded_data = json.load(in_file)
        else:
            downloaded_data = {}

        abs_path = str(file_attributes.local_path.resolve())
        if abs_path in downloaded_data:
            if downloaded_data[abs_path] == file_attributes.file_hash:
                # this file has already been logged;
                # there is nothing to do
                return None

        downloaded_data[abs_path] = file_attributes.file_hash
        with open(self._downloaded_data_path, 'w') as out_file:
            out_file.write(json.dumps(downloaded_data,
                                      indent=2,
                                      sort_keys=True))
        return None

    def _check_for_identical_copy(self, file_attributes):
        """
        Check the manifest of files that have been locally downloaded to
        see if a file with an identical hash to the requested file has already
        been downloaded. If it has, create a symlink to the downloaded file
        at the requested file's localpath, update the manifest of downloaded
        files, and return True.

        Else return False

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            The file we are considering downloading

        Returns
        -------
        bool
        """
        if not self._downloaded_data_path.exists():
            return False

        with open(self._downloaded_data_path, 'rb') as in_file:
            available_files = json.load(in_file)

        matched_path = None
        for abs_path in available_files:
            if available_files[abs_path] == file_attributes.file_hash:
                matched_path = pathlib.Path(abs_path)

                # check that the file still exists,
                # in case someone accidentally deleted
                # the file at the root of a symlink
                if matched_path.is_file():
                    break
                else:
                    matched_path = None

        if matched_path is None:
            return False

        local_parent = file_attributes.local_path.parent.resolve()
        if not local_parent.exists():
            os.makedirs(local_parent)

        file_attributes.local_path.symlink_to(matched_path.resolve())
        return True

    def _file_exists(self, file_attributes) -> bool:
        """
        Given a CacheFileAttributes describing a file, assess whether or
        not that file exists locally and is valid (i.e. has the expected
        file hash)

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            Description of the file to look for

        Returns
        -------
        bool
            True if the file exists and is valid; False otherwise

        Raises
        -----
        RuntimeError
            If file_attributes.local_path exists but is not a file.
            It would be unclear how the cache should proceed in this case.
        """
        file_exists = False

        if file_attributes.local_path.exists():
            if not file_attributes.local_path.is_file():
                raise RuntimeError(f"{file_attributes.local_path}\n"
                                   "exists, but is not a file;\n"
                                   "unsure how to proceed")

            file_exists = True

        if not file_exists:
            file_exists = self._check_for_identical_copy(file_attributes)

        return file_exists

    def download_data(self, file_id) -> pathlib.Path:
        """
        Return the local path to a data file, downloading the file
        if necessary

        Parameters
        ----------
        file_id:
            The unique identifier of the file to be accessed

        Returns
        -------
        pathlib.Path
            The path indicating where the file is stored on the
            local system

        Raises
        ------
        RuntimeError
            If the file cannot be downloaded
        """
        super_attributes = self.data_path(file_id)
        file_attributes = super_attributes['file_attributes']
        was_downloaded = self._download_file(file_attributes)
        if was_downloaded:
            self._update_list_of_downloads(file_attributes)
        return file_attributes.local_path

    def download_metadata(self, fname: str) -> pathlib.Path:
        """
        Return the local path to a metadata file, downloading the
        file if necessary

        Parameters
        ----------
        fname: str
            The name of the metadata file to be accessed

        Returns
        -------
        pathlib.Path
            The path indicating where the file is stored on the
            local system

        Raises
        ------
        RuntimeError
            If the file cannot be downloaded
        """
        super_attributes = self.metadata_path(fname)
        file_attributes = super_attributes['file_attributes']
        was_downloaded = self._download_file(file_attributes)
        if was_downloaded:
            self._update_list_of_downloads(file_attributes)
        return file_attributes.local_path

    def get_metadata(self, fname: str) -> pd.DataFrame:
        """
        Return a pandas DataFrame of metadata

        Parameters
        ----------
        fname: str
            The name of the metadata file to load

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        This method will check to see if the specified metadata file exists
        locally. If it does not, the method will download the file. Use
        self.metadata_path() to find where the file is stored
        """
        local_path = self.download_metadata(fname)
        return pd.read_csv(local_path)

    def _detect_changes(self, filename_to_hash):
        """
        Assemble list of changes between two manifests

        Parameters
        ----------
        filename_to_hash: dict
            filename_to_hash[0] is a dict mapping file names to file hashes
            for manifest 0

            filename_to_hash[1] is a dict mapping file names to file hashes
            for manifest 1

        Returns
        -------
        List[Tuple[str, str]]
            List of changes between manifest 0 and manifest 1.

        Notes
        -----
        Changes are tuples of the form
        (fname, string describing how fname changed)

        e.g.

        ('data/f1.txt', 'data/f1.txt renamed data/f5.txt')
        ('data/f2.txt', 'data/f2.txt deleted')
        ('data/f3.txt', 'data/f3.txt created')
        ('data/f4.txt', 'data/f4.txt changed')
        """
        output = []
        n0 = set(filename_to_hash[0].keys())
        n1 = set(filename_to_hash[1].keys())
        all_file_names = n0.union(n1)

        hash_to_filename: dict = dict()
        for v in (0, 1):
            hash_to_filename[v] = {}
            for fname in filename_to_hash[v]:
                hash_to_filename[v][filename_to_hash[v][fname]] = fname

        for fname in all_file_names:
            delta = None
            if fname in filename_to_hash[0] and fname in filename_to_hash[1]:
                h0 = filename_to_hash[0][fname]
                h1 = filename_to_hash[1][fname]
                if h0 != h1:
                    delta = f'{fname} changed'
            elif fname in filename_to_hash[0]:
                h0 = filename_to_hash[0][fname]
                if h0 in hash_to_filename[1]:
                    f1 = hash_to_filename[1][h0]
                    delta = f'{fname} renamed {f1}'
                else:
                    delta = f'{fname} deleted'
            elif fname in filename_to_hash[1]:
                h1 = filename_to_hash[1][fname]
                if h1 not in hash_to_filename[0]:
                    delta = f'{fname} created'
            else:
                raise RuntimeError("should never reach this line")

            if delta is not None:
                output.append((fname, delta))

        return output

    def summarize_comparison(self, manifest_0_name, manifest_1_name):
        """
        Compare two manifests from this dataset. Return a dict
        containing the list of metadata and data files that changed
        between them

        Note: this assumes that manifest_0 predates manifest_1 (i.e.
        changes are listed relative to manifest_0)

        Parameters
        ----------
        manifest_0_name: str

        manifest_1_name: str

        Returns
        -------
        result: Dict[List[Tuple[str, str]]]
            result['data_changes'] lists changes to data files
            result['metadata_changes'] lists changes to metadata files

        Notes
        -----
        Changes are tuples of the form
        (fname, string describing how fname changed)

        e.g.

        ('data/f1.txt', 'data/f1.txt renamed data/f5.txt')
        ('data/f2.txt', 'data/f2.txt deleted')
        ('data/f3.txt', 'data/f3.txt created')
        ('data/f4.txt', 'data/f4.txt changed')
        """
        for manifest_name in [manifest_0_name, manifest_1_name]:
            manifest_path = os.path.join(self._cache_dir, manifest_name)
            if not os.path.exists(manifest_path):
                self._download_manifest(manifest_name)

        man0 = self._load_manifest(manifest_0_name)
        man1 = self._load_manifest(manifest_1_name)

        result: dict = dict()
        for (result_key,
             file_id_list,
             attr_lookup) in zip(('metadata_changes', 'data_changes'),
                                 ((man0.metadata_file_names,
                                   man1.metadata_file_names),
                                  (man0.file_id_values,
                                   man1.file_id_values)),
                                 ((man0.metadata_file_attributes,
                                   man1.metadata_file_attributes),
                                  (man0.data_file_attributes,
                                   man1.data_file_attributes))):

            filename_to_hash: dict = dict()
            for version in (0, 1):
                filename_to_hash[version] = {}
                for file_id in file_id_list[version]:
                    obj = attr_lookup[version](file_id)
                    file_name = relative_path_from_url(obj.url)
                    file_name = '/'.join(file_name.split('/')[1:])
                    filename_to_hash[version][file_name] = obj.file_hash
            changes = self._detect_changes(filename_to_hash)
            result[result_key] = changes
        return result

    def compare_manifests(self,
                          manifest_0_name: str,
                          manifest_1_name: str
                          ) -> str:
        """
        Compare two manifests from this dataset. Return a dict
        containing the list of metadata and data files that changed
        between them

        Note: this assumes that manifest_0 predates manifest_1

        Parameters
        ----------
        manifest_0_name: str

        manifest_1_name: str

        Returns
        -------
        str
            A string summarizing all of the changes going from
            manifest_0 to manifest_1
        """

        changes = self.summarize_comparison(manifest_0_name,
                                            manifest_1_name)
        if len(changes['data_changes']) == 0:
            if len(changes['metadata_changes']) == 0:
                return "The two manifests are equivalent"

        data_change_dict = {}
        for delta in changes['data_changes']:
            data_change_dict[delta[0]] = delta[1]
        metadata_change_dict = {}
        for delta in changes['metadata_changes']:
            metadata_change_dict[delta[0]] = delta[1]

        msg = 'Changes going from\n'
        msg += f'{manifest_0_name}\n'
        msg += 'to\n'
        msg += f'{manifest_1_name}\n\n'

        m_keys = list(metadata_change_dict.keys())
        m_keys.sort()
        for m in m_keys:
            msg += f'{metadata_change_dict[m]}\n'
        d_keys = list(data_change_dict.keys())
        d_keys.sort()
        for d in d_keys:
            msg += f'{data_change_dict[d]}\n'
        return msg





class S3CloudCache(CloudCacheBase):
    """
    A class to handle the downloading and accessing of data served from
    an S3-based storage system

    Parameters
    ----------
    cache_dir: str or pathlib.Path
        Path to the directory where data will be stored on the local system

    bucket_name: str
        for example, if bucket URI is 's3://mybucket' this value should be
        'mybucket'

    project_name: str
        the name of the project this cache is supposed to access. This will
        be the root directory for all files stored in the bucket.

    ui_class_name: Optional[str]
        Name of the class users are actually using to maniuplate this
        functionality (used to populate helpful error messages)
    """

    def __init__(self, cache_dir, bucket_name, project_name,
                 ui_class_name=None):
        self._manifest = None
        self._bucket_name = bucket_name

        super().__init__(cache_dir=cache_dir, project_name=project_name,
                         ui_class_name=ui_class_name)

    _s3_client = None

    @property
    def bucket_name(self) -> str:
        return self._bucket_name

    @property
    def s3_client(self):
        if self._s3_client is None:
            s3_config = Config(signature_version=UNSIGNED)
            self._s3_client = boto3.client('s3',
                                           config=s3_config)
        return self._s3_client

    def _list_all_manifests(self):
        """
        Return a list of all of the file names of the manifests associated
        with this dataset
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        subset_iterator = paginator.paginate(
            Bucket=self._bucket_name,
            Prefix=self.manifest_prefix
        )

        output = []
        for subset in subset_iterator:
            if 'Contents' in subset:
                for obj in subset['Contents']:
                    output.append(pathlib.Path(obj['Key']).name)

        output.sort()
        return output

    def _download_manifest(self,
                           manifest_name: str):
        """
        Download a manifest from the dataset

        Parameters
        ----------
        manifest_name: str
            The name of the manifest to load. Must be an element in
            self.manifest_file_names
        """

        manifest_key = self.manifest_prefix + manifest_name
        response = self.s3_client.get_object(Bucket=self._bucket_name,
                                             Key=manifest_key)

        filepath = os.path.join(self._cache_dir, manifest_name)

        with open(filepath, 'wb') as f:
            for chunk in response['Body'].iter_chunks():
                f.write(chunk)

    def _download_file(self, file_attributes):
        """
        Check if a file exists locally. If it does not, download it
        and return True. Return False otherwise.

        Parameters
        ----------
        file_attributes: CacheFileAttributes
            Describes the file to download

        Returns
        -------
        bool
            True if the file was downloaded; False otherwise

        Raises
        ------
        RuntimeError
            If the path to the directory where the file is to be saved
            points to something that is not a directory.

        RuntimeError
            If it is not able to successfully download the file after
            10 iterations
        """
        was_downloaded = False

        local_path = file_attributes.local_path

        local_dir = pathlib.Path(safe_system_path(str(local_path.parents[0])))

        # make sure Windows references to Allen Institute
        # local networked file system get handled correctly
        local_path = pathlib.Path(safe_system_path(str(local_path)))

        # using os here rather than pathlib because safe_system_path
        # returns a str
        os.makedirs(local_dir, exist_ok=True)
        if not os.path.isdir(local_dir):
            raise RuntimeError(f"{local_dir}\n"
                               "is not a directory")

        bucket_name = bucket_name_from_url(file_attributes.url)
        obj_key = relative_path_from_url(file_attributes.url)

        n_iter = 0
        max_iter = 10  # maximum number of times to try download

        version_id = file_attributes.version_id

        pbar = None
        if not self._file_exists(file_attributes):
            response = self.s3_client.list_object_versions(Bucket=bucket_name,
                                                           Prefix=str(obj_key))
            object_info = [i for i in response["Versions"]
                           if i["VersionId"] == version_id][0]
            pbar = tqdm(desc=object_info["Key"].split("/")[-1],
                             total=object_info["Size"],
                             unit_scale=True,
                             unit_divisor=1000.,
                             unit="MB")

        while not self._file_exists(file_attributes):
            was_downloaded = True
            response = self.s3_client.get_object(Bucket=bucket_name,
                                                 Key=str(obj_key),
                                                 VersionId=version_id)

            if 'Body' in response:
                with open(local_path, 'wb') as out_file:
                    for chunk in response['Body'].iter_chunks():
                        out_file.write(chunk)
                        pbar.update(len(chunk))

            # Verify the hash of the downloaded file
            full_path = file_attributes.local_path.resolve()
            test_checksum = file_hash_from_path(full_path)
            if test_checksum != file_attributes.file_hash:
                file_attributes.local_path.exists()
                file_attributes.local_path.unlink()

            n_iter += 1
            if n_iter > max_iter:
                pbar.close()
                raise RuntimeError("Could not download\n"
                                   f"{file_attributes}\n"
                                   "In {max_iter} iterations")
        if pbar is not None:
            pbar.close()

        return was_downloaded


def relative_path_from_url(url: str) -> str:
    """
    Read in a url and return the relative path of the object

    Parameters
    ----------
    url: str
        The url of the object whose path you want

    Returns
    -------
    str:
        Relative path of the object

    Notes
    -----
    This method returns a str rather than a pathlib.Path because
    it is used to get the S3 object Key from a URL. If using
    Pathlib.path on a Windows system, the '/' will get transformed
    into '\', confusing S3.
    """
    url_params = url_parse.urlparse(url)
    return url_params.path[1:]

def safe_system_path(file_name):
    if platform.system() == "Windows":
        return linux_to_windows(file_name)
    else:
        return convert_from_titan_linux(os.path.normpath(file_name))
    
def convert_from_titan_linux(file_name):
    # Lookup table mapping project to program
    project_to_program = {
        "neuralcoding": "braintv",
        '0378': "celltypes",
        'conn': "celltypes",
        'ctyconn': "celltypes",
        'humancelltypes': "celltypes",
        'mousecelltypes': "celltypes",
        'shotconn': "celltypes",
        'synapticphys': "celltypes",
        'whbi': "celltypes",
        'wijem': "celltypes"
    }
    # Tough intermediary state where we have old paths
    # being translated to new paths
    m = re.match('/projects/([^/]+)/vol1/(.*)', file_name)
    if m:
        newpath = os.path.normpath(os.path.join(
            '/allen',
            'programs',
            project_to_program.get(m.group(1), 'undefined'),
            'production',
            m.group(1),
            m.group(2)
        ))
        return newpath
    return file_name


def linux_to_windows(file_name):
    # Lookup table mapping project to program
    project_to_program = {
        "neuralcoding": "braintv",
        '0378': "celltypes",
        'conn': "celltypes",
        'ctyconn': "celltypes",
        'humancelltypes': "celltypes",
        'mousecelltypes': "celltypes",
        'shotconn': "celltypes",
        'synapticphys': "celltypes",
        'whbi': "celltypes",
        'wijem': "celltypes"
    }

    # Simple case for new world order
    m = re.match('/allen', file_name)
    if m:
        return "\\" + file_name.replace('/', '\\')

    # /data/ paths are being retained (for now)
    # this will need to be extended to map directories to
    # /allen/{programs,aibs}/workgroups/foo
    m = re.match('/data/([^/]+)/(.*)', file_name)
    if m:
        return os.path.normpath(
            os.path.join('\\\\aibsdata', m.group(1), m.group(2)))

    # Tough intermediary state where we have old paths
    # being translated to new paths
    m = re.match('/projects/([^/]+)/vol1/(.*)', file_name)
    if m:
        newpath = os.path.normpath(os.path.join(
            '\\\\allen',
            'programs',
            project_to_program.get(m.group(1), 'undefined'),
            'production',
            m.group(1),
            m.group(2)
        ))
        return newpath

    # No matches found.  Clean up and return path given to us
    return os.path.normpath(file_name)


def bucket_name_from_url(url):
    """
    Read in a URL and return the name of the AWS S3 bucket it points towards.

    Parameters
    ----------
    URL: str
        A generic URL, suitable for retrieving an S3 object via an
        HTTP GET request.

    Returns
    -------
    str
        An AWS S3 bucket name. Note: if 's3.amazonaws.com' does not occur in
        the URL, this method will return None and emit a warning.

    Note
    -----
    URLs passed to this method should conform to the "new" scheme as described
    here
    https://aws.amazon.com/blogs/aws/amazon-s3-path-deprecation-plan-the-rest-of-the-story/
    """
    s3_pattern = re.compile('\.s3[\.,a-z,0-9,\-]*\.amazonaws.com')  # noqa: W605, E501
    url_params = url_parse.urlparse(url)
    raw_location = url_params.netloc
    s3_match = s3_pattern.search(raw_location)

    if s3_match is None:
        warnings.warn(f"{s3_pattern} does not occur in url {url}")
        return None

    s3_match = raw_location[s3_match.start():s3_match.end()]
    return url_params.netloc.replace(s3_match, '')

