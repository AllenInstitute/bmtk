import os
import sys
import errno
import json
from pathlib import Path

from .utils import json_handler



class ManifestVersionError(Exception): 
    @property
    def outdated(self):
        try:
            return self.found_version < self.version 
        except TypeError:
            return

    def __init__(self, message, version, found_version):
        super(ManifestVersionError, self).__init__(message)
        self.found_version = found_version
        self.version = version


class ManifestBuilder(object):
    df_columns = ['key', 'parent_key', 'spec', 'type', 'format']

    def __init__(self):
        self.path_info = []
        self.sections = {}

    def set_version(self, value):
        self.path_info.append({'type': Manifest.VERSION, 'value': value})

    def add_path(self, key, spec,
                 typename='dir',
                 parent_key=None,
                 format=None):
        entry = {
            'key': key,
            'type': typename,
            'spec': spec}

        if format is not None:
            entry['format'] = format

        if parent_key is not None:
            entry['parent_key'] = parent_key

        self.path_info.append(entry)

    def write_json_file(self, path, overwrite=False):
        mode = 'wb'

        if overwrite is True:
            mode = 'wb+'

        json_string = self.write_json_string()

        with open(path, mode) as f:
            try:
                f.write(json_string)   # Python 2.7
            except TypeError:
                f.write(bytes(json_string, 'utf-8'))  # Python 3

    def write_json_string(self):
        config = self.get_config()
    
        return json.dumps(
            config,
            indent=2,
            # ignore_nan=True,
            default=json_handler,
            # iterable_as_array=True,
        )

    


    def get_config(self):
        wrapper = {"manifest": self.path_info}
        for section in self.sections.values():
            wrapper.update(section)

        return wrapper



class Manifest(object):
    DIR = 'dir'
    FILE = 'file'
    DIRNAME = 'dir_name'
    VERSION = 'manifest_version'

    def __init__(self, config=None, relative_base_dir='.', version=None):
        self.path_info = {}
        self.relative_base_dir = relative_base_dir

        if config is not None:
            self.load_config(config, version=version)

    def load_config(self, config, version=None):
        ''' Load paths into the manifest from an Allen SDK config section.

        Parameters
        ----------
        config : Config
            Manifest section of an Allen SDK config.
        '''
        found_version = None
        for path_info in config:
            path_type = path_info['type']
            path_format = None
            if 'format' in path_info:
                path_format = path_info['format']

            if path_type == 'file':
                try:
                    parent_key = path_info['parent_key']
                except:
                    parent_key = None

                self.add_file(path_info['key'],
                              path_info['spec'],
                              parent_key,
                              path_format)
            elif path_type == 'dir':
                try:
                    parent_key = path_info['parent_key']
                except:
                    parent_key = None

                spec = path_info['spec']
                absolute = False
                if spec[0] == '/':
                    absolute = True
                self.add_path(path_info['key'],
                              path_info['spec'],
                              path_type,
                              absolute,
                              path_format,
                              parent_key)

            elif path_type == self.VERSION:
                found_version = path_info['value']
            else:
                Manifest.log.warning("Unknown path type in manifest: %s" %
                                     (path_type))


        if found_version != version:
            raise ManifestVersionError("", version, found_version)
        self.version = version

    def add_path(self, key, path, path_type=DIR,
                 absolute=True, path_format=None, parent_key=None):
        '''Insert a new entry.

        Parameters
        ----------
        key : string
            Identifier for referencing the entry.
        path : string
            Specification for a path using %s, %d style substitution.
        path_type : string enumeration
            'dir' (default) or 'file'
        absolute : boolean
            Is the spec relative to the process current directory.
        path_format : string, optional
            Indicate a known file type for further parsing.
        parent_key : string
            Refer to another entry.
        '''
        if parent_key:
            path_args = []

            try:
                parent_path = self.path_info[parent_key]['spec']
                path_args.append(parent_path)
            except:
                Manifest.log.error(
                    "cannot resolve directory key %s" % (parent_key))
                raise
            path_args.extend(path.split('/'))
            path = os.path.join(*path_args)

        # TODO: relative paths need to be considered better
        if absolute is True:
            path = os.path.abspath(path)
        else:
            path = os.path.abspath(os.path.join(self.relative_base_dir, path))

        if path_type == Manifest.DIRNAME:
            path = os.path.dirname(path)

        self.path_info[key] = {'type': path_type,
                               'spec': path}

        if path_type == Manifest.FILE and path_format is not None:
            self.path_info[key]['format'] = path_format

    def add_file(self,
                 file_key,
                 file_name,
                 dir_key=None,
                 path_format=None):
        '''Insert a new file entry.

        Parameters
        ----------
        file_key : string
            Reference to the entry.
        file_name : string
            Subtitutions of the %s, %d style allowed.
        dir_key : string
            Reference to the parent directory entry.
        path_format : string, optional
            File type for further parsing.
        '''
        path_args = []

        if dir_key:
            try:
                dir_path = self.path_info[dir_key]['spec']
                path_args.append(dir_path)
            except:
                Manifest.log.error(
                    "cannot resolve directory key %s" % (dir_key))
                raise
        elif not file_name.startswith('/'):
            path_args.append(os.curdir)
        else:
            path_args.append(os.path.sep)

        path_args.extend(file_name.split('/'))
        file_path = os.path.join(*path_args)

        self.path_info[file_key] = {'type': Manifest.FILE,
                                    'spec': file_path}

        if path_format:
            self.path_info[file_key]['format'] = path_format

    @classmethod
    def safe_mkdir(cls, directory):
        '''Create path if not already there.

        Parameters
        ----------
        directory : string
            create it if it doesn't exist

        Returns
        -------
        leftmost : string 
            most rootward directory created

        '''

        parts = Path(directory).parts
        sub_paths = [Path(parts[0])]
        for part in parts[1:]:
            sub_paths.append(sub_paths[-1] / part)

        leftmost = None
        for sub_path in sub_paths:
            if not sub_path.exists():
                leftmost = str(sub_path)

        try:
            os.makedirs(directory)
        except OSError as e:
            if ((sys.platform == "darwin") and (e.errno == errno.EISDIR) and \
                (e.filename == "/")):
                # undocumented behavior of mkdir on OSX where for / it raises
                # EISDIR and not EEXIST
                # https://bugs.python.org/issue24231 (old but still holds true)
                pass
            elif sys.platform == "win32" and e.errno == errno.EACCES:
                root_path = os.path.abspath(os.sep)
                if e.filename == root_path or \
                   e.filename == root_path.replace("\\", "/"):
                    # When attempting to os.makedirs the root drive letter on
                    # Windows, EACCES is raised, not EEXIST
                    pass
                else:
                    raise
            elif e.errno == errno.EEXIST:
                pass
            else:
                raise

        return leftmost

    @classmethod
    def safe_make_parent_dirs(cls, file_name):
        ''' Create a parent directories for file.

        Parameters
        ----------
        file_name : string

        Returns
        -------
        leftmost : string 
            most rootward directory created

        '''

        dirname = os.path.dirname(file_name)

        # do nothing if there are no parent directories
        if not dirname:
            return

        return Manifest.safe_mkdir(dirname)


    def get_path(self, path_key, *args):
        '''Retrieve an entry with substitutions.

        Parameters
        ----------
        path_key : string
            Refer to the entry to retrieve.
        args : any types, optional
           arguments to be substituted into the path spec for %s, %d, etc.

        Returns
        -------
        string
            Path with parent structure and substitutions applied.
        '''
        path_spec = self.path_info[path_key]['spec']

        if args is not None and len(args) != 0:
            path = path_spec % args
        else:
            path = path_spec

        return path

