import os
import json

from .manifest import Manifest, ManifestBuilder, ManifestVersionError


class Cache:
    def __init__(self,
                 manifest=None,
                 cache=True,
                 version=None,
                 **kwargs):
        self.cache = cache
        if version is None and hasattr(self, 'MANIFEST_VERSION'):
            version = self.MANIFEST_VERSION
        self.load_manifest(manifest, version)

    def load_manifest(self, file_name, version=None):
        if file_name is not None:
            if not os.path.exists(file_name):

                # make the directory if it doesn't exist already
                dirname = os.path.dirname(file_name)
                if dirname:
                    Manifest.safe_mkdir(dirname)

                self.build_manifest(file_name)

            try:
                with open(file_name, "rb") as f:
                    json_string = f.read().decode("utf-8")
                    if len(json_string) == 0:
                        json_string = "{}"
                    json_obj = json.loads(json_string)

                self.manifest = Manifest(
                    json_obj['manifest'],
                    os.path.dirname(file_name),
                    version=version)
            except ManifestVersionError as e:
                if e.outdated is True:
                    intro = "is out of date"
                elif e.outdated is False:
                    intro = "was made with a newer version of the AllenSDK"
                elif e.outdated is None:
                    intro = "version did not match the expected version"

                ref_url = "https://github.com/alleninstitute/allensdk/wiki"
                raise ManifestVersionError(("Your manifest file (%s) %s" +
                                            " (its version is '%s', but" +
                                            " version '%s' is expected). " +
                                            " Please remove this file" +
                                            " and it will be regenerated for" +
                                            " you the next time you" +
                                            " instantiate this class." +
                                            " WARNING: There may be new data" +
                                            " files available that replace" +
                                            " the ones you already have" +
                                            " downloaded. Read the notes" +
                                            " for this release for more" +
                                            " details on what has changed" +
                                            " (%s).") %
                                           (file_name, intro,
                                            e.found_version, e.version,
                                            ref_url),
                                           e.version, e.found_version)

            self.manifest_path = file_name

        else:
            self.manifest = None


    def build_manifest(self, file_name):
        manifest_builder = ManifestBuilder()
        manifest_builder.set_version(self.MANIFEST_VERSION)
        manifest_builder = self.add_manifest_paths(manifest_builder)
        manifest_builder.write_json_file(file_name)


    def add_manifest_paths(self, manifest_builder):
        manifest_builder.add_path('BASEDIR', '.')
        if hasattr(self, 'MANIFEST_CONFIG'):
            for key, config in self.MANIFEST_CONFIG.items():
                manifest_builder.add_path(key, **config)
        return manifest_builder


    def get_cache_path(self, file_name, manifest_key, *args):
        if self.cache:
            if file_name:
                return file_name
            elif self.manifest:
                return self.manifest.get_path(manifest_key, *args)

        return None

def get_default_manifest_file(cache_name):
    return os.environ.get(
        '{}_MANIFEST'.format(cache_name.upper()),
        '{}/manifest.json'.format(cache_name.lower())
    )