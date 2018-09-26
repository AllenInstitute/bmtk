# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import json
from jsonschema import Draft4Validator
from jsonschema.exceptions import ValidationError
import pandas as pd


class SimConfigValidator(Draft4Validator):
    """
    A JSON Schema validator class that will store a schema (passed into the constructor) and validate a json file.
        It has all the functionality of the JSONSchema format, plus includes special types and parameters like making
        sure a value is a file or directory type, checking csv files, etc.

        To Use:
        validator = SimConfigValidator(json_schema.json)
        validator.validate(file.json)
    """

    def __init__(self, schema, types=(), resolver=None, format_checker=None, file_formats=()):
        super(SimConfigValidator, self).__init__(schema, types, resolver, format_checker)

        # custom parameter
        self.VALIDATORS["exists"] = self._check_path

        self._file_formats = {}  # the "file_format" property the validity of a (non-json) file.
        for (name, schema) in file_formats:
            self._file_formats[name] = self._parse_file_formats(schema)
        self.VALIDATORS["file_format"] = self._validate_file

    def is_type(self, instance, dtype):
        # override type since checking for file and directory type is potentially more complicated.
        if dtype == "directory":
            return self._is_directory_type(instance)

        elif dtype == "file":
            return self._is_file_type(instance)

        else:
            return super(SimConfigValidator, self).is_type(instance, dtype)

    def _is_directory_type(self, instance):
        """Check if instance value is a valid directory file path name

        :param instance: string that represents a directory path
        :return: True if instance is a valid dir path (even if it doesn't exists).
        """
        # Always return true for now, rely on the "exists" property (_check_path) to actual determine if file exists.
        # TODO: check that instance string is a valid path string, even if path doesn't yet exists.
        return True

    def _is_file_type(self, instance):
        """Check if instance value is a valid file path.

        :param instance: string of file path
        :return: True if instance is a valid file path (but doesn't necessary exists), false otherwise.
        """
        # Same issue as with _is_directory_type
        return True

    def _parse_file_formats(self, schema_file):
        # Open the schema file and based on "file_type" property create a Format validator
        schema = json.load(open(schema_file, 'r'))
        if schema['file_type'] == 'csv':
            return self._CSVFormat(schema)
        else:
            return Exception("No format found")

    @staticmethod
    def _check_path(validator, schema_bool, path, schema):
        """Makes sure a file/directory exists or doesn't based on the "exists" property in the schema

        :param validator:
        :param schema_bool: True means file must exists, False means file should not exists
        :param path: path of the file
        :param schema:
        :return: True if schema is satisfied.
        """
        assert(schema['type'] == 'directory' or schema['type'] == 'file')
        path_exists = os.path.exists(path)
        if path_exists != schema_bool:
            raise ValidationError("{} {} exists.".format(path, "already" if path_exists else "does not"))

    def _validate_file(self, validator, file_format, file_path, schema):
        file_validator = self._file_formats.get(file_format, None)
        if file_validator is None:
            raise ValidationError("Could not find file validator {}".format(file_format))

        if not file_validator.check(file_path):
            raise ValidationError("File {} could not be validated against {}.".format(file_path, file_format))

    # A series of validators for indivdiual types of files. All of them should have a check(file) function that returns
    # true only when it is formated correctly.
    class _CSVFormat(object):
        def __init__(self, schema):
            self._properties = schema['file_properties']
            self._required_columns = [header for header, props in schema['columns'].items() if props['required']]

        def check(self, file_name):
            csv_headers = set(pd.read_csv(file_name, nrows=0, **self._properties).columns)
            for col in self._required_columns:
                if col not in csv_headers:
                    return False

            return True
