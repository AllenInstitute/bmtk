import os
import warnings
import pandas as pd
import requests
from contextlib import closing
import urllib
import json
from pathlib import Path

from jinja2 import Template


class Api(object):
    # _log = logging.getLogger('allensdk.api.api')
    # _file_download_log = logging.getLogger('allensdk.api.api.retrieve_file_over_http')
    default_api_url = 'http://api.brain-map.org'
    download_url = 'http://download.alleninstitute.org'

    def __init__(self, api_base_url_string=None):
        if api_base_url_string is None:
            api_base_url_string = Api.default_api_url

        self.set_api_urls(api_base_url_string)
        self.default_working_directory = os.getcwd()

    def set_api_urls(self, api_base_url_string):
        '''Set the internal RMA and well known file download endpoint urls
        based on a api server endpoint.

        Parameters
        ----------
        api_base_url_string : string
            url of the api to point to
        '''
        self.api_url = api_base_url_string

        # http://help.brain-map.org/display/api/Downloading+a+WellKnownFile
        self.well_known_file_endpoint = api_base_url_string + \
            '/api/v2/well_known_file_download'

        # http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data
        self.grid_data_endpoint = api_base_url_string + '/grid_data'

        # http://help.brain-map.org/display/api/Downloading+and+Displaying+SVG
        self.svg_endpoint = api_base_url_string + '/api/v2/svg'
        self.svg_download_endpoint = api_base_url_string + '/api/v2/svg_download'

        # http://help.brain-map.org/display/api/Downloading+an+Ontology%27s+Structure+Graph
        self.structure_graph_endpoint = api_base_url_string + \
            '/api/v2/structure_graph_download'

        # http://help.brain-map.org/display/api/Searching+a+Specimen+or+Structure+Tree
        self.tree_search_endpoint = api_base_url_string + '/api/v2/tree_search'

        # http://help.brain-map.org/display/api/Searching+Annotated+SectionDataSets
        self.annotated_section_data_sets_endpoint = api_base_url_string + \
            '/api/v2/annotated_section_data_sets'
        self.compound_annotated_section_data_sets_endpoint = api_base_url_string + \
            '/api/v2/compound_annotated_section_data_sets'

        # http://help.brain-map.org/display/api/Image-to-Image+Synchronization#Image-to-ImageSynchronization-ImagetoImage
        self.image_to_atlas_endpoint = api_base_url_string + '/api/v2/image_to_atlas'
        self.image_to_image_endpoint = api_base_url_string + '/api/v2/image_to_image'
        self.image_to_image_2d_endpoint = api_base_url_string + '/api/v2/image_to_image_2d'
        self.reference_to_image_endpoint = api_base_url_string + '/api/v2/reference_to_image'
        self.image_to_reference_endpoint = api_base_url_string + '/api/v2/image_to_reference'
        self.structure_to_image_endpoint = api_base_url_string + '/api/v2/structure_to_image'

        # http://help.brain-map.org/display/mouseconnectivity/API
        self.section_image_download_endpoint = api_base_url_string + \
            '/api/v2/section_image_download'
        self.atlas_image_download_endpoint = api_base_url_string + \
            '/api/v2/atlas_image_download'
        self.projection_image_download_endpoint = api_base_url_string + \
            '/api/v2/projection_image_download'
        self.image_download_endpoint = api_base_url_string + \
            '/api/v2/image_download'
        self.informatics_archive_endpoint = Api.download_url + '/informatics-archive'

        self.rma_endpoint = api_base_url_string + '/api/v2/data'

    def set_default_working_directory(self, working_directory):
        '''Set the working directory where files will be saved.

        Parameters
        ----------
        working_directory : string
             the absolute path string of the working directory.
        '''
        self.default_working_directory = working_directory

    def read_data(self, parsed_json):
        '''Return the message data from the parsed query.

        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.

        Notes
        -----
        See `API Response Formats - Response Envelope <http://help.brain-map.org/display/api/API+Response+Formats#APIResponseFormats-ResponseEnvelope>`_
        for additional documentation.
        '''
        return parsed_json['msg']

    def json_msg_query(self, url, dataframe=False):
        ''' Common case where the url is fully constructed
            and the response data is stored in the 'msg' field.

        Parameters
        ----------
        url : string
            Where to get the data in json form
        dataframe : boolean
            True converts to a pandas dataframe, False (default) doesn't

        Returns
        -------
        dict or DataFrame
            returned data; type depends on dataframe option
        '''

        data = self.do_query(lambda *a, **k: url,
                             self.read_data)

        if dataframe is True:
            warnings.warn("dataframe argument is deprecated", DeprecationWarning)
            data = pd.DataFrame(data)

        return data

    def do_query(self, url_builder_fn, json_traversal_fn, *args, **kwargs):
        '''Bundle an query url construction function
        with a corresponding response json traversal function.

        Parameters
        ----------
        url_builder_fn : function
            A function that takes parameters and returns an rma url.
        json_traversal_fn : function
            A function that takes a json-parsed python data structure and returns data from it.
        post : boolean, optional kwarg
            True does an HTTP POST, False (default) does a GET
        args : arguments
            Arguments to be passed to the url builder function.
        kwargs : keyword arguments
            Keyword arguments to be passed to the rma builder function.

        Returns
        -------
        any type
            The data extracted from the json response.

        Examples
        --------
        `A simple Api subclass example
        <data_api_client.html#creating-new-api-query-classes>`_.
        '''
        api_url = url_builder_fn(*args, **kwargs)

        post = kwargs.get('post', False)

        json_parsed_data = self.retrieve_parsed_json_over_http(api_url, post)

        return json_traversal_fn(json_parsed_data)

    def do_rma_query(self, rma_builder_fn, json_traversal_fn, *args, **kwargs):
        '''Bundle an RMA query url construction function
        with a corresponding response json traversal function.

        ..note:: Deprecated in AllenSDK 0.9.2
            `do_rma_query` will be removed in AllenSDK 1.0, it is replaced by
            `do_query` because the latter is more general.

        Parameters
        ----------
        rma_builder_fn : function
            A function that takes parameters and returns an rma url.
        json_traversal_fn : function
            A function that takes a json-parsed python data structure and returns data from it.
        args : arguments
            Arguments to be passed to the rma builder function.
        kwargs : keyword arguments
            Keyword arguments to be passed to the rma builder function.

        Returns
        -------
        any type
            The data extracted from the json response.

        Examples
        --------
        `A simple Api subclass example
        <data_api_client.html#creating-new-api-query-classes>`_.
        '''
        return self.do_query(rma_builder_fn, json_traversal_fn, *args, **kwargs)

    def load_api_schema(self):
        '''Download the RMA schema from the current RMA endpoint

        Returns
        -------
        dict
            the parsed json schema message

        Notes
        -----
        This information and other
        `Allen Brain Atlas Data Portal Data Model <http://help.brain-map.org/display/api/Data+Model>`_
        documentation is also available as a
        `Class Hierarchy <http://api.brain-map.org/class_hierarchy>`_
        and `Class List <http://api.brain-map.org/class_hierarchy>`_.

        '''
        schema_url = self.rma_endpoint + '/enumerate.json'
        json_parsed_schema_data = self.retrieve_parsed_json_over_http(
            schema_url)

        return json_parsed_schema_data

    def construct_well_known_file_download_url(self, well_known_file_id):
        '''Join data api endpoint and id.

        Parameters
        ----------
        well_known_file_id : integer or string representing an integer
            well known file id

        Returns
        -------
        string
            the well-known-file download url for the current api api server

        See Also
        --------
        retrieve_file_over_http: Can be used to retrieve the file from the url.
        '''
        return self.well_known_file_endpoint + '/' + str(well_known_file_id)

    def cleanup_truncated_file(self, file_path):
        '''Helper for removing files.

        Parameters
        ----------
        file_path : string
            Absolute path including the file name to remove.'''
        try:
            os.remove(file_path)
        except OSError as e:
            warnings(f'{e}')

    def retrieve_file_over_http(self, url, file_path, zipped=False):
        '''Get a file from the data api and save it.

        Parameters
        ----------
        url : string
            Url[1]_ from which to get the file.
        file_path : string
            Absolute path including the file name to save.
        zipped : bool, optional
            If true, assume that the response is a zipped directory and attempt 
            to extract contained files into the directory containing file_path. 
            Default is False.

        See Also
        --------
        construct_well_known_file_download_url: Can be used to construct the url.

        References
        ----------
        .. [1] Allen Brain Atlas Data Portal: `Downloading a WellKnownFile <http://help.brain-map.org/display/api/Downloading+a+WellKnownFile>`_.
        '''

        # self._file_download_log.info("Downloading URL: %s", url)

        try:
            if zipped:
                stream_zip_directory_over_http(url, os.path.dirname(file_path))
            else:
                stream_file_over_http(url, file_path)

        except Exception as e:
            # self._file_download_log.error("Couldn't retrieve file %s from %s" % (file_path, url))
            # self.cleanup_truncated_file(file_path)
            raise e


    def retrieve_parsed_json_over_http(self, url, post=False):
        '''Get the document and put it in a Python data structure

        Parameters
        ----------
        url : string
            Full API query url.
        post : boolean
            True does an HTTP POST, False (default) encodes the URL and does a GET

        Returns
        -------
        dict
            Result document as parsed by the JSON library.
        '''
        # self._log.info("Downloading URL: %s", url)
        
        if post is False:
            url = requests.utils.quote(url, ';/?:@&=+$,')
            response = urllib.request.urlopen(url)
            json_string = response.read().decode("utf-8")
            data = json.loads(json_string)
        else:
            data = json_utilities.read_url_post(url)

        return data

    def retrieve_xml_over_http(self, url):
        '''Get the document and put it in a Python data structure

        Parameters
        ----------
        url : string
            Full API query url.

        Returns
        -------
        string
            Unparsed xml string.
        '''
        self._log.info("Downloading URL: %s", url)
                
        response = requests.get(url)

        return response.content


def stream_zip_directory_over_http(url, directory, members=None, timeout=(9.05, 31.1)):
    ''' Supply an http get request and stream the response to a file.

    Parameters
    ----------
    url : str
        Send the request to this url
    directory : str
        Extract the response to this directory
    members : list of str, optional
        Extract only these files
    timeout : float or tuple of float, optional
        Specify a timeout for the request. If a tuple, specify seperate connect 
        and read timeouts.

    '''

    buf = io.BytesIO()

    with closing( requests.get(url, stream=True, timeout=timeout) ) as request:
        stream.stream_response_to_file( request, buf )

    zipper = zipfile.ZipFile(buf)
    zipper.extractall(path=directory, members=members)
    zipper.close()


def stream_file_over_http(url, file_path, timeout=(9.05, 31.1)):
    ''' Supply an http get request and stream the response to a file.

    Parameters
    ----------
    url : str
        Send the request to this url
    file_path : str
        Stream the response to this path
    timeout : float or tuple of float, optional
        Specify a timeout for the request. If a tuple, specify seperate connect 
        and read timeouts.

    '''
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(requests.get(url, stream=True, timeout=timeout)) as response:

        response.raise_for_status()
        with open(file_path, 'wb') as fil:
            for chunk in response.iter_content(chunk_size=8192):  # Adjust chunk_size as needed
                if chunk:  # Filter out keep-alive new chunks
                    fil.write(chunk)
            # stream.stream_response_to_file(response, path=fil)



class RmaApi(Api):
    '''
    See: `RESTful Model Access (RMA) <http://help.brain-map.org/display/api/RESTful+Model+Access+%28RMA%29>`_
    '''
    MODEL = 'model::'
    PIPE = 'pipe::'
    SERVICE = 'service::'
    CRITERIA = 'rma::criteria'
    INCLUDE = 'rma::include'
    OPTIONS = 'rma::options'
    ORDER = 'order'
    NUM_ROWS = 'num_rows'
    ALL = 'all'
    START_ROW = 'start_row'
    COUNT = 'count'
    ONLY = 'only'
    EXCEPT = 'except'
    EXCPT = 'excpt'
    TABULAR = 'tabular'
    DEBUG = 'debug'
    PREVIEW = 'preview'
    TRUE = 'true'
    FALSE = 'false'
    IS = '$is'
    EQ = '$eq'

    def __init__(self, base_uri=None):
        super(RmaApi, self).__init__(base_uri)

    def build_query_url(self,
                        stage_clauses,
                        fmt='json'):
        '''Combine one or more RMA query stages into a single RMA query.

        Parameters
        ----------
        stage_clauses : list of strings
            subqueries
        fmt : string, optional
            json (default), xml, or csv

        Returns
        -------
        string
            complete RMA url
        '''
        if not type(stage_clauses) is list:
            stage_clauses = [stage_clauses]

        url = ''.join([
            self.rma_endpoint,
            '/query.',
            fmt,
            '?q=',
            ','.join(stage_clauses)])

        return url

    def model_stage(self,
                    model,
                    **kwargs):
        '''Construct a model stage of an RMA query string.

        Parameters
        ----------
        model : string
            The top level data type
        filters : dict
            key, value comparisons applied to the top-level model to narrow the results.
        criteria : string
            raw RMA criteria clause to choose what object are returned
        include : string
            raw RMA include clause to return associated objects
        only : list of strings, optional
            to be joined into an rma::options only filter to limit what data is returned
        except : list of strings, optional
            to be joined into an rma::options except filter to limit what data is returned
        tabular : list of string, optional
            return columns as a tabular data structure rather than a nested tree.
        count : boolean, optional
            False to skip the extra database count query.
        debug : string, optional
            'true', 'false' or 'preview'
        num_rows : int or string, optional
            how many database rows are returned (may not correspond directly to JSON tree structure)
        start_row : int or string, optional
            which database row is start of returned data  (may not correspond directly to JSON tree structure)


        Notes
        -----
        See `RMA Path Syntax <http://help.brain-map.org/display/api/RMA+Path+Syntax#RMAPathSyntax-DoubleColonforAxis>`_
        for a brief overview of the normalized RMA syntax.
        Normalized RMA syntax differs from the legacy syntax
        used in much of the RMA documentation.
        Using the &debug=true option with an RMA URL will include debugging information in the
        response, including the normalized query.
        '''
        clauses = [RmaApi.MODEL + model]

        filters = kwargs.get('filters', None)

        if filters is not None:
            clauses.append(self.filters(filters))

        criteria = kwargs.get('criteria', None)

        if criteria is not None:
            clauses.append(',')
            clauses.append(RmaApi.CRITERIA)
            clauses.append(',')
            clauses.extend(criteria)

        include = kwargs.get('include', None)

        if include is not None:
            clauses.append(',')
            clauses.append(RmaApi.INCLUDE)
            clauses.append(',')
            clauses.extend(include)

        options_clause = self.options_clause(**kwargs)

        if options_clause != '':
            clauses.append(',')
            clauses.append(options_clause)

        stage = ''.join(clauses)

        return stage

    def pipe_stage(self,
                   pipe_name,
                   parameters):
        '''Connect model and service stages via their JSON responses.

        Notes
        -----
        See: `Service Pipelines <http://help.brain-map.org/display/api/Service+Pipelines>`_
        and
        `Connected Services and Pipes <http://help.brain-map.org/display/api/Connected+Services+and+Pipes>`_
        '''
        clauses = [RmaApi.PIPE + pipe_name]

        clauses.append(self.tuple_filters(parameters))

        stage = ''.join(clauses)

        return stage

    def service_stage(self,
                      service_name,
                      parameters=None):
        '''Construct an RMA query fragment to send a request to a connected service.

        Parameters
        ----------
        service_name : string
            Name of a documented connected service.
        parameters : dict
            key-value pairs as in the online documentation.

        Notes
        -----
        See: `Service Pipelines <http://help.brain-map.org/display/api/Service+Pipelines>`_
        and
        `Connected Services and Pipes <http://help.brain-map.org/display/api/Connected+Services+and+Pipes>`_
        '''
        clauses = [RmaApi.SERVICE + service_name]

        if parameters is not None:
            clauses.append(self.tuple_filters(parameters))

        stage = ''.join(clauses)

        return stage

    def model_query(self, *args, **kwargs):
        '''Construct and execute a model stage of an RMA query string.

        Parameters
        ----------
        model : string
            The top level data type
        filters : dict
            key, value comparisons applied to the top-level model to narrow the results.
        criteria : string
            raw RMA criteria clause to choose what object are returned
        include : string
            raw RMA include clause to return associated objects
        only : list of strings, optional
            to be joined into an rma::options only filter to limit what data is returned
        except : list of strings, optional
            to be joined into an rma::options except filter to limit what data is returned
        excpt : list of strings, optional
            synonym for except parameter to avoid a reserved word conflict.
        tabular : list of string, optional
            return columns as a tabular data structure rather than a nested tree.
        count : boolean, optional
            False to skip the extra database count query.
        debug : string, optional
            'true', 'false' or 'preview'
        num_rows : int or string, optional
            how many database rows are returned (may not correspond directly to JSON tree structure)
        start_row : int or string, optional
            which database row is start of returned data  (may not correspond directly to JSON tree structure)


        Notes
        -----
        See `RMA Path Syntax <http://help.brain-map.org/display/api/RMA+Path+Syntax#RMAPathSyntax-DoubleColonforAxis>`_
        for a brief overview of the normalized RMA syntax.
        Normalized RMA syntax differs from the legacy syntax
        used in much of the RMA documentation.
        Using the &debug=true option with an RMA URL will include debugging information in the
        response, including the normalized query.
        '''
        return self.json_msg_query(
            self.build_query_url(
                self.model_stage(*args, **kwargs)))

    def service_query(self, *args, **kwargs):
        '''Construct and Execute a single-stage RMA query
        to send a request to a connected service.

        Parameters
        ----------
        service_name : string
            Name of a documented connected service.
        parameters : dict
            key-value pairs as in the online documentation.

        Notes
        -----
        See: `Service Pipelines <http://help.brain-map.org/display/api/Service+Pipelines>`_
        and
        `Connected Services and Pipes <http://help.brain-map.org/display/api/Connected+Services+and+Pipes>`_
        '''
        return self.json_msg_query(
            self.build_query_url(
                self.service_stage(*args, **kwargs)))

    def options_clause(self, **kwargs):
        '''build rma:: options clause.

        Parameters
        ----------
        only : list of strings, optional
        except : list of strings, optional
        tabular : list of string, optional
        count : boolean, optional
        debug : string, optional
            'true', 'false' or 'preview'
        num_rows : int or string, optional
        start_row : int or string, optional
        '''
        clause = ''
        options_params = []

        only = kwargs.get(RmaApi.ONLY, None)

        if only is not None:
            options_params.append(
                self.only_except_tabular_clause(RmaApi.ONLY,
                                                only))

        # handle alternate 'except' spelling to avoid reserved word conflict
        excpt = kwargs.get(RmaApi.EXCEPT, None)
        excpt2 = kwargs.get(RmaApi.EXCPT, None)
        
        if excpt is not None and excpt2 is not None:
            warnings.warn('excpt and except options should not be used together',
                          Warning)
        elif excpt2 is not None:
            excpt = excpt2 

        if excpt is not None:
            options_params.append(
                self.only_except_tabular_clause(RmaApi.EXCEPT,
                                                excpt))

        tabular = kwargs.get(RmaApi.TABULAR, None)

        if tabular is not None:
            options_params.append(
                self.only_except_tabular_clause(RmaApi.TABULAR,
                                                tabular))

        num_rows = kwargs.get(RmaApi.NUM_ROWS, None)

        if num_rows is not None:
            if num_rows == RmaApi.ALL:
                options_params.append("[%s$eq'all']" % (RmaApi.NUM_ROWS))
            else:
                options_params.append('[%s$eq%d]' % (RmaApi.NUM_ROWS,
                                                     num_rows))

        start_row = kwargs.get(RmaApi.START_ROW, None)

        if start_row is not None:
            options_params.append('[%s$eq%d]' % (RmaApi.START_ROW,
                                                 start_row))

        order = kwargs.get(RmaApi.ORDER, None)

        if order is not None:
            options_params.append(self.order_clause(order))

        debug = kwargs.get(RmaApi.DEBUG, None)

        if debug is not None:
            options_params.append(self.debug_clause(debug))

        cnt = kwargs.get(RmaApi.COUNT, None)

        if cnt is not None:
            if cnt is True or cnt == 'true':
                options_params.append('[%s$eq%s]' % (RmaApi.COUNT,
                                                     RmaApi.TRUE))
            elif cnt is False or cnt == 'false':
                options_params.append('[%s$eq%s]' % (RmaApi.COUNT,
                                                     RmaApi.FALSE))
            else:
                pass

        if len(options_params) > 0:
            clause = RmaApi.OPTIONS + ''.join(options_params)

        return clause

    def only_except_tabular_clause(self, filter_type, attribute_list):
        '''Construct a clause to filter which attributes are returned
        for use in an rma::options clause.

        Parameters
        ----------
        filter_type : string
            'only', 'except', or 'tabular'
        attribute_list : list of strings
            for example ['acronym', 'products.name', 'structure.id']

        Returns
        -------
        clause : string
            The query clause for inclusion in an RMA query URL.

        Notes
        -----
        The title of tabular columns can be set by adding '+as+<title>'
        to the attribute.
        The tabular filter type requests a response that is row-oriented
        rather than a nested structure.
        Because of this, the tabular option can mask the lazy query behavior
        of an rma::include clause.
        The tabular option does not mask the inner-join behavior of an rma::include
        clause.
        The tabular filter is required for .csv format RMA requests.
        '''
        clause = ''

        if attribute_list is not None:
            clause = '[%s$eq%s]' % (filter_type,
                                    ','.join(attribute_list))

        return clause

    def order_clause(self, order_list=None):
        '''Construct a debug clause for use in an rma::options clause.

        Parameters
        ----------
        order_list : list of strings
            for example ['acronym', 'products.name+asc', 'structure.id+desc']

        Returns
        -------
        clause : string
            The query clause for inclusion in an RMA query URL.

        Notes
        -----
        Optionally adding '+asc' (default) or '+desc' after an attribute
        will change the sort order.
        '''
        clause = ''

        if order_list is not None:
            clause = '[order$eq%s]' % (','.join(order_list))

        return clause

    def debug_clause(self, debug_value=None):
        '''Construct a debug clause for use in an rma::options clause.
        Parameters
        ----------
        debug_value : string or boolean
            True, False, None (default) or 'preview'

        Returns
        -------
        clause : string
            The query clause for inclusion in an RMA query URL.

        Notes
        -----
        True will request debugging information in the response.
        False will request no debugging information.
        None will return an empty clause.
        'preview' will request debugging information without the query being run.

        '''
        clause = ''

        if debug_value is None:
            clause = ''
        if debug_value is True or debug_value == 'true':
            clause = '[debug$eqtrue]'
        elif debug_value is False or debug_value == 'false':
            clause = '[debug$eqfalse]'
        elif debug_value == 'preview':
            clause = "[debug$eq'preview']"

        return clause

    # TODO: deprecate for something that can preserve order
    def filters(self, filters):
        '''serialize RMA query filter clauses.

        Parameters
        ----------
        filters : dict
            keys and values for narrowing a query.

        Returns
        -------
        string
            filter clause for an RMA query string.
        '''
        filters_builder = []

        for (key, value) in filters.items():
            filters_builder.append(self.filter(key, value))

        return ''.join(filters_builder)

    # TODO: this needs to be more rigorous.
    def tuple_filters(self, filters):
        '''Construct an RMA filter clause.

        Notes
        -----

        See `RMA Path Syntax - Square Brackets for Filters <http://help.brain-map.org/display/api/RMA+Path+Syntax#RMAPathSyntax-SquareBracketsforFilters>`_ for additional documentation.
        '''
        filters_builder = []

        for filt in sorted(filters):
            if filt[-1] is None:
                continue
            if len(filt) == 2:
                val = filt[1]
                if type(val) is list:
                    val_array = []
                    for v in val:
                        if type(v) is str:
                            val_array.append(v)
                        else:
                            val_array.append(str(v))
                    val = ','.join(val_array)
                    filters_builder.append("[%s$eq%s]" % (filt[0], val))
                elif type(val) is int:
                    filters_builder.append("[%s$eq%d]" % (filt[0], val))
                elif type(val) is bool:
                    if val:
                        filters_builder.append("[%s$eqtrue]" % (filt[0]))
                    else:
                        filters_builder.append("[%s$eqfalse]" % (filt[0]))
                elif type(val) is str:
                    filters_builder.append("[%s$eq%s]" % (filt[0], filt[1]))
            elif len(filt) == 3:
                filters_builder.append("[%s%s%s]" % (filt[0],
                                                     filt[1],
                                                     str(filt[2])))

        return ''.join(filters_builder)

    def quote_string(self, the_string):
        '''Wrap a clause in single quotes.

        Parameters
        ----------
        the_string : string
            a clause to be included in an rma query that needs to be quoted

        Returns
        -------
        string
            input wrapped in single quotes
        '''
        return ''.join(["'", the_string, "'"])

    def filter(self, key, value):
        '''serialize a single RMA query filter clause.

        Parameters
        ----------
        key : string
            keys for narrowing a query.
        value : string
            value for narrowing a query.

        Returns
        -------
        string
            a single filter clause for an RMA query string.
        '''
        return "".join(['[',
                        key,
                        RmaApi.EQ,
                        str(value),
                        ']'])

    def build_schema_query(self, clazz=None, fmt='json'):
        '''Build the URL that will fetch the data schema.

        Parameters
        ----------
        clazz : string, optional
            Name of a specific class or None (default).
        fmt : string, optional
            json (default) or xml

        Returns
        -------
        url : string
            The constructed URL

        Notes
        -----
        If a class is specified, only the schema information for that class
        will be requested, otherwise the url requests the entire schema.
        '''
        if clazz is not None:
            class_clause = '/' + clazz
        else:
            class_clause = ''

        url = ''.join([self.rma_endpoint,
                       class_clause,
                       '.',
                       fmt])

        return url

    def get_schema(self, clazz=None):
        '''Retrieve schema information.'''
        schema_data = self.do_query(self.build_schema_query,
                                    self.read_data,
                                    clazz)

        return schema_data



class RmaTemplate(RmaApi):
    '''
    See: `Atlas Drawings and Ontologies
    <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies>`_
    '''

    def __init__(self, base_uri=None, query_manifest=None):
        super(RmaTemplate, self).__init__(base_uri)
        self.templates = query_manifest

    def to_filter_rhs(self, rhs):
        if type(rhs) == list:
            return ','.join(str(r) for r in rhs)

        return rhs

    def template_query(self, template_name, entry_name, **kwargs):
        cb = self.templates[template_name]
        templates = [e for e in cb if e['name'] == entry_name]

        if len(templates) > 0:
            template = templates[0]
        else:
            raise Exception('Entry %s not found.' % (entry_name))

        query_args = {'model': template['model']}

        if 'criteria' in template:
            criteria_template = Template(template['criteria'])

            if 'criteria_params' in template:
                criteria_params = {key: self.to_filter_rhs(kwargs.get(key))
                                   for key in template['criteria_params']
                                   if key in kwargs and kwargs.get(key) is not None}
            else:
                criteria_params = {}

            criteria_str = str(criteria_template.render(**criteria_params))
            if criteria_str:
                query_args['criteria'] = criteria_str

        if 'include' in template:
            include_template = Template(template['include'])

            if 'include_params' in template:
                include_params = {key: self.to_filter_rhs(kwargs.get(key))
                                  for key in template['include_params']
                                  if key in kwargs and kwargs.get(key) is not None}
            else:
                include_params = {}

            include_str = str(include_template.render(**include_params))
            if include_str:
                query_args['include'] = include_str

        if 'only' in kwargs:
            if kwargs.get('only') is not None:
                query_args['only'] = [self.quote_string(
                    ','.join(kwargs.get('only')))]
        elif 'only' in template:
            query_args['only'] = [
                self.quote_string(','.join(template['only']))]

        if 'except' in kwargs:
            if kwargs.get('except') is not None:
                query_args['except'] = [self.quote_string(
                    ','.join(kwargs.get('except')))]
        elif 'except' in template:
            query_args['except'] = template['except']

        if 'start_row' in kwargs:
            query_args['start_row'] = kwargs.get('start_row')
        elif 'start_row' in template:
            query_args['start_row'] = template['start_row']

        if 'num_rows' in kwargs:
            query_args['num_rows'] = kwargs.get('num_rows')
        elif 'num_rows' in template:
            query_args['num_rows'] = template['num_rows']

        if 'count' in kwargs:
            query_args['count'] = kwargs.get('count')
        elif 'count' in template:
            query_args['count'] = template['count']

        if 'order' in kwargs:
            query_args['order'] = kwargs.get('order')
        elif 'order' in template:
            query_args['order'] = template['order']

        query_args.update(kwargs)

        data = self.model_query(**query_args)

        return data
