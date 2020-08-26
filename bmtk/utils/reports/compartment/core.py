# class CompartmentReport(object):
#     def __init__(self, path, mode='r', **kwargs):
#         pass
#
#     def initialize(self):
#         pass
#
#     def add_cell(self, node_id, sections, segments, population=None, **attrs):
#         pass
#
#     def record_cell(self, node_id, segment_vals, tstep, population=None):
#         pass
#
#     def record_cell_block(self, node_ids, segment_vals, tbegin, tend, population=None):
#         pass
#
#     def close(self):
#         pass
#
#     def flush(self):
#         pass
#
#     def from_sonata(self, path):
#         pass
#
#     def from_nwb(self, path):
#         pass
#
#
#     def from_path(self, path):
#         pass
#
#     def __getitem__(self, item):
#         pass


class CompartmentReaderABC(object):
    @property
    def populations(self):
        raise NotImplementedError()

    def get_population(self, population):
        raise NotImplementedError()

    def units(self, population=None):
        raise NotImplementedError()

    def variable(self, population=None):
        raise NotImplementedError()

    def tstart(self, population=None):
        raise NotImplementedError()

    def tstop(self, population=None):
        raise NotImplementedError()

    def dt(self, population=None):
        raise NotImplementedError()

    def time_trace(self, population=None):
        raise NotImplementedError()

    def n_steps(self, population=None):
        raise NotImplementedError()

    def node_ids(self, population=None):
        raise NotImplementedError()

    def element_pos(self, node_id=None, population=None):
        raise NotImplementedError()

    def element_ids(self, node_id=None, population=None):
        raise NotImplementedError()

    def n_elements(self, node_id, population=None):
        raise NotImplementedError()

    def index(self, population=None):
        raise NotImplementedError()

    def data(self, node_id=None, population=None, time_window=None, sections='all', **opt_attrs):
        raise NotImplementedError()

    def custom_columns(self, population=None):
        raise NotImplementedError()

    def get_column(self, column_name, population=None):
        raise NotImplementedError()

    def get_node_description(self, node_id, population=None):
        raise NotImplementedError()

    def get_report_description(self, population=None):
        raise NotImplementedError()

    def validate_file(self, path, **attrs):
        return False

    def __getitem__(self, population):
        return self.get_population(population)


class CompartmentWriterABC(object):
    def __init__(self, path, mode='r', **kwargs):
        raise NotImplementedError()

    def set_units(self, val, population=None):
        raise NotImplementedError()

    def set_variable(self, val, population=None):
        raise NotImplementedError()

    def set_tstart(self, val, population=None):
        raise NotImplementedError()

    def set_tstop(self, val, population=None):
        raise NotImplementedError()

    def set_dt(self, val, population=None):
        raise NotImplementedError()

    def set_time_trace(self, val, population=None):
        raise NotImplementedError()

    def initialize(self, **kwargs):
        raise NotImplementedError()

    def add_cell(self, node_id, element_ids, element_pos, population=None, **attrs):
        raise NotImplementedError()

    def record_cell(self, node_id, values, tstep, population=None):
        raise NotImplementedError()

    def record_cell_block(self, node_id, values, tbegin, tend, population=None):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError()

    def merge(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()
