import h5py
import numpy as np
from mpi4py import MPI

from bmtk.utils.reports.compartment.compartment_report import CompartmentReportOLD
#from bmtk.utils.reports.compartment.compartment_reader import CompartmentReader

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nhosts = comm.Get_size()


def merge():
    #report = CompartmentReader('__bmtk_tmp_cellvars_0_tmp_report.h5')
    #for pop in report.populations:
    #    print(report[pop])

    h5final = h5py.File('merged_file.h5', 'w')
    # tmp_h5_handles = [h5py.File(name, 'r') for name in self._tmp_files]
    tmp_reports = [CompartmentReader(name) for name in ['__bmtk_tmp_cellvars_0_tmp_report.h5', '__bmtk_tmp_cellvars_1_tmp_report.h5']]
    populations = set()
    for r in tmp_reports:
        populations.update(r.populations)
        # populations = set(r.populations for r in tmp_reports)
    #print(populations)
    #exit()

    for pop in populations:
        # Find the gid and segment offsets for each temp h5 file
        gid_ranges = []  # list of (gid-beg, gid-end)
        gid_offset = 0
        total_gid_count = 0  # total number of gids across all ranks

        seg_ranges = []
        seg_offset = 0
        total_seg_count = 0  # total number of segments across all ranks
        times = None

        n_steps = 0

        for rpt in tmp_reports:
            if pop not in rpt.populations:
                continue
            report = rpt[pop]

            seg_count = len(report.element_pos) #['/mapping/element_pos'])
            seg_ranges.append((seg_offset, seg_offset + seg_count))
            seg_offset += seg_count
            total_seg_count += seg_count

            gid_count = len(report.node_ids) #h5_tmp['mapping/node_ids'])
            gid_ranges.append((gid_offset, gid_offset + gid_count))
            gid_offset += gid_count
            total_gid_count += gid_count

            times = report.time  #h5_tmp['mapping/time']

            n_steps = report.n_steps

        mapping_grp = h5final.create_group('/report/{}/mapping'.format(pop))
        if times is not None and len(times) > 0:
            mapping_grp.create_dataset('time', data=times)
        element_id_ds = mapping_grp.create_dataset('element_id', shape=(total_seg_count,), dtype=np.uint)
        el_pos_ds = mapping_grp.create_dataset('element_pos', shape=(total_seg_count,), dtype=np.float)
        gids_ds = mapping_grp.create_dataset('node_ids', shape=(total_gid_count,), dtype=np.uint)
        index_pointer_ds = mapping_grp.create_dataset('index_pointer', shape=(total_gid_count + 1,), dtype=np.uint)
        for rpt in tmp_reports:
            if pop not in rpt.populations:
                continue
            report = rpt[pop]
            for k, v in report.custom_columns.items():
                print(k, v)
                mapping_grp.create_dataset(k, shape=(total_seg_count,), dtype=type(v[0]))

        # combine the /mapping datasets
        for i, rpt in enumerate(tmp_reports):
            if pop not in rpt.populations:
                continue

            report = rpt[pop]

            #tmp_mapping_grp = h5_tmp['mapping']
            beg, end = seg_ranges[i]
            element_id_ds[beg:end] = report.element_ids #tmp_mapping_grp['element_id']
            el_pos_ds[beg:end] = report.element_pos #tmp_mapping_grp['element_pos']
            for k, v in report.custom_columns.items():
                mapping_grp[k][beg:end] = v

            # shift the index pointer values
            index_pointer = np.array(report.index_pointer) #tmp_mapping_grp['index_pointer'])
            update_index = beg + index_pointer

            beg, end = gid_ranges[i]
            gids_ds[beg:end] = report.node_ids  #tmp_mapping_grp['node_ids']
            index_pointer_ds[beg:(end + 1)] = update_index


        # combine the /var/data datasets
        data_name = '/report/{}/data'.format(pop)
        # data_name = '/{}/data'.format(var_name)
        var_data = h5final.create_dataset(data_name, shape=(n_steps, total_seg_count), dtype=np.float)
        #var_data.attrs['variable_name'] = var_name
        for i, rpt in enumerate(tmp_reports):
            if pop not in rpt.populations:
                continue
            report = rpt[pop]

            beg, end = seg_ranges[i]
            var_data[:, beg:end] = report.data


#merge()


class Cell(object):
    def __init__(self, node_id, n_segs):
        self.node_id = node_id
        self.sections = range(n_segs)
        self.segments = [0.5]*n_segs
        self.vals = [(node_id+1)*0.10]*n_segs

def run_report():
    report = CompartmentReportOLD(file_name='tmp_report.h5', units='mV', variable='Vm', tmp_dir='.', default_population='cortex')
    report.t_stop = 100.0
    report.dt = 0.1

    cells = [Cell(0, 10), Cell(1, 50), Cell(2, 100), Cell(3, 200)]
    rank_cells = [c for c in cells[rank::nhosts]]

    for c in rank_cells:
        report.add_cell(c.node_id, c.sections, c.segments)

    report.initialize(n_steps=1000, buffer_size=100)
    for i in range(1000):
        if i % 100 == 0 and i > 0:
            report.flush()

        for c in rank_cells:
            report.record_cell(c.node_id, c.vals, tstep=i)

    report.flush()
    report.close()
    comm.Barrier()
    report.merge()


run_report()
