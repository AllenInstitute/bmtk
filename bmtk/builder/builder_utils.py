import numpy as np
import hashlib
import logging


try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    barrier = comm.barrier

except ImportError:
    mpi_rank = 0
    mpi_size = 1
    barrier = lambda: None


MPI_fail_params_nonuniform = True  # throw exception unless params across MPI ranks are the same.
logger = logging.getLogger(__name__)


def add_hdf5_attrs(hdf5_handle):
    # TODO: move this as a utility function
    hdf5_handle['/'].attrs['magic'] = np.uint32(0x0A7A)
    hdf5_handle['/'].attrs['version'] = [np.uint32(0), np.uint32(1)]


def list_to_hash(str_list):
    str_list = str_list.copy()
    str_list.sort()
    combined_keys = ':'.join(str_list).encode('utf-8')
    return hashlib.md5(combined_keys).hexdigest()


def check_properties_across_ranks(properties, graph_type='node'):
    """Checks that a properties table is consistant across all MPI ranks. Mainly used by add_nodes() and add_edges()
    method due to bug where using random generator without rng_seed was causing issues building the network properties
    consistantly using multiple cores.

    Will throw an Exception or a warning message (if MPI_fail_params_nonuniform is false)

    :param properties: A dictionary
    :param graph_type: 'node' or 'edge', used in error message. default 'node'.
    """
    if mpi_size < 2:
        return

    # Check that model_properties have the same number of items and the keys match
    n_args = len(properties)
    ranked_args = comm.allgather(n_args)
    if len(set(ranked_args)) > 1:
        err_msg = '{} properties are not the same across all ranks.'.format(graph_type)
        if not MPI_fail_params_nonuniform:
            logger.warning(err_msg)
        else:
            raise IndexError(err_msg)

    if n_args == 0:
        return

    # create a string/id that will be uniform across all ranks, even if the dict on one rank returns keys out-of-order.
    prop_keys = list(properties.keys())
    prop_keys.sort()
    combined_keys = ':'.join(prop_keys).encode('utf-8')
    hash_id = hashlib.md5(combined_keys).hexdigest()
    ranked_keys = comm.allgather(hash_id)
    if len(set(ranked_keys)) > 1:
        err_msg = '{} properties are not the same across all ranks.'.format(graph_type)
        if not MPI_fail_params_nonuniform:
            logger.warning(err_msg)
        else:
            raise IndexError(err_msg)

    # For each item in model_properties dictionary try to check that values are the same
    for pkey in prop_keys:
        # Don't use Dict.items() method since it is possible the ret order is different across ranks.
        pval = properties[pkey]

        try:
            if isinstance(pval, bytes):
                phash = hashlib.md5(pval).hexdigest()

            elif isinstance(pval, str):
                phash = hashlib.md5(pval.encode('utf-8')).hexdigest()

            elif isinstance(pval, (int, float, bool)):
                phash = pval

            elif isinstance(pval, (list, tuple)):
                joined_keys = ':'.join([str(p) for p in pval]).encode('utf-8')
                phash = hashlib.md5(joined_keys).hexdigest()

            elif isinstance(pval, np.ndarray):
                phash = hashlib.md5(pval.data.tobytes()).hexdigest()

            else:
                continue

        except TypeError as te:
            # If the hashing fails assume there is no MPI data issue and continue with the next property.
            continue

        ranked_vals = comm.allgather(phash)
        if len(set(ranked_vals)) > 1:
            err_msg = '{} property "{}" varies across ranks, please make sure parameter value is uniform across all' \
                      'ranks or set bmtk.builder.MPI_fail_params_nonuniform to False'.format(graph_type, pkey)
            if not MPI_fail_params_nonuniform:
                logger.warning(err_msg)
            else:
                raise TypeError(err_msg)
