import numpy as np
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y

from bmtk.simulator.filternet.filters import TemporalFilterCosineBump, GaussianSpatialFilter, SpatioTemporalFilter
from bmtk.simulator.filternet.cell_models import TwoSubfieldLinearCell, OnUnit, OffUnit, LGNOnOffCell
from bmtk.simulator.filternet.transfer_functions import ScalarTransferFunction, MultiTransferFunction
from bmtk.simulator.filternet.utils import get_data_metrics_for_each_subclass, get_tcross_from_temporal_kernel
from bmtk.simulator.filternet.pyfunction_cache import py_modules


def create_two_sub_cell(dom_lf, non_dom_lf, dom_spont, non_dom_spont, onoff_axis_angle, subfield_separation,
                        dom_location):
    dsp = str(dom_spont)
    ndsp = str(non_dom_spont)
    two_sub_transfer_fn = MultiTransferFunction((symbolic_x, symbolic_y),
                                                'Heaviside(x+'+dsp+')*(x+'+dsp+')+Heaviside(y+'+ndsp+')*(y+'+ndsp+')')

    two_sub_cell = TwoSubfieldLinearCell(dom_lf, non_dom_lf, subfield_separation=subfield_separation,
                                         onoff_axis_angle=onoff_axis_angle, dominant_subfield_location=dom_location,
                                         transfer_function=two_sub_transfer_fn)
    return two_sub_cell


def createOneUnitOfTwoSubunitFilter(weights, kpeaks, delays, ttp_exp):
    delays = np.array(delays)
    filt = TemporalFilterCosineBump(weights, kpeaks, delays)
    tcross_ind = get_tcross_from_temporal_kernel(filt.get_kernel(threshold=-1.0).kernel)
    filt_sum = filt.get_kernel(threshold=-1.0).kernel[:tcross_ind].sum()

    # Calculate delay offset needed to match response latency with data and rebuild temporal filter
    del_offset = ttp_exp - tcross_ind
    if del_offset >= 0:
        delays_updated = delays + del_offset
        filt_new = TemporalFilterCosineBump(weights, kpeaks, delays_updated)
    else:
        raise Exception('del_offset < 0')

    return filt_new, filt_sum


def get_tf_params(node, dynamics_params, non_dom_props=False):
    if not non_dom_props:
        weights = node.weights if node.weights is not None else dynamics_params['opt_wts']
        kpeaks = node.kpeaks if node.kpeaks is not None else dynamics_params['opt_kpeaks']
        delays = node.delays if node.delays is not None else dynamics_params['opt_delays']
    else:
        dp = dynamics_params or {}
        weights = node.weights_non_dom if node.weights_non_dom is not None else dp.get('opt_wts', None)
        kpeaks = node.kpeaks_non_dom if node.kpeaks_non_dom is not None else dp.get('opt_kpeaks', None)
        delays = node.delays_non_dom if node.delays_non_dom is not None else dp.get('opt_delays', None)

    if node.predefined_jitter:
        jitter_fnc = np.vectorize(lambda a: np.random.uniform(a * node.jitter[0], a * node.jitter[0]))
        weights = jitter_fnc(weights) if weights is not None else weights
        kpeaks = jitter_fnc(kpeaks) if kpeaks is not None else kpeaks
        delays = jitter_fnc(delays) if delays is not None else delays

    return weights, kpeaks, delays


def default_cell_loader(node, template_name, dynamics_params):
    """

    :param node:
    :param template_name:
    :param dynamics_params:
    :return:
    """
    # Create the spatial filter
    origin = (0.0, 0.0)
    translate = (node['x'], node['y'])
    sigma = node['spatial_size'] / 3.0  # convert from degree to SD
    if np.isscalar(sigma):
        sigma = (sigma, sigma)
    if 'spatial_rotation' in node:
        rotation = node['spatial_rotation']
    else:
        rotation = 0.0

    spatial_filter = GaussianSpatialFilter(translate=translate, sigma=sigma, origin=origin, rotation=rotation)

    t_weights, t_kpeaks, t_delays = get_tf_params(node, dynamics_params)

    if template_name:
        model_name = template_name[1]
    else:
        model_name = node['pop_name']

    if model_name in ['sONsOFF_001', 'sONsOFF']:
        # sON temporal filter
        t_weights_nd, t_kpeaks_nd, t_delays_nd = get_tf_params(node, node.non_dom_params, non_dom_props=True)
        sON_filt_new, sON_sum = createOneUnitOfTwoSubunitFilter(t_weights_nd, t_kpeaks_nd, t_delays_nd, 121.0)

        sOFF_filt_new, sOFF_sum = createOneUnitOfTwoSubunitFilter(t_weights, t_kpeaks, t_delays, 115.0)

        amp_on = 1.0  # set the non-dominant subunit amplitude to unity
        spont = 4.0
        max_roff = 35.0
        max_ron = 21.0
        amp_off = -(max_roff/max_ron)*(sON_sum/sOFF_sum)*amp_on - (spont*(max_roff - max_ron))/(max_ron*sOFF_sum)

        # Create sON subunit:
        linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)

        # Create sOFF subunit:
        linear_filter_soff = SpatioTemporalFilter(spatial_filter, sOFF_filt_new, amplitude=amp_off)

        sf_sep = node.sf_sep
        if node.predefined_jitter:
            sf_sep = np.random.uniform(node.lower_jitter*sf_sep, node.upper_jitter*sf_sep)

        sep_ss_onoff_cell = create_two_sub_cell(linear_filter_soff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                node.tuning_angle, sf_sep, translate)
        cell = sep_ss_onoff_cell

    elif model_name in ['sONtOFF_001', 'sONtOFF']:
        t_weights_nd, t_kpeaks_nd, t_delays_nd = get_tf_params(node, node.non_dom_params, non_dom_props=True)
        sON_filt_new, sON_sum = createOneUnitOfTwoSubunitFilter(t_weights_nd, t_kpeaks_nd, t_delays_nd, 93.5)

        tOFF_filt_new, tOFF_sum = createOneUnitOfTwoSubunitFilter(t_weights, t_kpeaks, t_delays, 64.8)  # 64.8

        amp_on = 1.0  # set the non-dominant subunit amplitude to unity
        spont = 5.5
        max_roff = 46.0
        max_ron = 31.0
        amp_off = -0.7*(max_roff/max_ron)*(sON_sum/tOFF_sum)*amp_on - (spont*(max_roff - max_ron))/(max_ron*tOFF_sum)

        # Create sON subunit:
        linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)

        # Create tOFF subunit:
        linear_filter_toff = SpatioTemporalFilter(spatial_filter, tOFF_filt_new, amplitude=amp_off)

        sf_sep = node.sf_sep
        if node.predefined_jitter:
            sf_sep = np.random.uniform(node.lower_jitter*sf_sep, node.upper_jitter*sf_sep)

        sep_ts_onoff_cell = create_two_sub_cell(linear_filter_toff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                node.tuning_angle, sf_sep, translate)

        cell = sep_ts_onoff_cell

    elif model_name == 'LGNOnOFFCell':
        wts = [node['weight_dom_0'], node['weight_dom_1']]
        kpeaks = [node['kpeaks_dom_0'], node['kpeaks_dom_1']]
        delays = [node['delay_dom_0'], node['delay_dom_1']]
        # transfer_function = ScalarTransferFunction('s')
        temporal_filter = TemporalFilterCosineBump(wts, kpeaks, delays)

        spatial_filter_on = GaussianSpatialFilter(sigma=node['sigma_on'], origin=origin, translate=translate)
        on_linear_filter = SpatioTemporalFilter(spatial_filter_on, temporal_filter, amplitude=20)

        spatial_filter_off = GaussianSpatialFilter(sigma=node['sigma_off'], origin=origin, translate=translate)
        off_linear_filter = SpatioTemporalFilter(spatial_filter_off, temporal_filter, amplitude=-20)
        cell = LGNOnOffCell(on_linear_filter, off_linear_filter)

    else:
        type_split = model_name.split('_')

        if len(type_split) == 1:
            cell_type = model_name
            tf_str = 'TF8'
        else:
            cell_type, tf_str = type_split[0], type_split[1]

        # Get spontaneous firing rate, either from the cell property of calculate from experimental data
        if 'spont_fr' in node:
            spont_fr = node['spont_fr']
        else:
            exp_prs_dict = get_data_metrics_for_each_subclass(cell_type)
            subclass_prs_dict = exp_prs_dict[tf_str]
            spont_fr = subclass_prs_dict['spont_exp'][0]

        # Get filters
        transfer_function = ScalarTransferFunction('Heaviside(s+{})*(s+{})'.format(spont_fr, spont_fr))
        temporal_filter = TemporalFilterCosineBump(t_weights, t_kpeaks, t_delays)

        if cell_type.find('ON') >= 0:
            amplitude = 1.0
            linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
            cell = OnUnit(linear_filter, transfer_function)
        elif cell_type.find('OFF') >= 0:
            amplitude = -1.0
            linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
            cell = OffUnit(linear_filter, transfer_function)

    return cell


py_modules.add_cell_processor('default', default_cell_loader, overwrite=False)
py_modules.add_cell_processor('preset_params', default_cell_loader, overwrite=False)
