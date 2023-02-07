import numpy as np
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y
from six import string_types

from bmtk.simulator.filternet.filters import TemporalFilterCosineBump, GaussianSpatialFilter, SpatioTemporalFilter, \
    GaborFilter, SpectroTemporalFilter
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
        jitter_fnc = lambda a: np.array([np.random.uniform(x*node.jitter[0], x*node.jitter[1]) for x in a])
        weights = jitter_fnc(weights) if weights is not None else weights
        kpeaks = jitter_fnc(kpeaks) if kpeaks is not None else kpeaks
        delays = jitter_fnc(delays) if delays is not None else delays

    return weights, kpeaks, delays



def get_sigma(node, dynamics_params):
    if 'spatial_size' in node:
        sigma = node['spatial_size']
    elif 'sigma' in node:
        sigma = node['sigma']
    elif 'spatial_size' in dynamics_params:
        sigma = dynamics_params['spatial_size']
    elif 'sigma' in dynamics_params:
        sigma = dynamics_params['spatial_size']
    else:
        # TODO: Raise warning
        sigma = (1.0, 1.0)

    if np.isscalar(sigma):
        sigma = (sigma, sigma) # convert from degree to SD

    return sigma[0]/3.0, sigma[1]/3.0


def get_gb_params(node, dynamics_params):
    t_mod_freq = node.t_mod_freq if node.t_mod_freq is not None else dynamics_params['t_mod_freq']
    sp_mod_freq = node.sp_mod_freq if node.sp_mod_freq is not None else dynamics_params['sp_mod_freq']
    Lambda = 1/np.linalg.norm([t_mod_freq, sp_mod_freq])    # Wavelength of oscillatory component
    #sigma1_ratio = node.sigma1_ratio if node.sigma1_ratio is not None else dynamics_params['sigma1_ratio']
    #sigma1 = Lambda /sigma1_ratio     # Width of Gaussian in direction of oscillation
    #sigma2 = node.sigma2 if node.sigma2 is not None else dynamics_params['sigma2']
    sigma_f = node.sigma_f if node.sigma_f is not None else dynamics_params['sigma_f']
    sigma_t = node.sigma_t if node.sigma_t is not None else dynamics_params['sigma_t']
    if t_mod_freq != 0:
        theta = np.arctan(sp_mod_freq / t_mod_freq)
    else:
        theta = np.pi / 2
    if isinstance(dynamics_params['psi'], string_types):
        dynamics_params['psi'] = eval(dynamics_params['psi'].replace('pi', 'np.pi'))
    psi = node.psi if node.psi is not None else dynamics_params['psi']
    delay = node.delays if node.delays is not None else dynamics_params['delay']

    '''
    else:
        dp = dynamics_params or {}
        weights = node.weights_non_dom if node.weights_non_dom is not None else dp.get('opt_wts', None)
        kpeaks = node.kpeaks_non_dom if node.kpeaks_non_dom is not None else dp.get('opt_kpeaks', None)
        delays = node.delays_non_dom if node.delays_non_dom is not None else dp.get('opt_delays', None)
    
    if node.predefined_jitter:
        jitter_fnc = lambda a: np.array([np.random.uniform(x*node.jitter[0], x*node.jitter[1]) for x in a])
        weights = jitter_fnc(weights) if weights is not None else weights
        kpeaks = jitter_fnc(kpeaks) if kpeaks is not None else kpeaks
        delays = jitter_fnc(delays) if delays is not None else delays
    '''
    return Lambda, sigma_f, sigma_t, theta, psi, delay


def default_cell_loader(node, template_name, dynamics_params):
    """

    :param node:
    :param template_name:
    :param dynamics_params:
    :return:
    """
    if template_name[0] == 'lgnmodel':
        # Create the spatial filter
        origin = (0.0, 0.0)
        translate = (node['x'], node['y'])

        sigma = get_sigma(node, dynamics_params)
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
                sf_sep = np.random.uniform(node.jitter[0]*sf_sep, node.jitter[1]*sf_sep)

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
                sf_sep = np.random.uniform(node.jitter[0]*sf_sep, node.jitter[1]*sf_sep)

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

    elif template_name[0] == 'audmodel':
        # Currently tying y to center freq
        translate = (node['y'])

        Lambda, sigma_f, sigma_t, theta, psi, delay = get_gb_params(node, dynamics_params)

        spectrotemporal_filter = GaborFilter(translate, sigma_f, sigma_t, theta, Lambda, psi)

        if template_name:
            model_name = template_name[1]
        else:
            model_name = node['pop_name']

        if model_name == 'AUD_foo':
            # Create the spectro-temporal filter
            pass
            # transfer_function = ScalarTransferFunction('Heaviside(s+{})*(s+{})'.format(spont_fr, spont_fr))

        cell_type = model_name
        # Get spontaneous firing rate, either from the cell property of calculate from experimental data
        if 'spont_fr' in node:
            spont_fr = node['spont_fr']
        else:
            '''
            exp_prs_dict = get_data_metrics_for_each_subclass(cell_type)
            subclass_prs_dict = exp_prs_dict[tf_str]
            spont_fr = subclass_prs_dict['spont_exp'][0]
            '''
            spont_fr = 2
        transfer_function = ScalarTransferFunction('Heaviside(s+{})*(s+{})'.format(spont_fr, spont_fr))
        amplitude = 1.0
        linear_filter = SpectroTemporalFilter(spectrotemporal_filter, amplitude=amplitude)
        cell = OnUnit(linear_filter, transfer_function)

    else:
        pass

    return cell


py_modules.add_cell_processor('default', default_cell_loader, overwrite=False)
py_modules.add_cell_processor('preset_params', default_cell_loader, overwrite=False)
