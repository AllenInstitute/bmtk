from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y

from bmtk.simulator.filternet.filters import TemporalFilterCosineBump, GaussianSpatialFilter, SpatioTemporalFilter
from bmtk.simulator.filternet.cell_models import TwoSubfieldLinearCell, OnUnit, OffUnit
from bmtk.simulator.filternet.transfer_functions import ScalarTransferFunction, MultiTransferFunction
from bmtk.simulator.filternet.utils import get_data_metrics_for_each_subclass, get_tcross_from_temporal_kernel


def create_two_sub_cell(dom_lf, non_dom_lf, dom_spont, non_dom_spont, onoff_axis_angle, subfield_separation, dom_location):
    dsp = str(dom_spont)
    ndsp = str(non_dom_spont)
    two_sub_transfer_fn = MultiTransferFunction((symbolic_x, symbolic_y),
                                                'Heaviside(x+'+dsp+')*(x+'+dsp+')+Heaviside(y+'+ndsp+')*(y+'+ndsp+')')

    two_sub_cell = TwoSubfieldLinearCell(dom_lf, non_dom_lf, subfield_separation=subfield_separation,
                                         onoff_axis_angle=onoff_axis_angle, dominant_subfield_location=dom_location,
                                         transfer_function=two_sub_transfer_fn)
    return two_sub_cell


def create_temporal_filter(inp_dict):
    opt_wts = inp_dict['opt_wts']
    opt_kpeaks = inp_dict['opt_kpeaks']
    opt_delays = inp_dict['opt_delays']
    temporal_filter = TemporalFilterCosineBump(opt_wts, opt_kpeaks, opt_delays)

    return temporal_filter


def createOneUnitOfTwoSubunitFilter(prs, ttp_exp):
    filt = create_temporal_filter(prs)
    tcross_ind = get_tcross_from_temporal_kernel(filt.get_kernel(threshold=-1.0).kernel)
    filt_sum = filt.get_kernel(threshold=-1.0).kernel[:tcross_ind].sum()

    # Calculate delay offset needed to match response latency with data and rebuild temporal filter
    del_offset = ttp_exp - tcross_ind
    if del_offset >= 0:
        delays = prs['opt_delays']
        delays[0] = delays[0] + del_offset
        delays[1] = delays[1] + del_offset
        prs['opt_delays'] = delays
        filt_new = create_temporal_filter(prs)
    else:
        print 'del_offset < 0'

    return filt_new, filt_sum


def load_cell(node):
    origin = (0.0, 0.0)
    translate = (node['x'], node['y'])
    sigma = node['spatial_size'] / 3.0  # convert from degree to SD
    sigma = (sigma, sigma)
    spatial_filter = GaussianSpatialFilter(translate=translate, sigma=sigma, origin=origin)

    print node.node_id
    if node['pop_name'] == 'sONsOFF_001':

        # sON temporal filter
        sON_prs = {'opt_wts': [node['weight_non_dom_0'], node['weight_non_dom_1']],
                   'opt_kpeaks': [node['kpeaks_non_dom_0'], node['kpeaks_non_dom_1']],
                   'opt_delays': [node['delay_non_dom_0'], node['delay_non_dom_1']]}
        sON_filt_new = createOneUnitOfTwoSubunitFilter(sON_prs, 121.0)
        sON_sum = sON_filt_new[1]
        sON_filt_new = sON_filt_new[0]

        # tOFF temporal filter
        sOFF_prs = {'opt_wts': [node['weight_dom_0'], node['weight_dom_1']],
                    'opt_kpeaks': [node['kpeaks_dom_0'], node['kpeaks_dom_1']],
                    'opt_delays': [node['delay_dom_0'], node['delay_dom_1']]}
        sOFF_filt_new = createOneUnitOfTwoSubunitFilter(sOFF_prs, 115.0)
        sOFF_sum = sOFF_filt_new[1]
        sOFF_filt_new = sOFF_filt_new[0]

        amp_on = 1.0  # set the non-dominant subunit amplitude to unity
        spont = 4.0
        max_roff = 35.0
        max_ron = 21.0
        amp_off = -(max_roff / max_ron) * (sON_sum / sOFF_sum) * amp_on - (spont * (max_roff - max_ron)) / (
            max_ron * sOFF_sum)

        # Create sON subunit:
        xfer_fn_son = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
        linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)
        scell_on = OnUnit(linear_filter_son, xfer_fn_son)

        # Create sOFF subunit:
        xfer_fn_soff = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
        linear_filter_soff = SpatioTemporalFilter(spatial_filter, sOFF_filt_new, amplitude=amp_off)
        scell_off = OffUnit(linear_filter_soff, xfer_fn_soff)

        sep_ss_onoff_cell = create_two_sub_cell(linear_filter_soff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                node['tuning_angle'], node['sf_sep'], translate)
        cell = sep_ss_onoff_cell

    elif node['pop_name'] == 'sONtOFF_001':
        # spatial_filter.get_kernel(np.arange(120), np.arange(240)).imshow()
        # sON temporal filter
        sON_prs = {'opt_wts': [node['weight_non_dom_0'], node['weight_non_dom_1']],
                   'opt_kpeaks': [node['kpeaks_non_dom_0'], node['kpeaks_non_dom_1']],
                   'opt_delays': [node['delay_non_dom_0'], node['delay_non_dom_1']]}
        sON_filt_new = createOneUnitOfTwoSubunitFilter(sON_prs, 93.5)
        sON_sum = sON_filt_new[1]
        sON_filt_new = sON_filt_new[0]

        # tOFF temporal filter
        tOFF_prs = {'opt_wts': [node['weight_dom_0'], node['weight_dom_1']],
                    'opt_kpeaks': [node['kpeaks_dom_0'], node['kpeaks_dom_1']],
                    'opt_delays': [node['delay_dom_0'], node['delay_dom_1']]}
        tOFF_filt_new = createOneUnitOfTwoSubunitFilter(tOFF_prs, 64.8)  # 64.8
        tOFF_sum = tOFF_filt_new[1]
        tOFF_filt_new = tOFF_filt_new[0]

        amp_on = 1.0  # set the non-dominant subunit amplitude to unity
        spont = 5.5
        max_roff = 46.0
        max_ron = 31.0
        amp_off = -0.7 * (max_roff / max_ron) * (sON_sum / tOFF_sum) * amp_on - (spont * (max_roff - max_ron)) / (
            max_ron * tOFF_sum)

        # Create sON subunit:
        xfer_fn_son = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
        linear_filter_son = SpatioTemporalFilter(spatial_filter, sON_filt_new, amplitude=amp_on)
        scell_on = OnUnit(linear_filter_son, xfer_fn_son)
        # linear_filter_son.spatial_filter.get_kernel(np.arange(120), np.arange(240)).imshow()

        # Create tOFF subunit:
        xfer_fn_toff = ScalarTransferFunction('Heaviside(s+' + str(0.5 * spont) + ')*(s+' + str(0.5 * spont) + ')')
        linear_filter_toff = SpatioTemporalFilter(spatial_filter, tOFF_filt_new, amplitude=amp_off)
        tcell_off = OffUnit(linear_filter_toff, xfer_fn_toff)
        # linear_filter_toff.spatial_filter.get_kernel(np.arange(120), np.arange(240)).kernel

        sep_ts_onoff_cell = create_two_sub_cell(linear_filter_toff, linear_filter_son, 0.5 * spont, 0.5 * spont,
                                                node['tuning_angle'], node['sf_sep'], translate)

        cell = sep_ts_onoff_cell
    else:
        type_split = node['pop_name'].split('_')
        cell_type, tf_str = type_split[0], type_split[1]

        # For temporal filter
        wts = [node['weight_dom_0'], node['weight_dom_1']]
        kpeaks = [node['kpeaks_dom_0'], node['kpeaks_dom_1']]
        delays = [node['delay_dom_0'], node['delay_dom_1']]

        ################# End of extract cell parameters needed   #################

        # Get spont from experimental data
        exp_prs_dict = get_data_metrics_for_each_subclass(cell_type)
        subclass_prs_dict = exp_prs_dict[tf_str]
        spont_exp = subclass_prs_dict['spont_exp']
        spont_str = str(spont_exp[0])

        # Get filters
        transfer_function = ScalarTransferFunction('Heaviside(s+' + spont_str + ')*(s+' + spont_str + ')')
        temporal_filter = TemporalFilterCosineBump(wts, kpeaks, delays)

        if cell_type.find('ON') >= 0:
            amplitude = 1.0
            linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
            cell = OnUnit(linear_filter, transfer_function)
        elif cell_type.find('OFF') >= 0:
            amplitude = -1.0
            linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
            cell = OffUnit(linear_filter, transfer_function)

    return cell
