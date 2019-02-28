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
import numpy as np


def lif_aibs_converter(config):
    """

    :param config:
    :return:
    """
    coeffs = config['coeffs']
    params = {'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'g': coeffs['G'] / config['R_input'] * 1.0e09,
              'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'C_m': coeffs['C'] * config['C'] * 1.0e12,
              't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
              'V_reset': config['El_reference'] * 1.0e03,
              'V_dynamics_method': 'linear_exact'}  # 'linear_forward_euler' or 'linear_exact'
    return params


def lif_asc_aibs_converter(config):
    """

    :param config:
    :return:
    """
    coeffs = config['coeffs']
    params = {'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'g': coeffs['G'] / config['R_input'] * 1.0e09,
              'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'C_m': coeffs['C'] * config['C'] * 1.0e12,
              't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
              'V_reset': config['El_reference'] * 1.0e03,
              'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
              'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
              'asc_amps': np.array(config['asc_amp_array']) * np.array(coeffs['asc_amp_array']) * 1.0e12,
              'V_dynamics_method': 'linear_exact'}
    return params


def lif_r_aibs_converter(config):
    """

    :param config:
    :return:
    """
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    params = {'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'g': coeffs['G'] / config['R_input'] * 1.0e09,
              'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'C_m': coeffs['C'] * config['C'] * 1.0e12,
              't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
              'a_spike': threshold_params['a_spike'] * 1.0e03,
              'b_spike': threshold_params['b_spike'] * 1.0e-03,
              'a_reset': reset_params['a'],
              'b_reset': reset_params['b'] * 1.0e03,
              'V_dynamics_method': 'linear_exact'}
    return params


def lif_r_asc_aibs_converter(config):
    """Creates a nest glif_lif_r_asc object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
            'g': coeffs['G'] / config['R_input'] * 1.0e09,
            'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
            'C_m': coeffs['C'] * config['C'] * 1.0e12,
            't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
            'a_spike': threshold_params['a_spike'] * 1.0e03,
            'b_spike': threshold_params['b_spike'] * 1.0e-03,
            'a_reset': reset_params['a'],
            'b_reset': reset_params['b'] * 1.0e03,
            'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
            'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
            'asc_amps': np.array(config['asc_amp_array']) * np.array(coeffs['asc_amp_array']) * 1.0e12,
            'V_dynamics_method': 'linear_exact'}
    return params


def lif_r_asc_a_aibs_converter(config):
    """Creates a nest glif_lif_r_asc_a object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    params = {'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'g': coeffs['G'] / config['R_input'] * 1.0e09,
              'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'C_m': coeffs['C'] * config['C'] * 1.0e12,
              't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
              'a_spike': threshold_params['a_spike'] * 1.0e03,
              'b_spike': threshold_params['b_spike'] * 1.0e-03,
              'a_voltage': threshold_params['a_voltage'] * coeffs['a'] * 1.0e-03,
              'b_voltage': threshold_params['b_voltage'] * coeffs['b'] * 1.0e-03,
              'a_reset': reset_params['a'],
              'b_reset': reset_params['b'] * 1.0e03,
              'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
              'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
              'asc_amps': np.array(config['asc_amp_array']) * np.array(coeffs['asc_amp_array']) * 1.0e12,
              'V_dynamics_method': 'linear_exact'}
    return params


# synaptic ports testing
def lif_psc_aibs_converter(config, syn_tau=[5.5, 8.5, 2.8, 5.8]):
    """Creates a nest glif_lif_psc object"""
    coeffs = config['coeffs']
    params = {'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'g': coeffs['G'] / config['R_input'] * 1.0e09,
              'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'C_m': coeffs['C'] * config['C'] * 1.0e12,
              't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
              'V_reset': config['El_reference'] * 1.0e03,
              'tau_syn': syn_tau,  # in ms
              'V_dynamics_method': 'linear_exact'}  # 'linear_forward_euler' or 'linear_exact'
    return params


def lif_r_psc_aibs_converter(config, syn_tau=[5.5, 8.5, 2.8, 5.8]):
    """Creates a nest glif_lif_r_psc object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    params = {'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'g': coeffs['G'] / config['R_input'] * 1.0e09,
              'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'C_m': coeffs['C'] * config['C'] * 1.0e12,
              't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
              'a_spike': threshold_params['a_spike'] * 1.0e03,
              'b_spike': threshold_params['b_spike'] * 1.0e-03,
              'a_reset': reset_params['a'],
              'b_reset': reset_params['b'] * 1.0e03,
              'tau_syn': syn_tau,  # in ms
              'V_dynamics_method': 'linear_exact'}
    return params


def lif_asc_psc_aibs_converter(config, syn_tau=[5.5, 8.5, 2.8, 5.8]):
    """Creates a nest glif_lif_asc_psc object"""
    coeffs = config['coeffs']
    params={'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
            'g': coeffs['G'] / config['R_input'] * 1.0e09,
            'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
            'C_m': coeffs['C'] * config['C'] * 1.0e12,
            't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
            'V_reset': config['El_reference'] * 1.0e03,
            'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
            'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
            'asc_amps': np.array(config['asc_amp_array']) * np.array(coeffs['asc_amp_array']) * 1.0e12,
            'tau_syn': syn_tau,  # in ms
            'V_dynamics_method': 'linear_exact'}
    return params


def lif_r_asc_psc_aibs_converter(config, syn_tau=[5.5, 8.5, 2.8, 5.8]):
    """Creates a nest glif_lif_r_asc_psc object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    params = {'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'g': coeffs['G'] / config['R_input'] * 1.0e09,
              'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'C_m': coeffs['C'] * config['C'] * 1.0e12,
              't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
              'a_spike': threshold_params['a_spike'] * 1.0e03,
              'b_spike': threshold_params['b_spike'] * 1.0e-03,
              'a_reset': reset_params['a'],
              'b_reset': reset_params['b'] * 1.0e03,
              'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
              'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
              'asc_amps': np.array(config['asc_amp_array']) * np.array(coeffs['asc_amp_array']) * 1.0e12,
              'tau_syn': syn_tau,  # in ms
              'V_dynamics_method': 'linear_exact'}
    return params


def lif_r_asc_a_psc_aibs_converter(config, syn_tau=[5.5, 8.5, 2.8, 5.8]):
    """Creates a nest glif_lif_r_asc_a_psc object"""
    coeffs = config['coeffs']
    threshold_params = config['threshold_dynamics_method']['params']
    reset_params = config['voltage_reset_method']['params']
    params = {'V_th': coeffs['th_inf'] * config['th_inf'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'g': coeffs['G'] / config['R_input'] * 1.0e09,
              'E_L': config['El'] * 1.0e03 + config['El_reference'] * 1.0e03,
              'C_m': coeffs['C'] * config['C'] * 1.0e12,
              't_ref': config['spike_cut_length'] * config['dt'] * 1.0e03,
              'a_spike': threshold_params['a_spike'] * 1.0e03,
              'b_spike': threshold_params['b_spike'] * 1.0e-03,
              'a_voltage': threshold_params['a_voltage'] * coeffs['a'] * 1.0e-03,
              'b_voltage': threshold_params['b_voltage'] * coeffs['b'] * 1.0e-03,
              'a_reset': reset_params['a'],
              'b_reset': reset_params['b'] * 1.0e03,
              'asc_init': np.array(config['init_AScurrents']) * 1.0e12,
              'k': 1.0 / np.array(config['asc_tau_array']) * 1.0e-03,
              'asc_amps': np.array(config['asc_amp_array']) * np.array(coeffs['asc_amp_array']) * 1.0e12,
              'tau_syn': syn_tau,  # in ms
              'V_dynamics_method': 'linear_exact'}
    return params


converter_map = {
    'nest:glif_lif': lif_aibs_converter,
    'nest:glif_lif_r': lif_r_aibs_converter,
    'nest:glif_lif_asc': lif_asc_aibs_converter,
    'nest:glif_lif_r_asc': lif_r_asc_aibs_converter,
    'nest:glif_lif_r_asc_a': lif_r_asc_a_aibs_converter,
    'nest:glif_lif_psc': lif_psc_aibs_converter,
    'nest:glif_lif_r_psc': lif_r_psc_aibs_converter,
    'nest:glif_lif_asc_psc': lif_asc_psc_aibs_converter,
    'nest:glif_lif_r_asc_psc': lif_r_asc_psc_aibs_converter,
    'nest:glif_lif_r_asc_a_psc': lif_r_asc_a_psc_aibs_converter
}


def convert_aibs2nest(model_template, dynamics_params):
    if model_template in converter_map:
        return converter_map[model_template](dynamics_params)
    else:
        return dynamics_params
