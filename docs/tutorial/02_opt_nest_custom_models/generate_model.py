import numpy as np


from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils


nestml_izh_model = """
model custom_izh_neuron:

    state:
        v mV = -65 mV    # Membrane potential in mV
        u real = 0    # Membrane potential recovery variable

    equations:
        v' = (.04 * v * v / mV + 5 * v + (140 - u) * mV + (I_e * GOhm)) / ms
        u' = a * (b * v - u * mV) / (mV * ms)

    parameters:
        a real = .02    # describes time scale of recovery variable
        b real = .2    # sensitivity of recovery variable
        c mV = -65 mV    # after-spike reset value of v
        d real = 8.    # after-spike reset value of u

    input:
        spikes <- spike
        I_e pA <- continuous

    output:
        spike

    update:
        integrate_odes()

    onReceive(spikes):
        # add synaptic current
        v += spikes * mV * s

    onCondition(v >= 30mV):
        # threshold crossing
        v = c
        u += d
        emit_spike()

"""

# generate and build code
module_name, neuron_model_name = NESTCodeGeneratorUtils.generate_code_for(
    nestml_izh_model, module_name="nestml_izh_module", target_path="components/nestml"
)

print(module_name, neuron_model_name)
