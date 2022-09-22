import os
import pytest
import numpy as np
from collections import namedtuple
from neuron import h

try:
    from conftest import *
except ModuleNotFoundError as mnfe:
    from .conftest import *

from bmtk.simulator.bionet.nrn import load_neuron_modules
from bmtk.simulator.bionet.morphology import Morphology


RORB_SWC_PATH = os.path.join(MORPH_DIR, 'rorb_480169178_morphology.swc')

try:
    load_neuron_modules(mechanisms_dir='components/mechanisms', templates_dir='.')
    has_mechanism = True
except AttributeError as ae:
    has_mechanism = False


def load_hobj():
    # load_neuron_modules(mechanisms_dir='components/mechanisms', templates_dir='.')
    # load_templates(os.path.join(bionet_dir, 'default_templates'))
    hobj = h.Biophys1(RORB_SWC_PATH)
    return hobj


def fix_axon_peri(hobj):
    """Replace reconstructed axon with a stub"""
    for sec in hobj.axon:
        h.delete_section(sec=sec)

    h.execute('create axon[2]', hobj)
    for sec in hobj.axon:
        sec.L = 30
        sec.diam = 1
        hobj.axonal.append(sec=sec)
        hobj.all.append(sec=sec)  # need to remove this comment

    hobj.axon[0].connect(hobj.soma[0], 0.5, 0)
    hobj.axon[1].connect(hobj.axon[0], 1, 0)
    h.define_shape()


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_base():
    hobj = load_hobj()
    morph = Morphology(hobj)

    assert(morph.nseg == 87)
    assert(len(morph.seg_props.type) == 87)
    assert(len(morph.seg_props.area) == 87)
    assert(len(morph.seg_props.x) == 87)
    assert(len(morph.seg_props.dist) == 87)
    assert(len(morph.seg_props.length) == 87)
    assert(morph.seg_props.type[0] == 1 and morph.seg_props.type[1] == 2)
    assert(np.isclose(morph.seg_props.area[0], 425.74, atol=1.0e-2))
    assert(morph.seg_props.x[0] == 0.5)
    assert(np.isclose(morph.seg_props.dist[0], 5.82, atol=1.0e-2))
    assert(np.isclose(morph.seg_props.length[0], 11.64, atol=1.0e-2))

    assert(isinstance(morph.seg_coords.p0, np.ndarray) and morph.seg_coords.p0.shape == (3, 87))
    assert(isinstance(morph.seg_coords.p0, np.ndarray) and morph.seg_coords.p1.shape == (3, 87))
    assert(isinstance(morph.seg_coords.p0, np.ndarray) and morph.seg_coords.p05.shape == (3, 87))
    assert(np.allclose(morph.seg_coords.p0[:, 0], [-5.82, 0.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p05[:, 0], [0.0, 0.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p1[:, 0], [5.82, 0.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p0[:, 86], [-22.21, -18.18, -1.89], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p05[:, 86], [-38.98, -12.29, -4.19], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p1[:, 86], [-51.45, 0.73, -6.35], atol=1.0e-2))


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_seg_props_cache():
    # Check ability to cache segment props
    hobj = load_hobj()
    morph1 = Morphology.load(hobj, morphology_file=RORB_SWC_PATH, cache_seg_props=True)

    hobj = load_hobj()
    morph2 = Morphology.load(hobj, morphology_file=RORB_SWC_PATH, cache_seg_props=True)

    morph3 = Morphology.load(hobj, morphology_file=RORB_SWC_PATH, cache_seg_props=False)
    hobj = load_hobj()
    for sec in hobj.axon:
        h.delete_section(sec=sec)
    morph4 = Morphology.load(hobj, morphology_file=RORB_SWC_PATH, cache_seg_props=True)

    # morph1 and morph2 have different hobj but should share the same SegmentProps
    sp1 = morph1.seg_props
    assert(id(sp1) == id(morph1.seg_props) == id(morph2.seg_props))
    assert(id(morph1.segments[0]) != id(morph2.segments[0]))

    # morph3 explicity did not cache seg_props
    assert(id(morph2) != id(morph3))

    # morph4 has the same morphology_file, but not the same number of segments since axon with removed
    assert (id(morph1) != id(morph4))


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_full():
    hobj = load_hobj()
    fix_axon_peri(hobj)

    morph = Morphology(hobj)
    morph.set_segment_dl(20.0)
    morph = morph.move_and_rotate(
        soma_coords=[100.0, -100.0, 0.0],
        rotation_angles=[90.0, 180.0, 0.0]
    )

    assert(len(morph.seg_props.type) == 206)
    assert(len(morph.seg_props.area) == 206)
    assert(len(morph.seg_props.x) == 206)
    assert(len(morph.seg_props.dist) == 206)
    assert(len(morph.seg_props.length) == 206)
    assert(morph.seg_props.type[0] == 1 and morph.seg_props.type[1] == 3 and morph.seg_props.type[-1] == 2)
    assert(np.isclose(morph.seg_props.area[0], 425.74, atol=1.0e-2))
    assert(morph.seg_props.x[0] == 0.5)
    assert(np.isclose(morph.seg_props.dist[0], 5.82, atol=1.0e-2))
    assert(np.isclose(morph.seg_props.length[0], 11.64, atol=1.0e-2))

    assert(isinstance(morph.seg_coords.p0, np.ndarray) and morph.seg_coords.p0.shape == (3, 206))
    assert(isinstance(morph.seg_coords.p1, np.ndarray) and morph.seg_coords.p1.shape == (3, 206))
    assert(isinstance(morph.seg_coords.p05, np.ndarray) and morph.seg_coords.p05.shape == (3, 206))
    assert(np.allclose(morph.seg_coords.p0[:, 0], [103.48, -95.83, 2.08], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p05[:, 0], [100.0, -100.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p1[:, 0], [96.51, -104.16, -2.08], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p0[:, -1], [93.00, -120.74858452, 20.50896093], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p05[:, -1], [89.51, -131.12281575, 30.76346789], atol=1.0e-2))
    assert(np.allclose(morph.seg_coords.p1[:, -1], [86.01, -141.49, 41.01], atol=1.0e-2))


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_multicell():
    hobj = load_hobj()
    fix_axon_peri(hobj)

    morph = Morphology(hobj)
    morph.set_segment_dl(20.0)
    morph1 = morph.move_and_rotate(
        soma_coords=[0.0, 100.0, 0.0],
        rotation_angles=[0.0, 0.0, 0.0]
    )
    morph2 = morph.move_and_rotate(
        soma_coords=[50.0, -50.0, 300.6789],
        rotation_angles=[-90.0, 0.0, 90.0]
    )

    assert(len(morph1.seg_props.type) == len(morph2.seg_props.type) == 206)
    assert(len(morph1.seg_props.area) == 206)
    assert(len(morph1.seg_props.x) == 206)
    assert(len(morph1.seg_props.dist) == 206)
    assert(len(morph1.seg_props.length) == 206)
    assert(morph1.seg_props.type[0] == 1 and morph1.seg_props.type[1] == 3 and morph1.seg_props.type[-1] == 2)
    assert(np.isclose(morph1.seg_props.area[0], 425.74, atol=1.0e-2))
    assert(morph1.seg_props.x[0] == 0.5)
    assert(np.isclose(morph1.seg_props.dist[0], 5.82, atol=1.0e-2))
    assert(np.isclose(morph1.seg_props.length[0], 11.64, atol=1.0e-2))

    assert(isinstance(morph1.seg_coords.p0, np.ndarray) and morph1.seg_coords.p0.shape == (3, 206))
    assert(isinstance(morph1.seg_coords.p0, np.ndarray) and morph1.seg_coords.p1.shape == (3, 206))
    assert(isinstance(morph1.seg_coords.p0, np.ndarray) and morph1.seg_coords.p05.shape == (3, 206))
    assert(np.allclose(morph1.seg_coords.p0[:, 0], [-5.82, 100.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p05[:, 0], [0.0, 100.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p1[:, 0], [5.82, 100.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p0[:, -1], [11.68, 127.63, 0.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p05[:, -1], [17.52, 141.44, 0.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p1[:, -1], [23.36, 155.26, 0.0], atol=1.0e-2))

    assert(isinstance(morph2.seg_coords.p0, np.ndarray) and morph2.seg_coords.p0.shape == (3, 206))
    assert(isinstance(morph2.seg_coords.p0, np.ndarray) and morph2.seg_coords.p1.shape == (3, 206))
    assert(isinstance(morph2.seg_coords.p0, np.ndarray) and morph2.seg_coords.p05.shape == (3, 206))
    assert(np.allclose(morph2.seg_coords.p0[:, 0], [52.60793758495786, -47.66851250016918, 305.3306848207208],
                       atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, 0], [49.999927071351536, -50.00006519796841, 300.67876991698654],
                       atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p1[:, 0], [47.391916557745205, -52.33161789576764, 296.02685501325226],
                       atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p0[:, -1], [20.06251275325579, -49.132177777673576, 302.41037925570913],
                       atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, -1], [5.093809548685826, -48.69823418216671, 303.27618369633956],
                       atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p1[:, -1], [-9.87490156483996, -48.26429035737874, 304.1419885944318],
                       atol=1.0e-2))


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_find_sections():
    hobj = load_hobj()
    fix_axon_peri(hobj)

    morph = Morphology(hobj)
    morph.set_segment_dl(20.0)
    morph = morph.move_and_rotate(
        soma_coords=[100.0, -100.0, 0.0],
        rotation_angles=[90.0, 180.0, 0.0]
    )

    secs, _ = morph.find_sections(section_names=['dend', 'apic', 'soma', 'axon'], distance_range=[0.0, 1.0e20])
    assert(len(secs) == morph.nseg)

    secs, _ = morph.find_sections(section_names=('dend', 'apic', 'soma', 'axon'), distance_range=(0.0, 1.0e20))
    assert(len(secs) == morph.nseg)

    secs, sec_probs = morph.find_sections(section_names=['dend', 'apic'], distance_range=[150.0, 200.0])
    assert(np.isclose(np.sum(sec_probs), 1.0, atol=1.0e-3))
    for sec in secs:
        assert(morph.seg_props.type[sec] in [3, 4])
        assert(10.0 <= morph.seg_props.dist1[sec] and morph.seg_props.dist0[sec] <= 200.0)

    secs, sec_probs = morph.find_sections(section_names=['axon'], distance_range=[0.0, 10.0e20])
    for sec in secs:
        assert (morph.seg_props.type[sec] == 2)

    secs, sec_probs = morph.find_sections(section_names=['soma'], distance_range=[300.0, 500.0])
    assert(len(secs) == 0)
    assert(len(sec_probs) == 0)

    secs, sec_probs = morph.find_sections(section_names=['soma'], distance_range=[0.0, 100.0])
    assert(len(secs) == 1 and sec_probs[0] == 1.0)


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
@pytest.mark.skip(reason='no longer using get_target_segments() method in new Morphology')
def test_get_target_segments():
    hobj = load_hobj()
    fix_axon_peri(hobj)

    morph = Morphology(hobj)
    morph.set_segment_dl(20.0)
    morph = morph.move_and_rotate(
        soma_coords=[100.0, -100.0, 0.0],
        rotation_angles=[90.0, 180.0, 0.0]
    )

    EdgeType = namedtuple('EdgeType', ['target_sections', 'target_distance'])
    edge_type = EdgeType(target_sections=('dend', 'apic', 'soma', 'axon'), target_distance=(0.0, 1.0e20))
    secs, _ = morph.get_target_segments(edge_type=edge_type)
    assert(len(secs) == morph.nseg)

    # Check to see if it's cached
    secs, _ = morph.get_target_segments(edge_type=edge_type)
    assert(len(secs) == morph.nseg)

    secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=('dend', 'apic'),
                                                         target_distance=(150.0, 200.0)))
    assert(np.isclose(np.sum(sec_probs), 1.0, atol=1.0e-3))
    for sec in secs:
        assert(morph.seg_prop['type'][sec] in [3, 4])
        assert(10.0 <= morph.seg_prop['dist1'][sec] and morph.seg_prop['dist0'][sec] <= 200.0)


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_get_swc_id():
    hobj = load_hobj()
    fix_axon_peri(hobj)

    morph = Morphology(hobj, swc_path=RORB_SWC_PATH)
    morph.set_segment_dl(20.0)

    # check soma has swc id == 1
    assert(morph.get_swc_id(0, 0.0)[0] == 1)
    assert(morph.get_swc_id(0, 0.5)[0] == 1)
    assert(morph.get_swc_id(0, 0.5)[0] == 1)

    # check dend
    assert(morph.get_swc_id(1, 0.0)[0] == 2)
    assert(morph.get_swc_id(1, 0.0001)[0] == 2)
    assert(morph.get_swc_id(1, 1.0) == (19.0, 0.0))
    assert(morph.get_swc_id(2, 0.0) == (20.0, 0.0))

    # check apic
    assert(morph.get_swc_id(40, 0.0) == (3436, 0.0))
    assert(morph.get_swc_id(41, 0.0) == (964, 0.0))
    assert(morph.get_swc_id(85, 0.0) == (2824, 0.0))
    assert(morph.get_swc_id(85, 1.0) == (2857, 0.0))

    # check axon
    assert(morph.get_swc_id(86, 0.0) == (3495, 0.0))
    assert(morph.get_swc_id(86, 1.0) == (3508, 0.0))

    for sec_id, sec_type in zip(morph.seg_props.sec_id, morph.seg_props.type):
        swc_id, _ = morph.get_swc_id(sec_id, 0.0)
        if swc_id != -1:
            assert(morph.swc_map[morph.swc_map['id'] == swc_id]['type'].values[0] == sec_type)


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_remove_secs():
    # if you don't cut off the axon the second segment should be the axon
    hobj = load_hobj()
    morph = Morphology(hobj, swc_path=RORB_SWC_PATH)
    assert(len(morph.segments) == 87)
    assert(len(morph.sections) == 87)
    assert(len(morph.seg_props.sec_id) == 87)
    assert(morph.seg_props.sec_id[1] == 1)
    assert(morph.seg_props.type[1] == 2)
    assert(morph.seg_props.type[0] != 2)
    assert(morph.seg_props.type[2] != 2)

    # In the given swc file the last 14 ids are assigned to the axon
    assert(morph.get_swc_id(1, 0.0) == (3495, 0.0))
    assert(morph.get_swc_id(1, 1.0) == (3508, 0.0))
    for sec_x in np.linspace(0.0, 1.0, num=13, endpoint=False):
        assert(3495 <= morph.get_swc_id(1, sec_x)[0] < 3508)

    # remove the axon
    hobj = load_hobj()
    for sec in hobj.axon:
        h.delete_section(sec=sec)
    morph = Morphology(hobj, swc_path=RORB_SWC_PATH)

    assert(len(morph.seg_props.sec_id) == 86)
    assert(len(morph.segments) == 86)
    assert(len(morph.sections) == 86)
    assert(2 not in morph.seg_props.type)
    assert(set(morph.seg_props.type) == {1, 3, 4})
    axon_swc_ids = morph.swc_map[morph.swc_map['type'] == 2]['id'].values
    for sec_id in morph.seg_props.sec_id:
        for sec_x in np.linspace(0.0, 1.0, num=10, endpoint=True):
            swc_id, _ = morph.get_swc_id(sec_id, sec_x=sec_x)
            assert(sec_id not in axon_swc_ids)

    # remove apical and basal dendrites
    hobj = load_hobj()
    for sec in hobj.all:
        if 'apic' in sec.name() or 'dend' in sec.name():
            h.delete_section(sec=sec)

    morph = Morphology(hobj, swc_path=RORB_SWC_PATH)
    assert(len(morph.segments) == 2)
    assert(len(morph.sections) == 2)
    assert(len(morph.seg_props.sec_id) == 2)
    for sec_id in morph.seg_props.sec_id:
        for sec_x in np.linspace(0.0, 1.0, num=10, endpoint=True):
            swc_id, _ = morph.get_swc_id(sec_id, sec_x=sec_x)
            assert(morph.swc_map[morph.swc_map['id'] == swc_id]['type'].values[0] in [1, 2])


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_move_and_rotate():
    hobj = load_hobj()
    # fix_axon_peri(hobj)

    morph1 = Morphology(hobj)

    morph2 = morph1.move_and_rotate(
        soma_coords=[100.0, -100.0, 0.0],
        rotation_angles=[90.0, 180.0, 0.0],
        inplace=False
    )

    # Make sure inplace=False updating morph1 doesn't affect morph2
    assert(np.allclose(morph1.seg_coords.p0[:, 0], [-5.82, 0.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p05[:, 0], [0.0, 0.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p1[:, 0], [5.82, 0.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p0[:, 86], [-22.21, -18.18, -1.89], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p05[:, 86], [-38.98, -12.29, -4.19], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p1[:, 86], [-51.45, 0.73, -6.35], atol=1.0e-2))

    assert(np.allclose(morph2.seg_coords.p0[:, 0], [103.48, -95.83, 2.08], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, 0], [100.0, -100.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p1[:, 0], [96.51, -104.16, -2.08], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p0[:, -1], [114.81, -76.95,  -8.79], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, -1], [126.68, -68.81, 1.87], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p1[:, -1], [135.88, -66.87, 17.42], atol=1.0e-2))

    # Make sure inplace=True updating morph1 doesn't affect morph2
    morph1.move_and_rotate(
        soma_coords=[10.0, 10.0, 10.0],
        rotation_angles=[0.0, 0.0, 0.0],
        inplace=True
    )
    assert(np.allclose(morph1.seg_coords.p0[:, 0], [4.17966715, 10.0, 10.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p05[:, 0], [10.0, 10.0, 10.0], atol=1.0e-2))
    assert(np.allclose(morph1.seg_coords.p1[:, 0], [15.82, 10.0, 10.0], atol=1.0e-2))

    assert(np.allclose(morph2.seg_coords.p0[:, 0], [103.48, -95.83, 2.08], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, 0], [100.0, -100.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p1[:, 0], [96.51, -104.16, -2.08], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p0[:, -1], [114.81, -76.95,  -8.79], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, -1], [126.68, -68.81, 1.87], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p1[:, -1], [135.88, -66.87, 17.42], atol=1.0e-2))


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_move_and_rotate_cached():
    hobj = load_hobj()
    morph1 = Morphology.load(hobj, morphology_file=RORB_SWC_PATH, cache_seg_props=True)
    morph2 = Morphology.load(hobj, morphology_file=RORB_SWC_PATH, cache_seg_props=True)

    # initially morph1 and morph2 share the same SegmentCoords object, make sure after a move_and_rotate is done
    # the two morphology has different coordinates
    assert(morph1.seg_coords == morph2.seg_coords)

    morph2.move_and_rotate(
        soma_coords=[10.0, 10.0, 10.0],
        rotation_angles=[100.0, 100.0, 100.0],
        inplace=True
    )
    assert(morph1.seg_coords != morph2.seg_coords)
    assert(np.allclose(morph1.seg_coords.p05[:, 0], [0.0, 0.0, 0.0], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, 0], [10.0, 10.0, 10.0], atol=1.0e-2))


def check_lazy_loading():
    hobj = load_hobj()
    morph1 = Morphology.load(hobj, morphology_file=RORB_SWC_PATH, cache_seg_props=True)

    hobj = load_hobj()
    morph2 = Morphology.load(hobj, morphology_file=RORB_SWC_PATH, cache_seg_props=False)
    print(len(morph2.seg_props.x))

    print(len(morph1.seg_props.x))
    print(id(morph1.seg_props) == id(morph2.seg_props))

    # morph1.seg_props


if __name__ == '__main__':
    # test_base()
    # test_seg_props_cache()
    # test_full()
    # test_multicell()
    # test_find_sections()
    # test_get_target_segments()
    # test_get_swc_id()
    # test_remove_secs()
    # check_lazy_loading()
    # test_move_and_rotate()
    test_move_and_rotate_cached()