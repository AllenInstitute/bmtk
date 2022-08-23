import pytest
import numpy as np
from collections import namedtuple
from conftest import *
from neuron import h

from bmtk.simulator.bionet import utils
from bmtk.simulator.bionet.nrn import load_neuron_modules
from bmtk.simulator.bionet.morphology import Morphology, Morphology as MorphologyOLD


EdgeType = namedtuple('EdgeType', ['target_sections', 'target_distance'])


def load_hobj():
    load_neuron_modules(mechanisms_dir='components/mechanisms', templates_dir='.')
    # load_templates(os.path.join(bionet_dir, 'default_templates'))
    hobj = h.Biophys1('components/morphology/rorb_480169178_morphology.swc')
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


def test_base():
    hobj = load_hobj()
    morph = Morphology(hobj)
    # morph.set_seg_props()
    # seg_coords = morph.calc_seg_coords()

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


# def test_base():
#     hobj = load_hobj()
#     morph = MorphologyOLD(hobj)
#     morph.set_seg_props()
#     seg_coords = morph.calc_seg_coords()
#
#     assert(morph.nseg == 87)
#     assert(len(morph.seg_prop['type']) == 87)
#     assert(len(morph.seg_prop['area']) == 87)
#     assert(len(morph.seg_prop['x']) == 87)
#     assert(len(morph.seg_prop['dist']) == 87)
#     assert(len(morph.seg_prop['length']) == 87)
#     assert(morph.seg_prop['type'][0] == 1 and morph.seg_prop['type'][1] == 2)
#     assert(np.isclose(morph.seg_prop['area'][0], 425.74, atol=1.0e-2))
#     assert(morph.seg_prop['x'][0] == 0.5)
#     assert(np.isclose(morph.seg_prop['dist'][0], 5.82, atol=1.0e-2))
#     assert(np.isclose(morph.seg_prop['length'][0], 11.64, atol=1.0e-2))
#
#     assert(isinstance(seg_coords['p0'], np.ndarray) and seg_coords['p0'].shape == (3, 87))
#     assert(isinstance(seg_coords['p0'], np.ndarray) and seg_coords['p1'].shape == (3, 87))
#     assert(isinstance(seg_coords['p0'], np.ndarray) and seg_coords['p05'].shape == (3, 87))
#     assert(np.allclose(seg_coords['p0'][:, 0], [-5.82, 0.0, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p05'][:, 0], [0.0, 0.0, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p1'][:, 0], [5.82, 0.0, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p0'][:, 86], [-22.21, -18.18, -1.89], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p05'][:, 86], [-38.98, -12.29, -4.19], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p1'][:, 86], [-51.45, 0.73, -6.35], atol=1.0e-2))


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


# def OLD_test_full():
#     hobj = load_hobj()
#
#     # delete and replace axon
#     for sec in hobj.axon:
#         h.delete_section(sec=sec)
#     h.execute('create axon[2]', hobj)
#     for sec in hobj.axon:
#         sec.L = 30
#         sec.diam = 1
#         hobj.axonal.append(sec=sec)
#         hobj.all.append(sec=sec)  # need to remove this comment
#     hobj.axon[0].connect(hobj.soma[0], 0.5, 0)
#     hobj.axon[1].connect(hobj.axon[0], 1, 0)
#     h.define_shape()
#
#     # set max segment length
#     dL = 20.0
#     for sec in hobj.all:
#         sec.nseg = 1 + 2 * int(sec.L / (2 * dL))
#
#     morph = MorphologyOLD(hobj)
#     morph.set_seg_props()
#     seg_coords = morph.calc_seg_coords()
#     # print(seg_coords['p05'][:, 0])
#
#     # Rotate cell
#     phi_x = 90.0
#     phi_y = 180.0
#     phi_z = 0.0
#     pos_soma = np.array([100.0, -100.0, 0.0]).reshape((3, 1))
#     RotX = utils.rotation_matrix([1, 0, 0], phi_x)
#     RotY = utils.rotation_matrix([0, 1, 0], phi_y)  # rotate segments around yaxis normal to pia
#     RotZ = utils.rotation_matrix([0, 0, 1], phi_z)  # rotate segments around zaxis to get a proper orientation
#     RotXYZ = np.dot(RotX, RotY.dot(RotZ))
#     seg_coords['p0'] = pos_soma + np.dot(RotXYZ, seg_coords['p0'])
#     seg_coords['p1'] = pos_soma + np.dot(RotXYZ, seg_coords['p1'])
#     seg_coords['p05'] = pos_soma + np.dot(RotXYZ, seg_coords['p05'])
#
#     assert(len(morph.seg_prop['type']) == 206)
#     assert(len(morph.seg_prop['area']) == 206)
#     assert(len(morph.seg_prop['x']) == 206)
#     assert(len(morph.seg_prop['dist']) == 206)
#     assert(len(morph.seg_prop['length']) == 206)
#     assert(morph.seg_prop['type'][0] == 1 and morph.seg_prop['type'][1] == 3 and morph.seg_prop['type'][-1] == 2)
#     assert(np.isclose(morph.seg_prop['area'][0], 425.74, atol=1.0e-2))
#     assert(morph.seg_prop['x'][0] == 0.5)
#     assert(np.isclose(morph.seg_prop['dist'][0], 5.82, atol=1.0e-2))
#     assert(np.isclose(morph.seg_prop['length'][0], 11.64, atol=1.0e-2))
#
#
#     assert(isinstance(seg_coords['p0'], np.ndarray) and seg_coords['p0'].shape == (3, 206))
#     assert(isinstance(seg_coords['p0'], np.ndarray) and seg_coords['p1'].shape == (3, 206))
#     assert(isinstance(seg_coords['p0'], np.ndarray) and seg_coords['p05'].shape == (3, 206))
#     assert(np.allclose(seg_coords['p0'][:, 0], [103.48, -95.83, 2.08], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p05'][:, 0], [100.0, -100.0, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p1'][:, 0], [96.51, -104.16, -2.08], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p0'][:, -1], [93.00, -120.74858452, 20.50896093], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p05'][:, -1], [89.51, -131.12281575, 30.76346789], atol=1.0e-2))
#     assert(np.allclose(seg_coords['p1'][:, -1], [86.01, -141.49, 41.01], atol=1.0e-2))


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
    assert(np.allclose(morph2.seg_coords.p0[:, 0], [52.60793758495786, -47.66851250016918, 305.3306848207208], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, 0], [49.999927071351536, -50.00006519796841, 300.67876991698654], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p1[:, 0], [47.391916557745205, -52.33161789576764, 296.02685501325226], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p0[:, -1], [20.06251275325579, -49.132177777673576, 302.41037925570913], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p05[:, -1], [5.093809548685826, -48.69823418216671, 303.27618369633956], atol=1.0e-2))
    assert(np.allclose(morph2.seg_coords.p1[:, -1], [-9.87490156483996, -48.26429035737874, 304.1419885944318], atol=1.0e-2))


# def OLD_test_multicell():
#     def rotate_and_move(coords, soma_pos, rot):
#         soma_pos = np.array(soma_pos).reshape((3, 1))
#         RotX = utils.rotation_matrix([1, 0, 0], rot[0])
#         RotY = utils.rotation_matrix([0, 1, 0], rot[1])  # rotate segments around yaxis normal to pia
#         RotZ = utils.rotation_matrix([0, 0, 1], rot[2])  # rotate segments around zaxis to get a proper orientation
#         RotXYZ = np.dot(RotX, RotY.dot(RotZ))
#         return {
#             'p0': soma_pos + np.dot(RotXYZ, coords['p0']),
#             'p1': soma_pos + np.dot(RotXYZ, coords['p1']),
#             'p05': soma_pos + np.dot(RotXYZ, coords['p05'])
#         }
#
#     hobj = load_hobj()
#
#     # delete and replace axon
#     for sec in hobj.axon:
#         h.delete_section(sec=sec)
#     h.execute('create axon[2]', hobj)
#     for sec in hobj.axon:
#         sec.L = 30
#         sec.diam = 1
#         hobj.axonal.append(sec=sec)
#         hobj.all.append(sec=sec)  # need to remove this comment
#     hobj.axon[0].connect(hobj.soma[0], 0.5, 0)
#     hobj.axon[1].connect(hobj.axon[0], 1, 0)
#     h.define_shape()
#
#     # set max segment length
#     dL = 20.0
#     for sec in hobj.all:
#         sec.nseg = 1 + 2 * int(sec.L / (2 * dL))
#
#     morph = MorphologyOLD(hobj)
#     morph.set_seg_props()
#     seg_coords1 = morph.calc_seg_coords()
#     seg_coords1 = rotate_and_move(seg_coords1, [0.0, 100.0, 0.0], [0.0, 0.0, 0.0])
#     assert(len(morph.seg_prop['type']) == 206)
#     assert(len(morph.seg_prop['area']) == 206)
#     assert(len(morph.seg_prop['x']) == 206)
#     assert(len(morph.seg_prop['dist']) == 206)
#     assert(len(morph.seg_prop['length']) == 206)
#     assert(morph.seg_prop['type'][0] == 1 and morph.seg_prop['type'][1] == 3 and morph.seg_prop['type'][-1] == 2)
#     assert(np.isclose(morph.seg_prop['area'][0], 425.74, atol=1.0e-2))
#     assert(morph.seg_prop['x'][0] == 0.5)
#     assert(np.isclose(morph.seg_prop['dist'][0], 5.82, atol=1.0e-2))
#     assert(np.isclose(morph.seg_prop['length'][0], 11.64, atol=1.0e-2))
#
#     assert(isinstance(seg_coords1['p0'], np.ndarray) and seg_coords1['p0'].shape == (3, 206))
#     assert(isinstance(seg_coords1['p0'], np.ndarray) and seg_coords1['p1'].shape == (3, 206))
#     assert(isinstance(seg_coords1['p0'], np.ndarray) and seg_coords1['p05'].shape == (3, 206))
#     assert(np.allclose(seg_coords1['p0'][:, 0], [-5.82, 100.0, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords1['p05'][:, 0], [0.0, 100.0, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords1['p1'][:, 0], [5.82, 100.0, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords1['p0'][:, -1], [11.68, 127.63, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords1['p05'][:, -1], [17.52, 141.44, 0.0], atol=1.0e-2))
#     assert(np.allclose(seg_coords1['p1'][:, -1], [23.36, 155.26, 0.0], atol=1.0e-2))
#
#     seg_coords2 = morph.calc_seg_coords()
#     seg_coords2 = rotate_and_move(seg_coords2, [50.0, -50.0, 300.6789], [-90.0, 0.0, 90.0])
#     assert(np.allclose(seg_coords2['p0'][:, 0], [52.60793758495786, -47.66851250016918, 305.3306848207208], atol=1.0e-2))
#     assert(np.allclose(seg_coords2['p05'][:, 0], [49.999927071351536, -50.00006519796841, 300.67876991698654], atol=1.0e-2))
#     assert(np.allclose(seg_coords2['p1'][:, 0], [47.391916557745205, -52.33161789576764, 296.02685501325226], atol=1.0e-2))
#     assert(np.allclose(seg_coords2['p0'][:, -1], [20.06251275325579, -49.132177777673576, 302.41037925570913], atol=1.0e-2))
#     assert(np.allclose(seg_coords2['p05'][:, -1], [5.093809548685826, -48.69823418216671, 303.27618369633956], atol=1.0e-2))
#     assert(np.allclose(seg_coords2['p1'][:, -1], [-9.87490156483996, -48.26429035737874, 304.1419885944318], atol=1.0e-2))


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


def test_get_target_segments():
    hobj = load_hobj()
    fix_axon_peri(hobj)

    morph = Morphology(hobj)
    morph.set_segment_dl(20.0)
    morph = morph.move_and_rotate(
        soma_coords=[100.0, -100.0, 0.0],
        rotation_angles=[90.0, 180.0, 0.0]
    )


    # set max segment length
    # dL = 20.0
    # for sec in hobj.all:
    #     sec.nseg = 1 + 2 * int(sec.L / (2 * dL))

    # morph = MorphologyOLD(hobj)
    # morph.set_seg_props()
    # seg_coords = morph.calc_seg_coords()
    # # print(seg_coords['p05'][:, 0])
    #
    # # Rotate cell
    # phi_x = 90.0
    # phi_y = 180.0
    # phi_z = 0.0
    # pos_soma = np.array([100.0, -100.0, 0.0]).reshape((3, 1))
    # RotX = utils.rotation_matrix([1, 0, 0], phi_x)
    # RotY = utils.rotation_matrix([0, 1, 0], phi_y)  # rotate segments around yaxis normal to pia
    # RotZ = utils.rotation_matrix([0, 0, 1], phi_z)  # rotate segments around zaxis to get a proper orientation
    # RotXYZ = np.dot(RotX, RotY.dot(RotZ))
    # seg_coords['p0'] = pos_soma + np.dot(RotXYZ, seg_coords['p0'])
    # seg_coords['p1'] = pos_soma + np.dot(RotXYZ, seg_coords['p1'])
    # seg_coords['p05'] = pos_soma + np.dot(RotXYZ, seg_coords['p05'])

    # # TODO: Make sure it's able to work with arrays and lists.
    # secs, _ = morph.get_target_segments(edge_type=EdgeType(target_sections=['dend', 'apic', 'soma', 'axon'], target_distance=[0.0, 1.0e20]))
    # assert(len(secs) == morph.nseg)

    edge_type = EdgeType(target_sections=('dend', 'apic', 'soma', 'axon'), target_distance=(0.0, 1.0e20))
    secs, _ = morph.get_target_segments(edge_type=edge_type)
    assert(len(secs) == morph.nseg)

    # Check to see if it's cached
    secs, _ = morph.get_target_segments(edge_type=edge_type)
    assert(len(secs) == morph.nseg)

    secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=('dend', 'apic'), target_distance=(150.0, 200.0)))
    assert(np.isclose(np.sum(sec_probs), 1.0, atol=1.0e-3))
    for sec in secs:
        assert(morph.seg_prop['type'][sec] in [3, 4])
        assert(10.0 <= morph.seg_prop['dist1'][sec] and morph.seg_prop['dist0'][sec] <= 200.0)

    # secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=('axon'), target_distance=(0.0, 10.0e20)))
    # for sec in secs:
    #     assert (morph.seg_prop['type'][sec] == 2)
    #
    # secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=('soma'), distance_range=(300.0, 500.0]))
    # assert(len(secs) == 0)
    # assert(len(sec_probs) == 0)
    #
    # secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=['soma'], distance_range=(0.0, 100.0)))
    # assert(len(secs) == 1 and sec_probs[0] == 1.0)


# def OLD_test_get_target_segments():
#     hobj = load_hobj()
#
#     # delete and replace axon
#     for sec in hobj.axon:
#         h.delete_section(sec=sec)
#     h.execute('create axon[2]', hobj)
#     for sec in hobj.axon:
#         sec.L = 30
#         sec.diam = 1
#         hobj.axonal.append(sec=sec)
#         hobj.all.append(sec=sec)  # need to remove this comment
#     hobj.axon[0].connect(hobj.soma[0], 0.5, 0)
#     hobj.axon[1].connect(hobj.axon[0], 1, 0)
#     h.define_shape()
#
#     # set max segment length
#     dL = 20.0
#     for sec in hobj.all:
#         sec.nseg = 1 + 2 * int(sec.L / (2 * dL))
#
#     morph = MorphologyOLD(hobj)
#     morph.set_seg_props()
#     seg_coords = morph.calc_seg_coords()
#     # print(seg_coords['p05'][:, 0])
#
#     # Rotate cell
#     phi_x = 90.0
#     phi_y = 180.0
#     phi_z = 0.0
#     pos_soma = np.array([100.0, -100.0, 0.0]).reshape((3, 1))
#     RotX = utils.rotation_matrix([1, 0, 0], phi_x)
#     RotY = utils.rotation_matrix([0, 1, 0], phi_y)  # rotate segments around yaxis normal to pia
#     RotZ = utils.rotation_matrix([0, 0, 1], phi_z)  # rotate segments around zaxis to get a proper orientation
#     RotXYZ = np.dot(RotX, RotY.dot(RotZ))
#     seg_coords['p0'] = pos_soma + np.dot(RotXYZ, seg_coords['p0'])
#     seg_coords['p1'] = pos_soma + np.dot(RotXYZ, seg_coords['p1'])
#     seg_coords['p05'] = pos_soma + np.dot(RotXYZ, seg_coords['p05'])
#
#     # # TODO: Make sure it's able to work with arrays and lists.
#     # secs, _ = morph.get_target_segments(edge_type=EdgeType(target_sections=['dend', 'apic', 'soma', 'axon'], target_distance=[0.0, 1.0e20]))
#     # assert(len(secs) == morph.nseg)
#
#     edge_type = EdgeType(target_sections=('dend', 'apic', 'soma', 'axon'), target_distance=(0.0, 1.0e20))
#     secs, _ = morph.get_target_segments(edge_type=edge_type)
#     assert(len(secs) == morph.nseg)
#
#     # Check to see if it's cached
#     secs, _ = morph.get_target_segments(edge_type=edge_type)
#     assert(len(secs) == morph.nseg)
#
#     secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=('dend', 'apic'), target_distance=(150.0, 200.0)))
#     assert(np.isclose(np.sum(sec_probs), 1.0, atol=1.0e-3))
#     for sec in secs:
#         assert(morph.seg_prop['type'][sec] in [3, 4])
#         assert(10.0 <= morph.seg_prop['dist1'][sec] and morph.seg_prop['dist0'][sec] <= 200.0)
#
#     # secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=('axon'), target_distance=(0.0, 10.0e20)))
#     # for sec in secs:
#     #     assert (morph.seg_prop['type'][sec] == 2)
#     #
#     # secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=('soma'), distance_range=(300.0, 500.0]))
#     # assert(len(secs) == 0)
#     # assert(len(sec_probs) == 0)
#     #
#     # secs, sec_probs = morph.get_target_segments(EdgeType(target_sections=['soma'], distance_range=(0.0, 100.0)))
#     # assert(len(secs) == 1 and sec_probs[0] == 1.0)


def test_get_swc_id():
    hobj = load_hobj()
    fix_axon_peri(hobj)


    morph = Morphology(hobj, swc_path='components/morphology/rorb_480169178_morphology.swc')
    morph.set_segment_dl(2.0)
    # morph.set_segment_dl(20.0)
    # morph = morph.move_and_rotate(
    #     soma_coords=[100.0, -100.0, 0.0],
    #     rotation_angles=[90.0, 180.0, 0.0]
    # )

    for sec in morph.hobj.all:
        for seg in sec:
            print(sec, seg, seg.x)

    print(morph.seg_props.sec_id)
    print(morph.seg_props.x)
    print(morph.seg_props.dist0)
    print(morph.seg_props.dist1)

    # print(morph.nseg)
    # print()
    # print(morph.segments)
    # print(morph.get_swc_id(1, 0.5))


    # morph = Morphology(hobj, swc_path='components/morphology/rorb_480169178_morphology.swc')
    # print(morph.sections)
    # morph.set_segment_dl(20.0)
    # print(morph.get_swc_id(2, 0.0))
    # print(morph.swc_map)


if __name__ == '__main__':
    # test_base()
    # test_full()
    # test_multicell()
    # test_find_sections()
    test_get_target_segments()
    # test_get_swc_id()
