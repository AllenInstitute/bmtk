from .swc_reader import SWCReader
from .swc_reader import get_swc


def rand_syn_locations(src, trg, sections=('soma', 'apical', 'basal'), distance_range=(0.0, 1.0e20),
                       morphology_dir='./components/morphologies'):
    trg_swc = get_swc(trg, morphology_dir=morphology_dir, use_cache=True)

    sec_ids, seg_xs = trg_swc.choose_sections(sections, distance_range, n_sections=1)
    sec_id, seg_x = sec_ids[0], seg_xs[0]
    swc_id, swc_dist = trg_swc.get_swc_id(sec_id, seg_x)
    # coords = trg_swc.get_coords(sec_id, seg_x)

    return [sec_id, seg_x, swc_id, swc_dist]  # coords[0], coords[y], coords[z]
