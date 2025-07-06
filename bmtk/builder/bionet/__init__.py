from .swc_reader import SWCReader
from .swc_reader import get_swc


def rand_syn_locations(src, trg, sections=('soma', 'apical', 'basal'), distance_range=(0.0, 1.0e20),
                       morphology_dir='./components/morphologies', return_swc=True, return_coords=False, dL=None):
    trg_swc = get_swc(trg, morphology_dir=morphology_dir, use_cache=True, dL=dL)

    sec_ids, seg_xs = trg_swc.choose_sections(sections, distance_range, n_sections=1)
    sec_id, seg_x = sec_ids[0], seg_xs[0]
    ret_vals = [sec_id, seg_x]

    if return_swc:
        ret_vals.extend(trg_swc.get_swc_id(sec_id, seg_x))
    
    if return_coords:
        ret_vals.extend(trg_swc.get_coords(sec_id, seg_x))
        
    return ret_vals
