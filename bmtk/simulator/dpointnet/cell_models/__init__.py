from .glif3_cell import GLIF3Cell

cell_models = {
    'default': GLIF3Cell,
    'GLIF3Cell': GLIF3Cell
}


def get_cell_model(name, default=GLIF3Cell):
    """Registered cell class by name; ``glif3_release`` is imported on first use."""
    if name == 'glif3_release' and name not in cell_models:
        from ..recurrent_release import ReleaseGLIF3Cell
        cell_models[name] = ReleaseGLIF3Cell
    return cell_models.get(name, default)


