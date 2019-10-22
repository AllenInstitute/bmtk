try:
    from bmtk.simulator import pointnet

    nest_installed = True

except ImportError:
    nest_installed = False