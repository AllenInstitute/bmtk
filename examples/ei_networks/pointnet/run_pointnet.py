from bmtk.simulator import pointnet
import sys, os

def main(config_file):
    config_file = config_file 
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:    
    	print('Please give a config file')
