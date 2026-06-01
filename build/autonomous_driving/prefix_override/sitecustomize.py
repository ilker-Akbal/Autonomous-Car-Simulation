import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ilker/Masaüstü/Autonomous-Car-Simulation/install/autonomous_driving'
