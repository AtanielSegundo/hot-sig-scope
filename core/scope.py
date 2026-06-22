import numpy as np

from typing import Tuple
from core.configs import ScreenCfg

'''
Encapsulates the View Aspects of an Osciloscope
'''

class Scope:
    x_max = 2
    y_max = 2
    n_divisions  = 20        # Number of Divisions in range
    resolution   = 48000     # Number of points used in range 
    n_markers    = n_divisions * 5 
    mark_size_percent = 0.1

    def __init__(self):
        pass
    
    @staticmethod
    def translate(cfg:ScreenCfg,point:Tuple[float,float]) -> Tuple[float,float]:
        '''
        Convert An Scope Coord To Screen Coord
        '''
        point      = np.array(point) * np.array((1, -1))
        scope_dims = np.array((cfg.width,cfg.height))
        scaled = (0.5 + point / (2 * np.array((Scope.x_max,Scope.y_max)))) * scope_dims
        result = (int(scaled[0]), int(scaled[1]))
        return result
