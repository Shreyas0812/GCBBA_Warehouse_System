"""
Task class for warehouse GCBBA
"""

import numpy as np


class GCBBA_Task:
    """
    Task class, defined by an id, an induct position (x,y,z), and eject position (x,y,z)
    """
    def __init__(self, id, char_t):
        self.id = id
        self.induct_pos = np.array([char_t[0], char_t[1], char_t[2]])
        self.eject_pos = np.array([char_t[3], char_t[4], char_t[5]])