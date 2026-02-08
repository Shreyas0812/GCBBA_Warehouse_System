"""
Task class for warehouse GCBBA
"""

import numpy as np


class GCBBA_Task:
    """
    Task class, defined by an id, a induct position (x,y), and eject position (x,y)
    """
    def __init__(self, id, char_t):
        self.id = id
        self.induct_pos = np.array([char_t[0], char_t[1]])
        self.eject_pos = np.array([char_t[2], char_t[3]])