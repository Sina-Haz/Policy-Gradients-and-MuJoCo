from utils import *
from value import *

"""
In this file we simplify our problem to be one with deterministic actions to see if we can do better with that
"""

moves = torch.tensor([[0, 0], [.25, 0], [0, .25], [-.25, 0], [0, -.25]], dtype=float)
