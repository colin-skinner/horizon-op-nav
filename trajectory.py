import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from time import time as time
from multiprocessing import Pool
import spiceypy as spice

from PlanetaryData import Body

class OrbitPropogator:
    def __init__(self, state0: np.ndarray | list, t_final: float, dt: float, bodies: list[Body]):
        """
        - State in km, km/s.
        - t_final in seconds
        - dt in seconds
        
        """

        # Time
        self.t_final = t_final
        self.dt= dt
        self.N = int(np.ceil(self.t_final/self.dt))
        self.t_arr = np.arange(self.N + 1)*self.dt

        # Arrays
        self.state_dim = len(state0)
        self.state_truth = np.zeros((self.N, self.state_dim))

        # Initial Conditions
        self.step = 0
        self.state_truth[0] = state0

        # Solar system bodies
        self.bodies = bodies

    def simulate():

        # initiate solver
        # self.solver = ode(self.diffy_q)
        # self.solver.set_integrator(propagator)
        # self.solver.set_initial_value(self.y[0,:],0)
        pass