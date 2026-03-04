import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import ode
from time import time as time
# from multiprocessing import Pool
# import spiceypy as spice
from numpy.linalg import norm

from PlanetaryData import Body

class OrbitPropagator:
    def __init__(self, state0: np.ndarray | list, t_final: float, dt: float, bodies: list[Body]):
        """
        - State in km, km/s.
        - t_final in seconds
        - dt in seconds
        - Bodies
        """

        # Time
        self.t_final = t_final
        self.dt= dt
        self.N = int(np.ceil(self.t_final/self.dt))
        self.ts = np.arange(self.N + 1)*self.dt

        # Arrays
        self.state_dim = len(state0)
        self.states = np.zeros((self.N + 1, self.state_dim))

        # Initial Conditions
        self.step = 0
        self.t = 0
        self.states[0] = state0

        # Solar system bodies
        self.bodies = bodies

    def simulate(self, state_print = False):

        # initiate solver
        self.solver = ode(self.diff)
        self.solver.set_integrator("lsoda")
        self.solver.set_initial_value(self.states[0], 0)

        while self.solver.successful() and self.step<self.N:
            # integrate step
            self.solver.integrate(self.solver.t+self.dt)
            self.step+=1

            # Set arrays
            self.ts[self.step]=self.solver.t
            self.states[self.step]=self.solver.y

            if state_print:
                print(self.states[self.step])


        
        # extract arrays at the step where propogation stopped
        s = self.step
        self.ts = self.ts[0:s]
        self.rs=self.states[0:s,0:3]
        self.vs=self.states[0:s,3:6]

        self.stop_step = self.step

    def diff(self, t, state):

        r = state[0:3]
        v = state[3:6]

        # Grav accel
        for body in self.bodies:
            r_body = r - body.position # Relative position
            a = -r_body*body.mu/norm(r_body)**3 # 3D vector with [km/s^2]

        # SRP?

        # Thrust?

        return np.array([*v,*a])

