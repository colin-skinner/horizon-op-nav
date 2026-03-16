import numpy as np
from numpy.linalg import norm, inv, lstsq
from trajectory.Constants import G


class ChristianRobinson:

    def __init__(self, K_inv: np.ndarray, a: float, b: float, c: float):
        """
        K: camera calibration matrix

        a,b,c: ellipsoid parameters"""
        self.K_inv = K_inv
        self.a = a
        self.b = b
        self.c = c

    def run(self, u: np.ndarray, T_p_c: np.ndarray):
        """
        Parameters:
            u: set of measurements (Nx3) e.g. [[0,0,0], [1,1,1]]
            T_p_c: Rotation matrix from camera (C) to planet (P)
        Returns:
            r_c: Vector from CAMERA TO PLANET in CAMERA FRAME
        """
        N = len(u)
        a,b,c = self.a, self.b, self.c

        # Step 2 (eq. 98)
        D = np.diag([1/a, 1/b, 1/c])
        D_inv = np.diag([a, b, c])

        # Step 3 (eq. 102)
        R = D @ T_p_c @ self.K_inv

        # Step 4-6
        xs = np.zeros((N, 3))
        s = np.zeros((N, 3))

        for i in range(N):
            xs[i] = R @ u[i]
            s[i] = xs[i] / norm(xs[i])

        # Step 7
        assert xs.shape == (N,3)
        H = s # Already in this form I guess

        # Step 8
        one_arr = np.ones((N, 1))
        n = lstsq(H, one_arr)[0] # Could extract other metrics maybe?
        n = n.flatten()

        # Step 9
        T_c_p = T_p_c.T # From planet to camera

        # Step 10
        r_prime = n / np.sqrt(np.dot(n, n) - 1)

        # Step 11
        r_c = T_c_p @ D_inv @ r_prime

        return r_c

class SatEKF():
    """Child class for the Satellite EKF"""
    def __init__(self, Q: np.ndarray, R: np.ndarray, dt: float,
                 mu0: np.ndarray, sigma0: np.ndarray, central_body_mass: np.ndarray):
        """
        Initialize the EKF with the system matrices
        """
        self.Q = np.array(Q)
        self.R = np.array(R)

        self.mu = mu0
        self.sigma = sigma0

        self.dt = dt
        self.mu_cb = G * central_body_mass

    def f_func(self, mu: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """State Transition Function"""
        
        r = mu[0:3]
        v = mu[3:6]

        a = -r*self.mu_cb/norm(r)**3 # No input

        r_new = r + self.dt*v + 0.5*a*self.dt**2
        v_new = v + self.dt*a

        return np.array([*r_new, *v_new])
    

    

    def f_jac(self, mu: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """Jacobian of the State Transition Function"""

        r = mu[0:3]
        dt = self.dt
        mu_cb = self.mu_cb  # central body mu

        # a = -r*mu_cb/norm(r)**3
        dvdr = -mu_cb * (np.eye(3) / norm(r)**3 - 3 * np.outer(r, r) / norm(r)**5) # hand calcs for this
        
        F = np.eye(6)
        F[0:3, 3:6] = dt * np.eye(3)              # dr/dv
        F[0:3, 0:3] += 0.5 * dt**2 * dvdr  # dr/dr
        F[3:6, 0:3] = dt * dvdr            # dv/dr

        return F


    def g_func(self, mu: np.ndarray, T_c_p: np.ndarray) -> np.ndarray: # So simple
        """Measurement function

        Args:
            mu (np.ndarray): [x,y,z,vx,vy,vz]
            T_c_p (np.ndarray): Passive rot. from Planet to Camera frame

        Returns:
            mu_next (np.ndarray): [x,y,z,vx,vy,vz]
        """
        meas = T_c_p @ mu[0:3]
        print(f"Measurement function output: {meas}")
        return meas
    
    def phony_g_func(self, mu: np.ndarray, r_c: np.ndarray, T_c_p: np.ndarray) -> np.ndarray:
        """Measurement function that ignores the state and just returns the measurement (for testing)"""

        print(f"Phony measurement function output: {r_c}")

        return r_c
    
    def phony_g_jac(self, mu: np.ndarray, r_c: np.ndarray, T_c_p: np.ndarray) -> np.ndarray:
        """Jacobian of the phony measurement function (just identity)"""

        H = np.zeros((3,6))
        H[:,0:3] = np.eye(3)
        return H
    

    

    def g_jac(self, mu: np.ndarray, T_c_p: np.ndarray) -> np.ndarray: # So simple
        """Jacobian of the Measurement Function

        Args:
            mu (np.ndarray): [x,y,z,vx,vy,vz] (not really used though)
            T_c_p (np.ndarray): Passive rot. from Planet to Camera frame

        Returns:
            mu_next (np.ndarray): [x,y,z,vx,vy,vz]
        """
        H = np.zeros((3,6))
        H[:,0:3] = T_c_p
        return H
    
    def predict(self, u: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """EKF Predict Step (no input required though. kept in case we want to add)"""

        # Predict the next state (t|t-1) from the current state (t-1|t-1)
        a_mat = self.f_jac(self.mu, u)             # ✓ Evaluate at current state FIRST
        self.sigma = a_mat @ self.sigma @ a_mat.T + self.Q
        self.mu = self.f_func(self.mu, u)          # Then update state

        return self.mu, self.sigma
    
    def update(self, r_c: np.ndarray, T_c_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """EKF Update Step

        Args:
            r_c (np.ndarray): (Measurement) Output of CR algorithm (camera->planet in CAMERA frame)
            T_c_p (np.ndarray): Passive rot. from Planet to Camera frame

        Returns:
            mu_next (np.ndarray), Sigma_next (np.ndarray)
        """

        # Calculate the Kalman Gain
        c_mat = self.g_jac(self.mu, T_c_p)
        s_mat = c_mat @ self.sigma @ c_mat.T + self.R
        k_mat = self.sigma @ c_mat.T @ np.linalg.inv(s_mat)

        # Update the state estimate (t|t) from the predicted state (t|t-1)
        self.mu += k_mat @ (r_c - self.g_func(self.mu, T_c_p))
        # Update the covariance
        self.sigma = (np.eye(self.sigma.shape[0]) - k_mat @ c_mat) @ self.sigma

        return self.mu, self.sigma
    