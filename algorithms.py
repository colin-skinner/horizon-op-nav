import numpy as np
from numpy.linalg import norm, inv, lstsq

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
            T_p_c: Rotation matrix from celestial (C) to camera (P)
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
        T_c_p = T_p_c.T
        # T_c_p = inv(T_p_c)

        # Step 10
        r_prime = n / np.sqrt(np.dot(n, n) - 1)

        # Step 11
        r_c = T_c_p @ D_inv @ r_prime

        return r_c

if __name__ == "__main__":
    cr = ChristianRobinson(np.eye(3), 1, 1, 1)

    u = np.array([
        [1,2,3],
        [1,2,3.5],
        [1.2,2.5, 4.5]
    ])
    T_p_c = np.eye(3)

    r_c = cr.run(u, T_p_c)

    print(r_c)




