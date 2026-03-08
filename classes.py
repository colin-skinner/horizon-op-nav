import numpy as np

from numpy.linalg import norm, det
from numpy import sin, cos, tan, sqrt, pi, arctan
from Constants import RAD_TO_DEG, DEG_TO_RAD, ARCSEC_TO_RAD, RAD_TO_ARCSEC
from utils import circle_points, noise

mu_unicode = "\u03bc"
deg_unicode = "\u00b0"

class Camera:
    def __init__(self,
        f: float,
        mu: float,
        resolution: float
    ):
        """Camera object

        Args:
            f (float): Focal length [m]
            mu (float): Pixel resolution [m/px]
            resolution (np.ndarray(2)): Total pixels (x, y) [px]
        """
        self.f = f
        self.mu = mu
        self.resolution = resolution

        self.mu_angle = mu/f
        self.fov = self.resolution * self.mu_angle

        self.are_specs_calc = False

    # mu_angle = fov/n_pixels[0]       # pixel angle resolution [rad/px]
    # mu = f * mu_angle             # pixel spacing [m/px]
    def calc_specs(self, alpha_skew: float = 0):

        dx = dy = self.f / self.mu
        u_p, v_p = self.resolution / 2

        self.K_inv = np.array([
            [1/dx, -alpha_skew/dx/dy, (alpha_skew*v_p - dy*u_p)/dx/dy],
            [0, 1/dy, -v_p/dy],
            [0, 0, 1]
        ])

        self.u_p = u_p
        self.v_p = v_p
        self.dx = dx
        self.dy = dy

        self.are_specs_calc = True
        

    def __str__(self):
        s = "Camera base specs:\n"
        s += f" - Pixel angular resolution: {self.mu_angle*RAD_TO_ARCSEC:.2f} arcs/px\n"
        s += f" - Pixel resolution ({mu_unicode}): {self.mu*1000000:.2f} {mu_unicode}m/px\n"
        s += f" - FOV: {self.fov*RAD_TO_DEG}{deg_unicode}\n"

        if self.are_specs_calc:
            s += " Calculated specs\n"
            s += f" - Center (u_p, v_p) = {np.array([self.u_p, self.v_p])} arcs/px\n"
            s += f" - dx, dy = {np.array([self.dx, self.dy])} (unitless?)\n"
            # np.set_printoptions(formatter={'all': lambda x: '\t' + str(x)})
            s += f" - K_inv = \n{self.K_inv}\n"
            # s += f" - Pixel resolution: {self.mu*1000000:.2f} {mu_unicode}m/px\n"
            # s += f" - FOV: {self.fov*RAD_TO_DEG}{deg_unicode}\n"

        return s

class Body:

    def __init__(self,
        a: float,
        b: float,
        c: float
    ):
        """
        Args:
            a (float): Ellipsoid parameter [m]
            b (float): Ellipsoid parameter [m]
            c (float): Ellipsoid parameter [m]
        """
        self.a = a
        self.b = b
        self.c = c

    def __str__(self):
        s = "Worldview:\n"
        s += f" - Ellipsoid semi-axes a,b,c = [{self.a}, {self.b}, {self.c}] m\n"
        return s
    
class Pose:
    def __init__(self,
        r_truth: np.ndarray,
#         a: float,
#         b: float,
#         c: float,
        T_p_c: np.ndarray
    ):
            
        """

        Args:
            r_truth (np.ndarray): Position relative to central body [m]
            T_p_c (np.ndarray(3,3)): Passive rotation from PLANET to CAMERA 
        """
        self.r_truth = r_truth
        self.T_p_c = T_p_c

    def __str__(self):
        s = "Pose:\n"
        s += f" - Spacecraft position (r_truth) = {self.r_truth} m\n"
        s += f" - Absolute position norm = {norm(self.r_truth):.2f} m\n"
        return s
    
class PlanetImage:
    def __init__(self, c: Camera, b: Body, sigma_cross_boresight_px: float = 0.5):
        """Init and generate planet points

        Args:
            c (Camera): Camera
            b (Body): Body specs with ellipsoid parameters
            sigma_cross_boresight_px (float, optional): Standard deviation [px]. Defaults to 0.5.
        """
        self.camera_center_px = np.array([c.u_p, c.v_p])
        self.camera_mu_angle = c.mu_angle
        self.res = c.resolution

        # Noise
        s = sigma_cross_boresight_px
        self.R = np.array([
            [s**2,0,0],
            [0,s**2,0],
            [0,0,0]
        ])

        # TODO: change if not circular, or if manually specifying center and radius
        self.dx = c.dx # no idea what this is really
        self.body_radius = b.a

        self.image_generated = False

    def gen_points(self, pose: Pose, N_RADIAL_POINTS: int = 100,
                 body_center_px: np.ndarray = None):
        """Generate planet points based on position

        Args:
            pose (Pose): Pose with position and orientation
            N_PLANET_POINTS (int): Number of planet points to generate 
            planet_center_px (np.ndarray(2), optional): Planet center [px]. Defaults to image center.
        """
        self.image_generated = True


        # Planet pixel locations in image 
        circular = False
        
        self.body_angle = np.arcsin(self.body_radius / norm(pose.r_truth)  )

        if circular: # TODO: change this for non circular or offset
            # self.body_angle = np.arcsin(self.body_radius / norm(pose.r_truth))
            
            self.body_center_px = (
                self.camera_center_px if body_center_px is None else body_center_px
            )

        else:
            # From 
            #   -   planet->camera in camera frame
            #   -   camera->planet in camera frame (for image generation)
            planet_vector_cam = pose.T_p_c @ -pose.r_truth 

            Xc, Yc, Zc = planet_vector_cam

            # projection: u = f * Xc / Zc / mu --> in pixels
            u = (Xc / Zc) / self.camera_mu_angle + self.camera_center_px[0]
            v = (Yc / Zc) / self.camera_mu_angle + self.camera_center_px[1]

            self.body_center_px = np.array([u, v])

        # self.body_radius_px = (
        #         self.body_angle / self.camera_mu_angle if body_radius_px is None else body_radius_px
        #     )
        self.body_radius_px = self.dx * tan(self.body_angle)

        # Generate points
        self.N = N_RADIAL_POINTS
        self.body_points = circle_points(self.body_radius_px, self.N, # RETURNS (2, N) ARRAY
                                    position=self.body_center_px) 
        


        self.u_points = np.hstack((self.body_points.T, np.ones((N_RADIAL_POINTS, 1))))
        self.u_noise = np.array([v + noise(self.R) for v in self.u_points])
        self.u_in_frame = np.array([v for v in self.u_noise 
                       if v[0] > 0 and v[0] < self.res[0]
                       and v[1] > 0 and v[1] < self.res[1]])
    

    def __str__(self):
        s = "PlanetImage:\n"
        s += f" - Noise covariance R [px^2]= \n{self.R}\n"
        s += f" - Planet center = {self.camera_center_px} px\n"
        
        if self.image_generated:
            s += " Current image specs:\n"
            s += f" - Planet angular radius (rad) = {self.body_angle:.6f} rad\n"
            s += f" - Planet radius = {self.body_radius_px:.2f} px\n"
            s += f" - Number of points = {self.N}\n"
            s += f" - First 3 noisy points = \n{self.u_noise[:3]} px\n"

        return s

