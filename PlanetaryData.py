from Constants import G
class Body:
    def __init__(self, name: str, mass: float, mu: float, radius: float):
        self.name = name
        self.mass = mass
        self.mu = mu
        self.radius = radius

Sun = Body(
    name = "Sun",
    mass = 1.989e30, # kg
    mu = 1.989e30*G, # km3/s2
    radius = 695510.0 # km
)

Earth = Body(
    name = "Sun",
    mass = 5.972e24, # kg
    mu = 5.972e24*G, # km3/s2
    radius = 6378.0 # km
)

Luna = Body(
    name = "THE Moon",
    mass = 7.34767309e22, # kg
    mu = 7.34767309e22*G, # km3/s2
    radius = 1737.1 # km
)