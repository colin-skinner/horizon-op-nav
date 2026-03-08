# import numpy as np
from Constants import G

print("PlanetaryData: Make sure to update position of bodies with position at specific time")
print('e.g. `Earth["position"] = [1.0, 2.0, 3.0]  # km`')

Sun = {
    "name": "Sun",
    "mass": 1.989e30,       # kg
    "mu": 1.989e30 * G,     # km^3/s^2
    "radius": 695510.0,     # km
    "position": None,
}

Earth = {
    "name": "Earth",
    "mass": 5.972e24,       # kg
    "mu": 5.972e24 * G,     # km^3/s^2
    "radius": 6378.0,       # km
    "position": None,
}

Luna = {
    "name": "Moon",
    "mass": 7.34767309e22,  # kg
    "mu": 7.34767309e22 * G,# km^3/s^2
    "radius": 1737.1,       # km
    "position": None,
}