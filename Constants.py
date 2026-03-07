import numpy as np

G_meters = 6.67408e-11 # m^3/kg/s^2
G = G_meters * 10**-9 # km^3/kg/s^2
DAY_TO_SEC = 24*3600.0 # sec/day
SEC_TO_DAY = 1 / DAY_TO_SEC # sec/day

RAD_TO_DEG = 180/np.pi
DEG_TO_RAD = 1 / RAD_TO_DEG
ARCSEC_TO_RAD = DEG_TO_RAD / 3600
RAD_TO_ARCSEC = 1 / ARCSEC_TO_RAD