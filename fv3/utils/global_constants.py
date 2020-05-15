RADIUS = 6.3712e6  # Radius of the Earth [m] #6371.0e3
PI = 3.1415926535897931  #  3.14159265358979323846
OMEGA = 7.2921e-5  # Rotation of the earth  # 7.292e-5
GRAV = 9.80665  # Acceleration due to gravity [m/s^2]
RDGAS = 287.05  # Gas constant for dry air [J/kg/deg] # 287.04
RVGAS = 461.50  # Gas constant for water vapor [J/kg/deg]
HLV = 2.5e6  # Latent heat of evaporation [J/kg]
HLF = 3.3358e5  # Latent heat of fusion [J/kg]  # 3.34e5
CP_AIR = 1004.6  # Specific heat capacity of dry air at constant pressure [J/kg/deg] # RDGAS / KAPPA
KAPPA = RDGAS / CP_AIR  # 2.0 / 7.0
DZ_MIN = 2.0
CV_AIR = CP_AIR - RDGAS
RDG = -RDGAS / GRAV
CNST_0P20 = 0.2
TFREEZE = 273.15  # Freezing temperature of fresh water [K]
