import numpy as np


# Water mass properties from ROMS output
#
# CAUTION: these functions assume that the input is taken directly 
# from ROMS, with no post processing with e.g. interpolation to fixed 
# depth values.
#
# Contents:
# 
# function fwh - returns the freshwater height
# function ppe - returns the vertically integrated density anomaly
# function wmd - returns a 2D histogram of the salinity and temperature 
#                distribution weighted by volume
# function tad - returns the vertical distribution of the Turner angle 
# function tad2D - returns a map of the Turner angle in the upper ocean


def fwh(salt, z_w, saltref=35.0, maxdepth=-10.0):

    """ 
    This function returns the freshwater height, which is defined as
    the integral from a given depth to the surface, using the integrand
        
        max(S_ref - S,0)/S_ref
                
    where S_ref is a reference salinity value, typically chosen as a value
    representative of open ocean conditions where there is little influence
    from riverine forcing. The freshwater height can be interpreted as 
    the volume of freshwater per m^2 that must be added to a water mass 
    with salinity S_ref to obtain the observed or modeled salinity profile.

    For example, if S_ref = 35.0, and the freshwater height between z=(-10,0)
    is 5 m, then the total amount of salt is equivalent to having S = 35.0 
    in the bottom 5 m, and S = 0.0 in the top 5 m. 
    
    Keep in mind that maps based on the output from this function might appear
    strange if the chosen maximum depth is larger than the minimum depth of
    the model, hence it is best to mask the map where the local depth is smaller
    than the maximum depth.
    
    2025-04-15, kaihc@met.no
    
    Usage:
    
        fresh_water_height = fwh(salt, z_w, saltref=35.0, maxdepth=-10.0)
        
    Variables:

        fresh_water_height  - 3D freshwater height in [m] (ndarray [T,Y,X])
        salt                - 4D salinity field from ROMS (ndarray [T,Z,Y,X])
        z_w                 - 4D depth values of w-points (ndarray [T,Z,Y,X])
        saltref             - reference salinity value, default = 35.0
        maxdepth            - thickness of layer, default = -10.0
                              time dependent zeta is taken into account
    """

    # Truncate z vector, keeping in mind that the surface coordinate is time dependent
    z = np.where(z_w < maxdepth + z_w[:,-1:,:,:], maxdepth + z_w[:,-1:,:,:], z_w)

    # Calculate dz
    dz = np.diff(z, axis=1)

    # Calculate integrand
    S = np.max(saltref-salt,0)/saltref

    # Integrate and return
    return np.sum(S*dz, axis=1)


def ppe(rho, z_w, rhoref=1027.0, maxdepth=-10.0):

    """ 
    This function returns the profile potential energy, which is defined as
    the integral from a given depth to the surface, using the integrand
        
        -g*max(rhoref - rho,0)*z/rhoref
                
    where rhoref is a reference density value, typically chosen as a value
    representative of open ocean conditions where there is little influence
    from riverine forcing. 
    
    To quote Gustafsson (1999, Cont. Shelf Res., 19), "the profile potential 
    energy is the excess potential energy built up in a stratified water column 
    as compared to one with uniform density". Strictly speaking, a stratified
    water column has _lower_ potential energy, but view this integrated parameter
    as a measure of the amount of stratification. 
    
    An interpretation is to look at the value as expressing potential energy 
    that can be released if the water column is adjacent to a water column with 
    uniform density = rhoref. The profile potential energy can be shown to be 
    proportional to the geostrophic transport stream-function, see Gustafsson 
    (1999) for details.  
    
    The reference density should be such that there is some reference waters 
    in the area of interest. Keep in mind that maps based on the output from 
    this function might appear strange if the chosen maximum depth is larger 
    than the minimum depth of the model, hence it is best to mask the map where 
    the local depth is smaller than the maximum depth.
    
    2025-04-15, kaihc@met.no
    
    Usage:
    
        profile_potential_energy = ppe(rho, z_w, rhoref=1027.0, maxdepth=-10.0)
        
    Input variables:

        rho                         - 4D density field from ROMS (ndarray [T,Z,Y,X])
        z_w                         - 4D depth values of w-points (ndarray [T,Z,Y,X])
        rhoref                      - reference density value, default = 1027.0
        maxdepth                    - thickness of layer, default = -10.0
                                      time dependent zeta is taken into account

    Output:

        profile_potential_energy    -  [m^3 s^-2] (ndarray [T,Y,X])
    
    """

    # Set constants
    g = 9.81

    # Truncate z vector, keeping in mind that the surface coordinate is time dependent
    z = np.where(z_w < maxdepth + z_w[:,-1:,:,:], maxdepth + z_w[:,-1:,:,:], z_w)

    # Calculate dz
    dz = np.diff(z, axis=1)

    # Calculate z values
    z_mid = (z[:,:-1,:,:] + z[:,1:,:,:])/2

    # Calculate integrand
    P = -g*np.max(rhoref-rho,0)*z_mid/rhoref

    # Integrate and return
    return np.sum(P*dz, axis=1)


def wmd(salt, temp, volume, bins_salt=50, bins_temp=50):

    """
    This function returns the 2D distribution of temperature
    and salinity weighted by volume. When visualised, the result 
    is typically very different from the impression you get
    when plotting TS diagrams based on grid points, since the 
    variable upper ocean is disproportionally represented in terms
    of number of grid points. The large volumes are in the deep 
    ocean! The outputs of this function can be conveniently 
    plotted using e.g. contour/contourf.  

    2025-04-15, kaihc@met.no

    Usage:

        x, y, vol_frac = wmd(salt, temp, volume, bins_salt=50, bins_temp=50)

    Input variables:

        salt        - ndarray with salinity
        temp        - ndarray with temperature
        volume      - cell volume [m^3] (same dims as the above variables)
        bins_salt   - number of bins for salinity (integer)
        bins_temp   - number of bins for temperature (integer)

    Output variables:

        x, y        - meshgrid variables (salt, temp) for plotting
        vol_frac    - volumetric fraction per bin [km^3]
    """

    # Remove any NaNs
    temp = temp[~np.isnan(temp)]
    salt = salt[~np.isnan(salt)]
    volume = volume[~np.isnan(volume)]

    # Change from m3 to km3
    volume = volume/1.0e9

    # Get 2D histogram
    volume_fraction, x_edges, y_edges = np.histogram2d(salt, temp, 
                                                       bins=[bins_salt,bins_temp], weights=volume)
    
    # Calculate coordinates for plotting
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Return distribution and coordinates. Note that 
    # the volumetric fraction is transposed such that 
    # a contour plot can be made as
    #
    # contour(x,y,vol_frac)
    #
    # if this function is called as
    #
    # x, y, vol_frac = wmd(...)
    #
    return X, Y, volume_fraction.T


def tad():

    return True

def tad2D():

    return True

