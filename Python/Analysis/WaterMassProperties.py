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
# function pea - returns the vertically integrated potential energy anomaly
# function wmd - returns a 2D histogram of the salinity and temperature 
#                distribution weighted by volume
# function tad - returns the vertical distribution of the Turner angle in
#                the form of quantiles interpolated to fixed depths


def fwh(salt, z_w, saltref=35.0, maxdepth=-10.0):

    """ 
    This function returns the freshwater height, which is defined as
    the integral from a given depth below the surface to the surface, 
    using the integrand
        
        max(S_ref - S,0)/S_ref
                
    where S_ref is a reference salinity value, typically chosen as a value
    representative of open ocean conditions where there is little influence
    from riverine forcing. The freshwater height can be interpreted as 
    the volume of freshwater per m^2 that must be added to a water mass 
    with salinity S_ref to obtain the observed or modeled salinity profile.

    For example, if S_ref = 35.0, and the freshwater height between z=(-10,0)
    is 5 m, then the total amount of salt is equivalent to having S = 35.0 
    in the bottom 5 m, and S = 0.0 in the top 5 m. In reduced gravity models, 
    the potential energy anomaly can be shown to be proportional to the geostrophic 
    surface current stream function, see Gustafsson (Cont. Shelf Res., 19(8), 1999).  
    
    Keep in mind that maps based on the outputs from this function might appear
    strange if the chosen maximum depth is larger than the minimum depth of
    the model, hence it is best to mask the map where the local depth is smaller
    than the maximum depth.
    
    Usage:
    
        fresh_water_height = fwh(salt, z_w, saltref=35.0, maxdepth=-10.0)
        
    Inputs:

        salt                - 4D salinity field from ROMS (ndarray [T,Z,Y,X])
        z_w                 - 4D depth values of w-points (ndarray [T,Z,Y,X])
        saltref             - reference salinity value, default = 35.0
        maxdepth            - thickness of layer, default = -10.0
                              time dependent zeta is taken into account

    Output:

        fresh_water_height  - 3D freshwater height in [m] (ndarray [T,Y,X])
                              
    2025-04-15, kaihc@met.no
    """

    # Truncate z vector, keeping in mind that the surface coordinate is time dependent
    z = np.where(z_w < maxdepth + z_w[:,-1:,:,:], maxdepth + z_w[:,-1:,:,:], z_w)

    # Calculate dz
    dz = np.diff(z, axis=1)

    # Calculate integrand
    S = np.max(saltref-salt,0)/saltref

    # Integrate and return
    return np.sum(S*dz, axis=1)


def pea(rho, z_w, rhoref=1027.0, maxdepth=-10.0):

    """ 
    This function returns the potential energy anomaly, which is defined as
    the integral from a given depth below the surface to the surface, using 
    the integrand
        
        -g*max(rhoref - rho,0)*z/rhoref
                
    where rhoref is a reference density value, typically chosen as a value
    representative of open ocean conditions where there is little influence
    from riverine forcing. In reduced gravity models, the potential energy 
    anomaly can be shown to be proportional to the geostrophic transport 
    stream function, see Gustafsson (Cont. Shelf Res., 19(8), 1999).  
    
    The reference density should be such that there is some reference waters 
    in the area of interest. Keep in mind that maps based on the output from 
    this function might appear strange if the chosen maximum depth is larger 
    than the minimum depth of the model, hence it is best to mask the map where 
    the local depth is smaller than the maximum depth.
    
    Usage:
    
        potential_energy_anomaly = pea(rho, z_w, rhoref=1027.0, maxdepth=-10.0)
        
    Input variables:

        rho                         - 4D density field from ROMS (ndarray [T,Z,Y,X])
        z_w                         - 4D depth values of w-points (ndarray [T,Z,Y,X])
        rhoref                      - reference density value, default = 1027.0
        maxdepth                    - thickness of layer, default = -10.0
                                      time dependent zeta is taken into account

    Output:

        potential_energy_anomaly    -  [m^3 s^-2] (ndarray [T,Y,X])
    
    2025-04-23, kaihc@met.no
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
    ocean! The outputs of this function can be plotted using 
    e.g. contour/contourf.  

    Note that the volumetric fraction returned from this function
    is transposed such that a contour plot can be made as
    
    ----
    [X]: x, y, vol_frac = wmd(...)
    [X]: contour(x,y,vol_frac)
    ----

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

    2025-04-15, kaihc@met.no
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

    # Return distribution and coordinates. 
    return X, Y, volume_fraction.T


def tad(salt, temp, z_rho, maxdepth=-100.0, resolution=100):

    """
    This function returns the Turner angle distribution for the 
    salinity and temperature values supplied. The Turner angle
    is a function of the thermal expansion coefficient and the 
    haline contraction coefficient in addition to the vertical 
    gradients of salinity and temperature. It is a more useful 
    alternative to the density ratio for assessing the relative 
    influence of salinity and temperature for the stratification.

    Negative values indicate that the water column is salinity
    stratified, while positive values indicate that it is temperature
    stratified. For absolute values in the range (45,90) degrees, we may
    have double diffusion (salt fingering for [45,90] and diffusive 
    convection for [-90,-45]), and for absolute values larger than 
    90 degrees, the water column is unconditionally unstable.

    Example code for plotting the output:

    ----
    plt.subplot(1,1,1)

    plt.axvspan(-120, -90, facecolor=[1.0, 0.9, 0.9])
    plt.axvspan(-90, -45, facecolor='0.95')
    plt.axvspan(45, 90, facecolor='0.95')
    plt.axvspan(90, 120, facecolor=[1.0, 0.9, 0.9])

    plt.fill_betweenx(z, tu[:,0], tu[:,4], color=[0.8, 0.8, 1.0]) 
    plt.fill_betweenx(z, tu[:,1], tu[:,3], color=[0.5, 0.5, 0.7]) 
    plt.plot(tu[:,2], z, color=[0.2, 0.2, 0.5], linewidth = 2)

    plt.plot([0,0],[maxdepth, 0],'k-')
    plt.plot([-45,-45],[maxdepth, 0], color='0.3', linestyle=':')
    plt.plot([45,45],[maxdepth, 0], color='0.3', linestyle=':')
    plt.plot([-90,-90],[maxdepth, 0],'r:')
    plt.plot([90,90],[maxdepth, 0],'r:')

    plt.xlim((-120, 120))
    plt.ylim((maxdepth, -1))
    plt.xticks([-90, -45, 0, 45, 90])

    plt.xlabel('Turner angle [degrees]')
    plt.ylabel('Depth [m]')
    ----

    Usage:
    
        turner_angle_distribution, zout = tad(salt, temp, z_rho)
        
    Input variables:

        salt                        - 4D salinity field from ROMS (ndarray [T,Z,Y,X])
        temp                        - 4D temperature field from ROMS (ndarray [T,Z,Y,X])
        z_rho                       - 4D depth values of rho-points (ndarray [T,Z,Y,X])
        maxdepth                    - maximum depth for return values
        resolution                  - number of fixed depths

    Output:

        turner_angle_distribution  - median and percentiles [5, 25, median, 75, 95] Turner angle values [deg]
        zout                       - corresponding ndarray with predetermined depth values [m]   

    2025-04-23, kaihc@met.no        
    """

    import gsw
    from scipy.interpolate import interp1d

    # Shift zeta to zero everywhere
    for i in range(z_rho.shape[1]):
        z_rho[:,i,:,:] -= z_rho[:,-1,:,:]

    # Reshape arrays to [Z, (T * Y * X)]
    Ncolumns = int(np.size(z_rho)/z_rho.shape[1])
    Nlevels = z_rho.shape[1]

    z_rho = np.transpose(z_rho, (1,0,2,3)).reshape((Nlevels, Ncolumns))
    salt = np.transpose(salt, (1,0,2,3)).reshape((Nlevels, Ncolumns))
    temp = np.transpose(temp, (1,0,2,3)).reshape((Nlevels, Ncolumns))

    # Remove columns on land and reverse layers so that surface comes first
    not_nan_ind = np.where(~np.isnan(z_rho[0,:]))[0]
    z_rho = z_rho[::-1, not_nan_ind]
    salt = salt[::-1, not_nan_ind]
    temp = temp[::-1, not_nan_ind]

    # Convert salinity and temperature from ROMS to TEOS-10 standards
    p = gsw.conversions.p_from_z(z_rho, 60.0)
    SA = gsw.conversions.SA_from_SP(salt, p, 0.0, 60.0)
    CT = gsw.conversions.CT_from_pt(SA, temp)

    # Get Turner angle
    Tu, Rs, pmid = gsw.Turner_Rsubrho(SA,CT,p)
    
    # Get z coordinates
    z = gsw.z_from_p(pmid, 60.0)

    # Interpolate to fixed depths
    zout = np.linspace(maxdepth,-1,resolution)
    Tu_interp = np.zeros((zout.shape[0], Tu.shape[1]))
    for i in range(Tu.shape[1]):
        f = interp1d(z[:,i], Tu[:,i], bounds_error=False, fill_value=np.nan)
        Tu_interp[:,i] = f(zout)

    # Loop over levels and compute percentiles
    Tu_out = np.zeros((zout.shape[0],5))

    for i in range(len(zout)):
        tu_vals = Tu_interp[i,:].ravel()
        Tu_out[i,:] = np.nanquantile(tu_vals, [0.05,0.25,0.50,0.75,0.95])


    return Tu_out, zout

