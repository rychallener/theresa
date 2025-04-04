import fitclass as fc
import utils
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from jaxoplanet import units
from jaxoplanet.starry import Surface, Ylm
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.orbits.keplerian import Central, Body
from jaxoplanet.units import unit_registry as ureg

# Create and configure the fit object

def testLon():
    fit1 = fc.Fit()
    fit1.read_config("/home/abrar/wasp76-example.cfg")

    # Initialize the system with JAXoplanet
    central, central_surface, body, body_surface, system = utils.initsystem(fit1, 1)
    # Create observation times for testing
    times = np.linspace(.4*body_surface.period.magnitude, .6*body_surface.period.magnitude, 20)  # 4 days of observations

    # Simple data container class to hold observation times
    class DataContainer:
        def __init__(self, t):
            self.t = t

    # Create data object with observation times
    data = DataContainer(times)

    # Test the JAXoplanet vislon function
    min_lon, max_lon = utils.vislon(body_surface, data)

    print(f"Minimum visible longitude: {min_lon}")
    print(f"Maximum visible longitude: {max_lon}")

    # Optional: Plot the visible longitude range over time
    try:
        
        # Calculate visible longitude range at each time point
        visible_ranges = []
        
        for t in times:
            data_single = DataContainer(np.array([t]))
            # Get relevant parameters
            prot = body_surface.period.magnitude  # days / rotation

            phase = body_surface.phase  # Initial phase in radians
            theta0 = 180
            t0 = 0
           
            
            # Calculate central longitude 
            centlon = theta0 - (data_single.t - t0) / prot * 360
            
          
            
            # Calculate visible longitude range
            limb1 = centlon - 90
            limb2 = centlon + 90
           
            # Normalize limbs to [-180, 180] range
            limb1 = (limb1 + 180) % 360 - 180
            limb2 = (limb2 + 180) % 360 - 180
            
            # For debugging
            print(f"Time: {t:.2f}, Central longitude: {float(centlon[0]):.1f}°, Visible: [{float(limb1[0]):.1f}° to {float(limb2[0]):.1f}°]")
            
            visible_ranges.append((min(float(limb1[0]),float(limb2[0])), max(float(limb1[0]),float(limb2[0]))))
        
        # Extract min and max longitudes at each time
        visible_min = [r[0] for r in visible_ranges]
        visible_max = [r[1] for r in visible_ranges]
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(times, visible_min, 'b-', label='Minimum Visible Longitude')
        plt.plot(times, visible_max, 'r-', label='Maximum Visible Longitude')
        plt.fill_between(times, visible_min, visible_max, alpha=0.3, color='gray', label='Visible Range')
        plt.axhline(-180, color='k', linestyle='--', alpha=0.5)
        plt.axhline(180, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Time (days)')
        plt.ylabel('Longitude (degrees)')
        plt.title('Visible Longitude Range over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-190, 190)
        plt.savefig('visible_longitude_range.png')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")



#testLon()

