import spiceypy as spice
from datetime import datetime
from pathlib import Path
import pytz

j2000_epoch_utc = datetime(2000, 1, 1, 11, 58, 55, 816000, tzinfo=pytz.utc)
class SpiceImporter:
    def __init__(self, kernel_dir="kernels"):
        self.kernel_dir = Path(kernel_dir)
        self.kernel_dir.mkdir(exist_ok=True)
        self.loaded_kernels = []

    def download_kernels(self):
        """Download necessary SPICE kernels from NAIF"""
        import urllib.request
        
        # Base URL for NAIF kernels
        base_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        
        kernels = {
            'lsk': 'lsk/naif0012.tls',                          # Leap seconds (hate this)
            'pck': 'pck/gm_de440.tpc',                          # Planetary constants
            'spk': 'spk/planets/de440.bsp',                     # DE440 (planet and moon) ephemeris (2020-2030)
        }

        print("Downloading SPICE kernels...")
        for kernel_path in kernels.values():
            url = base_url + kernel_path
            local_path = self.kernel_dir / Path(kernel_path).name
            
            if not local_path.exists():
                print(f"  Downloading {Path(kernel_path).name}...")
                try:
                    urllib.request.urlretrieve(url, local_path)
                except Exception as e:
                    print(f"  Warning: Could not download {kernel_path}: {e}")
            else:
                print(f"  {Path(kernel_path).name} already exists")
        
        print("Download complete!")

    def load_kernels(self):
        kernel_files = (list(self.kernel_dir.glob("*.tls"))
                      + list(self.kernel_dir.glob("*.tpc"))
                      + list(self.kernel_dir.glob("*.bsp"))
                      + list(self.kernel_dir.glob("*.bpc")))
        
        for kernel in kernel_files:
            try:
                spice.furnsh(str(kernel))
                self.loaded_kernels.append(str(kernel))
                print(f"Loaded: {kernel.name}")
            except Exception as e:
                print(f"Error loading {kernel.name}: {e}")
        
    
    def get_state(self, body_name: str, time: str | datetime, observer: str = 'EARTH') -> dict:
        """
        Get position and velocity of a body at a specific time.
        
        Args:
            body_name: Name of celestial body (e.g., 'MOON', 'EARTH')
            time: Time in UTC format (e.g., '2024-01-01 00:00:00')
            observer: Observer body (default: 'EARTH')
            
        Returns:
            Dictionary with name, time, position, and velocity
        """

        # So we can use datetime (for simple visualization stuff)
        if isinstance(time, datetime):
            et = (time.astimezone(pytz.utc) - j2000_epoch_utc).total_seconds()
        else:
            et = spice.str2et(time)
        state, _ = spice.spkpos(body_name, et, 'J2000', 'NONE', observer)
        
        return {
            'name': body_name,
            'time': time,
            'position': tuple(state[:3])
        }

if __name__ == "__main__":

    spicer = SpiceImporter()
    spicer.download_kernels()
    spicer.load_kernels()

    t = datetime(2026, 6, 13, 11, 25, 00, tzinfo=pytz.utc)

    # earth_data = spicer.get_state("MOON", "2026-06-13 23:25:00")
    earth_data = spicer.get_state(body_name="EARTH", time=t, observer="MOON")

    print(earth_data)