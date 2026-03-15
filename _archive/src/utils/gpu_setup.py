
import os
import sys
import platform

def setup_gpu_paths():
    """
    Adds NVIDIA library paths to the DLL search path on Windows.
    This is necessary when using pip-installed CUDA libs with TensorFlow < 2.11 on Windows.
    """
    if platform.system() != "Windows":
        return

    try:
        import site
        # We look for 'nvidia' in site-packages
        site_packages = [p for p in site.getsitepackages() if "site-packages" in p]
        
        nvidia_path = None
        for sp in site_packages:
            p = os.path.join(sp, "nvidia")
            if os.path.exists(p):
                nvidia_path = p
                break
        
        if not nvidia_path:
            # Fallback: try to find via import
            try:
                import nvidia
                if hasattr(nvidia, "__path__"):
                    nvidia_path = list(nvidia.__path__)[0]
            except ImportError:
                print("Warning: Could not import nvidia package.")
                return

        if not nvidia_path:
             print("Warning: Could not locate nvidia site-packages directory.")
             return

        # Scan all directories in nvidia folder
        dll_paths = []
        for item in os.listdir(nvidia_path):
            item_path = os.path.join(nvidia_path, item)
            if os.path.isdir(item_path):
                bin_path = os.path.join(item_path, "bin")
                lib_path = os.path.join(item_path, "lib")
                
                if os.path.exists(bin_path):
                    dll_paths.append(bin_path)
                elif os.path.exists(item_path): # Check root if bin doesn't exist
                     # Some libs might be in root, but careful not to add junk
                     files = [f for f in os.listdir(item_path) if f.endswith(".dll")]
                     if files:
                         dll_paths.append(item_path)

        # Add to PATH and DLL directory
        for p in dll_paths:
            if os.path.exists(p) and p not in os.environ["PATH"]:
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(p)
                    except Exception as e:
                        print(f"Warning: Failed to add DLL directory {p}: {e}")
                print(f"Added GPU DLL path: {p}")
                
    except Exception as e:
        print(f"Error setting up GPU paths: {e}")

# Run on import
setup_gpu_paths()
