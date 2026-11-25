# Check CUDA setup.
# Modified from draft written by QwQ-32B.

import subprocess
import platform

from .. import __version__

print(f"Raven-check-cuda version {__version__}")

def print_error(msg):
    print(f"\033[91m[ERROR] {msg}\033[0m")

def print_success(msg):
    print(f"\033[92m[SUCCESS] {msg}\033[0m")

def print_note(msg):
    print(f"\033[93m[NOTE] {msg}\033[0m")

def is_command_available(command):
    try:
        subprocess.check_output([command, "--version"])
        return True
    except Exception:
        return False

def main():
    print("Checking dependencies...")

    # PyTorch check
    print("1. PyTorch availability check", end=" ")
    try:
        import torch
        print_success("✅")
    except ImportError:
        print_error("❌")
        print("   Install PyTorch first with: pdm update")
        return

    # A ROCm build of torch presents an AMD GPU through the `torch.cuda` API, and sets `torch.version.hip`.
    is_rocm = torch.version.hip is not None

    print("2. CUDA device availability check", end=" ")
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name()
        backend = " via ROCm" if is_rocm else ""
        print_success(f"✅ (Using {device}{backend})")

        if is_rocm:
            # e.g. "gfx1151", possibly followed by feature flags such as ":sramecc+:xnack-"
            arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            print(f"   GPU architecture {arch}", end=" ")
            if arch in torch.cuda.get_arch_list():
                print_success("✅ (supported by this PyTorch build)")
            else:
                print_error("❌ (not in this PyTorch build)")
                print(f"   This build supports: {', '.join(torch.cuda.get_arch_list())}")
                print("   A different ROCm version in pyproject.toml's `pytorch-gpu` source may support it.")

        print("3. CuPy & CuPyX (for spaCy NLP)", end=" ")
        try:
            import cupy  # noqa: F401, only for checking.
            import cupyx  # noqa: F401, only for checking.
            print_success("✅")
        except ImportError:
            if is_rocm:
                print_note("not installed")
                print("   On an AMD GPU, spaCy runs on the CPU.")
            else:
                print_error("❌")
                print("   Install CuPy with: pdm install -G cuda")

    else:
        print_error("❌")
        print("   Your system does not have accessible GPU resources")

        print("Troubleshooting:")
        driver_ok = False
        if is_command_available("nvidia-smi"):
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
                driver_ok = "driver_version" in output.decode()
            except Exception:
                driver_ok = False
        print("GPU driver ok: ", end=" ")
        if driver_ok:
            print_success("✅")
        else:
            print_error("❌")

        print("Suggested actions:")
        print("1. Update NVIDIA drivers.")

        # Windows specific checks
        if platform.system() == "Windows":
            print("   Windows users:")
            print("      Follow NVIDIA's guide for driver installation:")
            print("      https://www.nvidia.com/Download/index.aspx")
        else:
            print("   Linux users:")
            print("      On Linux Mint, use the Driver Manager, which can be found in the start menu.")
            print()
            print("      On other Ubuntu- or Debian-based distributions, this may be something like:")
            print("          sudo apt install nvidia-driver-555")
            print()
            print("      If unsure, check the instructions for your particular Linux distribution.")
        print("   Reboot the system after installing or updating drivers.")
        print("2. Verify paths to GPU libraries:")
        print("      source env.sh")

    # Display additional info
    print("\nSystem information:")
    print(f"   Python version: {platform.python_version()}")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   PyTorch version: {torch.__version__}")
    if is_rocm:
        print(f"   HIP version: {torch.version.hip}")

if __name__ == "__main__":
    main()
