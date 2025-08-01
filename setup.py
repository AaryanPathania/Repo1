#!/usr/bin/env python3
"""
Setup script for AD Prediction Demo
===================================

This script sets up the environment and installs all necessary dependencies
for running the Alzheimer's Disease prediction models.
"""

import subprocess
import sys
import os
import platform

def run_command(command, check=True):
    """Run a shell command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error message: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("ERROR: Python 3.7 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_pytorch():
    """Install PyTorch with appropriate version for the system"""
    print("Installing PyTorch...")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ PyTorch with CUDA support already installed")
            return True
    except ImportError:
        pass
    
    # Install PyTorch
    system = platform.system().lower()
    
    if system == "linux":
        # For Linux, try to install CUDA version if available
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        # For other systems, install CPU version
        torch_cmd = "pip install torch torchvision torchaudio"
    
    if run_command(torch_cmd):
        print("✓ PyTorch installed successfully")
        return True
    else:
        # Fallback to CPU version
        print("Falling back to CPU-only PyTorch...")
        fallback_cmd = "pip install torch torchvision torchaudio"
        if run_command(fallback_cmd):
            print("✓ PyTorch (CPU) installed successfully")
            return True
        else:
            print("✗ Failed to install PyTorch")
            return False

def install_requirements():
    """Install other required packages"""
    print("Installing other requirements...")
    
    requirements = [
        "numpy>=1.21.0",
        "matplotlib>=3.4.0", 
        "scikit-image>=0.18.0",
        "nibabel>=3.2.0",
        "Pillow>=8.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        if not run_command(f"pip install {req}"):
            print(f"✗ Failed to install {req}")
            return False
    
    print("✓ All requirements installed successfully")
    return True

def test_installation():
    """Test if all packages can be imported"""
    print("Testing installation...")
    
    packages = [
        "torch", "torchvision", "numpy", "matplotlib", 
        "skimage", "nibabel", "PIL", "sklearn", "scipy", "tqdm"
    ]
    
    failed = []
    for package in packages:
        try:
            if package == "skimage":
                import skimage
            elif package == "PIL":
                import PIL
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check available memory (rough estimate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Available RAM: {memory_gb:.1f} GB")
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM available. Consider using smaller batch sizes.")
    except ImportError:
        print("Could not check memory (psutil not available)")
    
    # Check disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)
        print(f"Available disk space: {free_space:.1f} GB")
        if free_space < 2:
            print("⚠️  Warning: Less than 2GB disk space available.")
    except:
        print("Could not check disk space")
    
    return True

def create_demo_files():
    """Ensure demo files are present"""
    print("Checking demo files...")
    
    required_files = ["demo_data_generator.py", "ad_prediction_demo.py", "requirements.txt"]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} not found")
            return False
    
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("AD Prediction Demo Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Check if demo files exist
    if not create_demo_files():
        print("✗ Some required files are missing")
        sys.exit(1)
    
    # Install PyTorch
    if not install_pytorch():
        print("✗ Failed to install PyTorch")
        sys.exit(1)
    
    # Install other requirements
    if not install_requirements():
        print("✗ Failed to install requirements")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("✗ Installation test failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Setup completed successfully! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Generate demo data: python demo_data_generator.py")
    print("2. Run AlexNet demo: python ad_prediction_demo.py --mode alexnet --epochs 5")
    print("3. Run 3D CNN demo: python ad_prediction_demo.py --mode 3dcnn --epochs 3")
    print("4. Run both models: python ad_prediction_demo.py --mode both --epochs 5")
    print("\nFor help: python ad_prediction_demo.py --help")

if __name__ == "__main__":
    main()