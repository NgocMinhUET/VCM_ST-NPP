#!/usr/bin/env python
"""
Script to fix TensorBoard and TensorFlow compatibility issues.

This script:
1. Checks installed versions of TensorFlow and TensorBoard
2. Ensures they are compatible
3. Provides options to uninstall or upgrade as needed
"""

import subprocess
import sys
import re
from typing import Optional, Tuple

def get_installed_version(package: str) -> Optional[str]:
    """Get the installed version of a package."""
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "show", package],
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
        
        # Extract version using regex
        match = re.search(r"Version: (\S+)", output)
        if match:
            return match.group(1)
        return None
    except subprocess.CalledProcessError:
        return None

def uninstall_package(package: str) -> bool:
    """Uninstall a package."""
    try:
        print(f"Uninstalling {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error uninstalling {package}: {e}")
        return False

def install_package(package: str, version: Optional[str] = None) -> bool:
    """Install a package with optional version specification."""
    try:
        package_spec = f"{package}=={version}" if version else package
        print(f"Installing {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_spec}: {e}")
        return False

def run_pip_check() -> bool:
    """Run pip check to verify environment consistency."""
    try:
        print("Running pip check...")
        output = subprocess.check_output([sys.executable, "-m", "pip", "check"],
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
        print(output)
        return "No broken requirements found" in output
    except subprocess.CalledProcessError as e:
        print(f"pip check found issues: {e.output}")
        return False

def fix_tensor_deps() -> bool:
    """Main function to fix TensorFlow and TensorBoard compatibility."""
    tf_version = get_installed_version("tensorflow")
    tb_version = get_installed_version("tensorboard")
    
    print("\n=== Current Versions ===")
    print(f"TensorFlow: {tf_version or 'Not installed'}")
    print(f"TensorBoard: {tb_version or 'Not installed'}")
    
    # Case 1: Neither is installed
    if not tf_version and not tb_version:
        print("\nNeither TensorFlow nor TensorBoard is installed.")
        choice = input("Do you want to install TensorFlow and TensorBoard? (y/n): ").strip().lower()
        if choice == 'y':
            install_package("tensorflow")
            # TensorBoard will be installed as a dependency
            return run_pip_check()
        else:
            print("Skipping installation.")
            return True
    
    # Case 2: TensorFlow is installed but TensorBoard is not
    if tf_version and not tb_version:
        print("\nTensorFlow is installed but TensorBoard is not.")
        # Get major.minor version
        tf_major_minor = '.'.join(tf_version.split('.')[:2])
        compatible_tb = f"{tf_major_minor}.0"
        install_package("tensorboard", compatible_tb)
        return run_pip_check()
    
    # Case 3: TensorBoard is installed but TensorFlow is not
    if not tf_version and tb_version:
        print("\nTensorBoard is installed but TensorFlow is not.")
        choice = input("Do you want to install compatible TensorFlow or uninstall TensorBoard? (install/uninstall/skip): ").strip().lower()
        if choice == 'install':
            # Get major.minor version
            tb_major_minor = '.'.join(tb_version.split('.')[:2])
            compatible_tf = f"{tb_major_minor}.0"
            install_package("tensorflow", compatible_tf)
        elif choice == 'uninstall':
            uninstall_package("tensorboard")
        else:
            print("Skipping action.")
        return run_pip_check()
    
    # Case 4: Both are installed but may be incompatible
    if tf_version and tb_version:
        tf_major_minor = '.'.join(tf_version.split('.')[:2])
        tb_major_minor = '.'.join(tb_version.split('.')[:2])
        
        if tf_major_minor == tb_major_minor:
            print(f"\nTensorFlow and TensorBoard versions appear compatible ({tf_major_minor}.x).")
            return run_pip_check()
        else:
            print(f"\nVersion mismatch: TensorFlow {tf_version} and TensorBoard {tb_version}")
            choice = input("Do you want to make them compatible? (y/n): ").strip().lower()
            if choice == 'y':
                print("Options:")
                print(f"1. Upgrade TensorBoard to {tf_major_minor}.0")
                print(f"2. Upgrade TensorFlow to {tb_major_minor}.0")
                print("3. Uninstall both and reinstall")
                print("4. Uninstall both and skip reinstall")
                
                option = input("Choose option (1-4): ").strip()
                if option == '1':
                    uninstall_package("tensorboard")
                    install_package("tensorboard", f"{tf_major_minor}.0")
                elif option == '2':
                    uninstall_package("tensorflow")
                    install_package("tensorflow", f"{tb_major_minor}.0")
                elif option == '3':
                    uninstall_package("tensorflow")
                    uninstall_package("tensorboard")
                    install_package("tensorflow")  # TensorBoard will be installed as a dependency
                elif option == '4':
                    uninstall_package("tensorflow")
                    uninstall_package("tensorboard")
                else:
                    print("Invalid option selected.")
            else:
                print("Skipping compatibility fix.")
            
            return run_pip_check()
    
    return True

if __name__ == "__main__":
    print("=== TensorFlow and TensorBoard Compatibility Fixer ===")
    if fix_tensor_deps():
        print("\nEnvironment check passed!")
    else:
        print("\nEnvironment check failed. Please resolve remaining issues manually.") 