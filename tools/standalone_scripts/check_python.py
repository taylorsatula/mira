#!/usr/bin/env python
import sys
import os
import subprocess

# Print Python version info
print("Python Version Info:")
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Python Path: {':'.join(sys.path)}")

# Try to check for other Python versions
print("\nChecking for other Python versions:")
potential_paths = [
    # Standard paths
    "/usr/bin/python", "/usr/bin/python3",
    # Common version-specific paths
    "/usr/bin/python3.6", "/usr/bin/python3.7", "/usr/bin/python3.8", 
    "/usr/bin/python3.9", "/usr/bin/python3.10", "/usr/bin/python3.11",
    # AlmaLinux/CentOS/RHEL paths
    "/usr/bin/python2", "/usr/bin/python2.7",
    # Alt Python paths (common on cPanel)
    "/opt/alt/python36/bin/python", "/opt/alt/python36/bin/python3",
    "/opt/alt/python37/bin/python", "/opt/alt/python37/bin/python3",
    "/opt/alt/python38/bin/python", "/opt/alt/python38/bin/python3",
    "/opt/alt/python39/bin/python", "/opt/alt/python39/bin/python3",
    "/opt/alt/python310/bin/python", "/opt/alt/python310/bin/python3",
    "/opt/alt/python311/bin/python", "/opt/alt/python311/bin/python3",
    # cPanel specific
    "/opt/cpanel/ea-python27/bin/python",
    "/opt/cpanel/ea-python36/bin/python",
    "/opt/cpanel/ea-python38/bin/python",
    "/opt/cpanel/ea-python310/bin/python"
]

for path in potential_paths:
    try:
        output = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT, universal_newlines=True)
        print(f"Found: {path} - {output.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

# Check for WSGI/FCGI related paths
print("\nChecking for WSGI/FCGI paths:")
wsgi_paths = [
    "/opt/alt/python*/bin/lswsgi",
    "/usr/local/lsws/fcgi-bin/lswsgi",
    "/usr/local/bin/lswsgi*", 
    "/usr/bin/lswsgi*",
    "/opt/python*/bin/wsgi*"
]

for pattern in wsgi_paths:
    try:
        files = subprocess.check_output(f"ls -la {pattern} 2>/dev/null || echo 'Not found'", 
                                      shell=True, universal_newlines=True)
        print(f"Pattern {pattern}: {files.strip()}")
    except subprocess.CalledProcessError:
        pass

# Check for Python modules in various directories
print("\nChecking for Python module directories:")
module_paths = [
    "/opt/alt/python*", 
    "/usr/lib/python*", 
    "/usr/local/lib/python*"
]

for pattern in module_paths:
    try:
        dirs = subprocess.check_output(f"ls -la {pattern} 2>/dev/null || echo 'Not found'", 
                                      shell=True, universal_newlines=True)
        print(f"Pattern {pattern}: {dirs.strip()}")
    except subprocess.CalledProcessError:
        pass