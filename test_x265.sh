#!/bin/bash

# Source the environment variables
source ~/.bashrc

# Print environment information
echo "=== Environment Information ==="
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check x265 installation
echo -e "\n=== x265 Installation Check ==="
if command -v x265 &> /dev/null; then
    echo "x265 is installed!"
    echo "Location: $(which x265)"
    echo "Version information:"
    x265 --version
else
    echo "x265 is not found in PATH"
fi

# Check if the library exists
echo -e "\n=== x265 Library Check ==="
if [ -f "$HOME/local/lib/libx265.so" ]; then
    echo "x265 library found at: $HOME/local/lib/libx265.so"
    ls -l "$HOME/local/lib/libx265.so"
else
    echo "x265 library not found in $HOME/local/lib/"
fi 