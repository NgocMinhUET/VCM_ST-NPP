#!/bin/bash

echo "Checking for FFmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is already installed:"
    ffmpeg -version
    exit 0
fi

echo "FFmpeg not found. Installing FFmpeg..."

# Detect OS type
OS_TYPE=$(uname -s)

if [ "$OS_TYPE" == "Linux" ]; then
    # Check for common package managers
    if command -v apt-get &> /dev/null; then
        echo "Installing FFmpeg using apt-get..."
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        echo "Installing FFmpeg using yum..."
        sudo yum install -y epel-release
        sudo yum install -y ffmpeg ffmpeg-devel
    elif command -v dnf &> /dev/null; then
        echo "Installing FFmpeg using dnf..."
        sudo dnf install -y ffmpeg ffmpeg-devel
    elif command -v pacman &> /dev/null; then
        echo "Installing FFmpeg using pacman..."
        sudo pacman -S --noconfirm ffmpeg
    else
        echo "Could not detect package manager. Trying to compile from source..."
        compile_from_source
    fi
elif [ "$OS_TYPE" == "Darwin" ]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "Installing FFmpeg using Homebrew..."
        brew install ffmpeg
    else
        echo "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        echo "Installing FFmpeg using Homebrew..."
        brew install ffmpeg
    fi
else
    echo "Unsupported operating system: $OS_TYPE"
    exit 1
fi

# Function to compile FFmpeg from source if package manager is not available
compile_from_source() {
    echo "Compiling FFmpeg from source..."
    
    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR" || exit 1
    
    # Install build dependencies
    echo "Installing build dependencies..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget
    elif command -v yum &> /dev/null; then
        sudo yum install -y autoconf automake bzip2 cmake freetype-devel gcc gcc-c++ git libtool make mercurial pkgconfig zlib-devel
    fi
    
    # Download FFmpeg source
    wget https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
    tar xjf ffmpeg-snapshot.tar.bz2
    cd ffmpeg || exit 1
    
    # Configure and build
    ./configure --enable-gpl --enable-nonfree
    make -j "$(nproc)"
    sudo make install
    
    # Clean up
    cd "$TEMP_DIR" || exit 1
    rm -rf "$TEMP_DIR"
}

# Verify installation
echo "Verifying FFmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg has been installed successfully:"
    ffmpeg -version
else
    echo "FFmpeg installation failed."
    echo "Please install FFmpeg manually from https://ffmpeg.org/download.html"
    exit 1
fi

echo "FFmpeg setup complete." 