#!/bin/bash

# Server connection details
SERVER_IP="203.145.216.170"
SERVER_PORT="53771"
SERVER_USER="u9564043"

# Create SSH command with password (using sshpass if available)
if command -v sshpass &> /dev/null; then
    SSH_CMD="sshpass -p 'Sang@26071998' ssh -p $SERVER_PORT $SERVER_USER@$SERVER_IP"
else
    SSH_CMD="ssh -p $SERVER_PORT $SERVER_USER@$SERVER_IP"
fi

# Create the installation script
cat > install_x265_remote.sh << 'EOL'
#!/bin/bash

# Set up installation directory in user's home
INSTALL_DIR=$HOME/local
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/src

# Add local bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/local/bin:"* ]]; then
    echo 'export PATH=$HOME/local/bin:$PATH' >> $HOME/.bashrc
fi

# Add local lib to LD_LIBRARY_PATH if not already there
if [[ ":$LD_LIBRARY_PATH:" != *":$HOME/local/lib:"* ]]; then
    echo 'export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH' >> $HOME/.bashrc
fi

# Install dependencies in local directory
cd $INSTALL_DIR/src

# Install CMake if not available
if ! command -v cmake &> /dev/null; then
    echo "Installing CMake locally..."
    wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4.tar.gz
    tar xzf cmake-3.26.4.tar.gz
    cd cmake-3.26.4
    ./bootstrap --prefix=$INSTALL_DIR
    make -j$(nproc)
    make install
    cd ..
fi

# Install NASM if not available
if ! command -v nasm &> /dev/null; then
    echo "Installing NASM locally..."
    wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.gz
    tar xzf nasm-2.15.05.tar.gz
    cd nasm-2.15.05
    ./configure --prefix=$INSTALL_DIR
    make -j$(nproc)
    make install
    cd ..
fi

# Clone and build x265
echo "Building x265..."
if [ ! -d "x265" ]; then
    git clone https://bitbucket.org/multicoreware/x265_git.git x265
fi
cd x265/build/linux
PATH=$INSTALL_DIR/bin:$PATH cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../../source
make -j$(nproc)
make install

echo "Installation complete!"
echo "Please run: source ~/.bashrc"
echo "x265 has been installed to: $INSTALL_DIR"
EOL

# Make the installation script executable
chmod +x install_x265_remote.sh

# Copy the installation script to the server
scp -P $SERVER_PORT install_x265_remote.sh $SERVER_USER@$SERVER_IP:~/

# Execute the installation script on the server
$SSH_CMD "bash install_x265_remote.sh"

# Clean up local installation script
rm install_x265_remote.sh 