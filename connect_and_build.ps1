# Server connection details
$SERVER_IP = "203.145.216.170"
$SERVER_PORT = "53771"
$SERVER_USER = "u9564043"
$SERVER_PASS = "Sang@26071998"

# Create the installation script content
$installScript = @'
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
'@

# Save the installation script with Unix line endings
$installScript = $installScript.Replace("`r`n", "`n")
[System.IO.File]::WriteAllText("install_x265_remote.sh", $installScript)

# Check if OpenSSH is available
if (-not (Get-Command "ssh" -ErrorAction SilentlyContinue)) {
    Write-Host "OpenSSH is required but not found."
    Write-Host "Please enable OpenSSH in Windows Features or install it."
    exit 1
}

# Create known_hosts entry if it doesn't exist
$knownHostsPath = "$env:USERPROFILE\.ssh\known_hosts"
$sshDir = "$env:USERPROFILE\.ssh"

if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir
}

# Get the server's key and add it to known_hosts
Write-Host "Adding server to known_hosts..."
$null = ssh-keyscan -p $SERVER_PORT $SERVER_IP 2>$null | Out-File -Append -Encoding ASCII $knownHostsPath

# Create a temporary expect script for SSH password automation
$expectScript = @"
#!/usr/bin/expect -f
spawn ssh -p $SERVER_PORT $SERVER_USER@$SERVER_IP "cat > install_x265_remote.sh"
expect "password:"
send "$SERVER_PASS\r"
interact
"@

$expectScript2 = @"
#!/usr/bin/expect -f
spawn ssh -p $SERVER_PORT $SERVER_USER@$SERVER_IP "chmod +x install_x265_remote.sh && ./install_x265_remote.sh"
expect "password:"
send "$SERVER_PASS\r"
interact
"@

# Save expect scripts
[System.IO.File]::WriteAllText("ssh_copy.exp", $expectScript.Replace("`r`n", "`n"))
[System.IO.File]::WriteAllText("ssh_run.exp", $expectScript2.Replace("`r`n", "`n"))

Write-Host "Copying installation script to server..."
Get-Content "install_x265_remote.sh" | expect ssh_copy.exp

Write-Host "Making script executable and running installation..."
expect ssh_run.exp

# Clean up temporary files
Remove-Item "install_x265_remote.sh"
Remove-Item "ssh_copy.exp"
Remove-Item "ssh_run.exp"

Write-Host "Installation process completed!" 