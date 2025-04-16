# PowerShell script to build x265 from source
# Ensure you have Git, CMake and NASM installed

# Clone x265 if not already cloned
if (-not (Test-Path "x265_git")) {
    Write-Host "Cloning x265 repository..."
    git clone https://bitbucket.org/multicoreware/x265_git.git
}

# Navigate to build directory
Set-Location x265_git/build/windows

# Run the build script
Write-Host "Building x265..."
./make-solutions.bat

Write-Host "Build complete. Check the build directory for x265.exe"

# Return to original directory
Set-Location ../../.. 