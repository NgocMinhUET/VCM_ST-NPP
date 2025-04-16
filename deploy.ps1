# Server connection details
$SERVER_IP = "203.145.216.170"
$SERVER_PORT = "53771"
$SERVER_USER = "u9564043"
$SERVER_PASS = "Sang@26071998"

Write-Host "Converting script to Unix line endings..."
$content = Get-Content "install_x265_remote.sh" -Raw
$content = $content.Replace("`r`n", "`n")
[System.IO.File]::WriteAllText("install_x265_remote.sh", $content)

Write-Host "Copying script to server..."
$scpCommand = "scp -P $SERVER_PORT install_x265_remote.sh $SERVER_USER@$SERVER_IP`:~/"
Write-Host "Running: $scpCommand"
cmd /c $scpCommand

Write-Host "Making script executable and running installation..."
$sshCommand = "ssh -p $SERVER_PORT $SERVER_USER@$SERVER_IP 'cd ~ && chmod +x install_x265_remote.sh && bash install_x265_remote.sh'"
Write-Host "Running: $sshCommand"
cmd /c $sshCommand

Write-Host "Installation process completed!" 