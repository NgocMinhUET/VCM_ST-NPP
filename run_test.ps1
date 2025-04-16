# Server connection details
$SERVER_IP = "203.145.216.170"
$SERVER_PORT = "53771"
$SERVER_USER = "u9564043"
$SERVER_PASS = "Sang@26071998"

Write-Host "Converting test script to Unix line endings..."
$content = Get-Content "test_x265.sh" -Raw
$content = $content.Replace("`r`n", "`n")
[System.IO.File]::WriteAllText("test_x265.sh", $content)

Write-Host "Copying test script to server..."
$scpCommand = "scp -P $SERVER_PORT test_x265.sh $SERVER_USER@$SERVER_IP`:~/"
Write-Host "Running: $scpCommand"
cmd /c $scpCommand

Write-Host "Making script executable and running test..."
$sshCommand = "ssh -tt -p $SERVER_PORT $SERVER_USER@$SERVER_IP `"cd ~ && chmod +x test_x265.sh && bash test_x265.sh`""
Write-Host "Running: $sshCommand"
cmd /c $sshCommand 