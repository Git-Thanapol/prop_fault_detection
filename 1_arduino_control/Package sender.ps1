$udpClient = New-Object System.Net.Sockets.UdpClient
$udpClient.Connect("192.168.0.200", 8888)
$sendBytes = [System.Text.Encoding]::ASCII.GetBytes("$CO1,50,-50")
$udpClient.Send($sendBytes, $sendBytes.Length)
$udpClient.Close()
