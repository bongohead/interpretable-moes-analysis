$remoteFiles = (ssh -p 24439 root@195.26.232.184 "ls -1 /workspace/interpretable-moes/experiments/base-olmoe-lflb/logs/*.csv") -split "`n"
foreach ($file in $remoteFiles) {
	$filename = Split-Path $file -Leaf
	if (-not (Test-Path "data/$filename")) {
		Write-Host "Downloading new file: $filename"
		scp -P 24439 "root@195.26.232.184:/workspace/interpretable-moes/experiments/base-olmoe-lflb/logs/$filename" data
	} else {
		Write-Host "Skipping existing file: $filename"
	}
}