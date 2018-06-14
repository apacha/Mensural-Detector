$pathToSourceRoot = "C:\Users\Alex\Repositories\Mensural-Detector\object_detection\"
$pathToTranscript = "$($pathToSourceRoot)"

cd $pathToSourceRoot
echo "Appending source root $($pathToSourceRoot) to temporary PYTHONPATH"
$env:PYTHONPATH = $pathToSourceRoot

Start-Transcript -path "$($pathToTranscript)Transcript.txt" -append

# Compile Protoc files
Add-Type -AssemblyName System.IO.Compression.FileSystem
function Unzip
{
    param([string]$zipfile, [string]$outpath)

    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

$url = "https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip"
$output = $pathToSourceRoot + "protoc-3.4.0-win32.zip"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $url -OutFile $output

$protoc_folder = $pathToSourceRoot + "Protoc"
Unzip $output $protoc_folder

.\Protoc\bin\protoc.exe --version

cd ..
.\object_detection\Protoc\bin\protoc.exe object_detection/protos/*.proto --python_out=.

rm object_detection/Protoc -Recurse
rm object_detection/protoc-3.4.0-win32.zip

Stop-Transcript