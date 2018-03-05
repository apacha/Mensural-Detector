$pathToSourceRoot = "C:\Users\Alex\Repositories\Mensural-Detector\position_classification"
$pathToTranscript = "$($pathToSourceRoot)\transcripts"

cd $pathToSourceRoot
echo "Appending source root $($pathToSourceRoot) to temporary PYTHONPATH"
$env:PYTHONPATH = $env:PYTHONPATH;"$($pathToSourceRoot)\"

Start-Transcript -path "$($pathToTranscript)\2018-05-03_vgg4_128x448_mb8.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name vgg4 --width 128 --height 448
Stop-Transcript