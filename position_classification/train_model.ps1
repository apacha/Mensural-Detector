$pathToSourceRoot = "C:\Users\Alex\Repositories\Mensural-Detector\position_classification"
$pathToTranscript = "$($pathToSourceRoot)\transcripts"

cd $pathToSourceRoot
echo "Appending source root $($pathToSourceRoot) to temporary PYTHONPATH"
$env:PYTHONPATH = $env:PYTHONPATH;"$($pathToSourceRoot)\"

# Run on first time
python extract_sub_image_for_classification.py
python dataset_splitter.py

Start-Transcript -path "$($pathToTranscript)\2018-05-03_vgg4_128x448_mb16.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name vgg4 --width 128 --height 448
Stop-Transcript

exit
Start-Transcript -path "$($pathToTranscript)\2018-05-03_res_net_4_128x448_mb16.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name res_net_4 --width 128 --height 448
Stop-Transcript