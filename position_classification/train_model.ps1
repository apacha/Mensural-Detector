$pathToGitRoot = "C:/Users/Alex/Repositories/Mensural-Detector"
$pathToSourceRoot = "$($pathToGitRoot)/position_classification"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToSourceRoot

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)\2018-03-20_inception_resnet_v2_160x448_mb8.txt" -append

# Delete and re-create dataset directory
Remove-Item -Recurse -Force data
python extract_sub_image_for_classification.py --width 160 --height 448 --group_by staff_position
python dataset_splitter.py

python "$($pathToSourceRoot)\train_model.py" --model_name inception_resnet_v2 --width 160 --height 448
Stop-Transcript


exit

Start-Transcript -path "$($pathToTranscript)\2018-03-06_vgg4_128x448_mb16_reduced_classes.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name vgg4 --width 128 --height 448
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-03-06_vgg4_128x448_mb16_reduced_classes_sklearn_balance.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name vgg4 --width 128 --height 448
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-03-05_vgg4_128x448_mb16.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name vgg4 --width 128 --height 448
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-03-05_res_net_4_128x448_mb16.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name res_net_4 --width 128 --height 448
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-05-03_vgg4_128x448_mb16.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name vgg4 --width 128 --height 448
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-05-03_res_net_4_128x448_mb16.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name res_net_4 --width 128 --height 448
Stop-Transcript