$pathToGitRoot = "C:/Users/Alex/Repositories/Mensural-Detector"
$pathToSourceRoot = "$($pathToGitRoot)/position_classification"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToSourceRoot

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)\2018-03-21_inception_resnet_v2_pretrained_160x448_group_by_classes.txt" -append

# Delete and re-create dataset directory
$dataset_directory = "data_classes"
Remove-Item -Recurse -Force $dataset_directory
python extract_sub_image_for_classification.py --output_directory $dataset_directory --width 160 --height 448 --group_by class_name
python dataset_splitter.py --source_directory $dataset_directory --destination_directory $dataset_directory

python "$($pathToSourceRoot)\train_model.py" --dataset_directory $dataset_directory --model_name inception_resnet_v2_pretrained --width 160 --height 448 --output_name 2018-03-21_inception_resnet_v2_pretrained_160x448_group_by_classes
Stop-Transcript

exit

Start-Transcript -path "$($pathToTranscript)\2018-03-21_inception_resnet_v2_pretrained_160x448.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name inception_resnet_v2_pretrained --width 160 --height 448
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-03-20_inception_resnet_v2_160x448_mb8.txt" -append
python "$($pathToSourceRoot)\train_model.py" --model_name inception_resnet_v2 --width 160 --height 448
Stop-Transcript

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