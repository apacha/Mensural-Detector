$pathToGitRoot = "C:/Users/Alex/Repositories/Mensural-Detector"
$pathToSourceRoot = "$($pathToGitRoot)/position_classification"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToSourceRoot

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot)"

$dataset_directory = "data"
$width = 160
$height = 448

## Pick by what criterion the samples should be grouped by 
## (staff_position for pitch-detection, class_name for general symbol classification, e.g. for finding incorrectly labelled symbols)
$group_by = "staff_position"
#$group_by = "class_name"

$models = @() # create empty array
## Pick The ones, you want to run
# $models += "inception_resnet_v2_pretrained"
# $models += "inception_resnet_v2"
# $models += "res_net_4"
# $models += "res_net_50"
$models += "dense_net_201"
# $models += "vgg4"
$models += "vgg16"

foreach ($model in $models) {
	# Run a training per selected model
	$transcript_path = "$($pathToTranscript)\2018-03-21_$($model_name)_$($width)x$($height).txt"

	Start-Transcript -path $transcript_path -append
	Remove-Item -Recurse -Force $dataset_directory
	python extract_sub_image_for_classification.py --output_directory $dataset_directory --width $width --height $height --group_by $group_by
	python dataset_splitter.py --source_directory $dataset_directory --destination_directory $dataset_directory
	python "$($pathToSourceRoot)\train_model.py" --dataset_directory $dataset_directory --model_name $model_name --width $width --height $height
	Stop-Transcript
}