$pathToGitRoot = "C:/Users/Alex/Repositories/Mensural-Detector"
$pathToSourceRoot = "$($pathToGitRoot)/object_detection"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToSourceRoot

Start-Transcript -path "$($pathToTranscript)/Freeze_Model.txt" -append

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot);$($pathToGitRoot)/slim"

# Replace with your path to the checkpoint folder and the respective checkpoint number
$pathToCheckpoint = "C:\Users\Alex\Repositories\Mensural-Detector\object_detection\data\checkpoints-faster_rcnn_inception_resnet_v2_atrous_600_proposals_cross_validation-1-train"
$checkpointNumber = "40583"

python export_inference_graph.py `
    --input_type image_tensor `
    --pipeline_config_path "$($pathToCheckpoint)\pipeline.config"  `
    --trained_checkpoint_prefix "$($pathToCheckpoint)\model.ckpt-$($checkpointNumber)" `
    --output_directory output_inference_graph