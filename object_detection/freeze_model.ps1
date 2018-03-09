$pathToGitRoot = "C:/Users/Alex/Repositories/Mensural-Detector"
$pathToSourceRoot = "$($pathToGitRoot)/object_detection"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToSourceRoot

Start-Transcript -path "$($pathToTranscript)/Freeze_Model.txt" -append

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot);$($pathToGitRoot)/slim"

# Replace with your path to the checkpoint folder and the respective checkpoint number
$pathToCheckpoint = "E:\MensuralDetector\checkpoints-faster_rcnn_inception_resnet_v2_atrous_600_proposals_small_scale-train"
$checkpointNumber = "49683"

python export_inference_graph.py `
    --input_type image_tensor `
    --pipeline_config_path "$($pathToCheckpoint)\pipeline.config"  `
    --trained_checkpoint_prefix "$($pathToCheckpoint)\model.ckpt-$($checkpointNumber)" `
    --output_directory output_inference_graph.pb