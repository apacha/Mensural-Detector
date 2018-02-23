$pathToGitRoot = "C:/Users/Alex/Repositories/Mensural-Detector"
$pathToSourceRoot = "$($pathToGitRoot)/tensorflow_object_detection"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
cd $pathToSourceRoot

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
$configuration = "faster_rcnn_inception_resnet_v2_atrous"


Start-Transcript -path "$($pathToTranscript)/TestModel-$($configuration).txt" -append
echo "Testing with $($configuration) configuration"
python eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/checkpoints-$($configuration)-train" --eval_dir="$($pathToSourceRoot)/data/checkpoints-$($configuration)-test"
# Inbetween the tests, change the config to use the weighted Pascal VOC metrics.
python eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/checkpoints-$($configuration)-train" --eval_dir="$($pathToSourceRoot)/data/checkpoints-$($configuration)-test-weighted"
Stop-Transcript
