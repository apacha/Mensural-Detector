$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
cd $pathToGitRoot/research

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_no_staff_lines"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes_no_staff_lines"
# $configuration = "faster_rcnn_resnet50_muscima_pretrained"
# $configuration = "faster_rcnn_resnet50_muscima_pretrained2"
# $configuration = "faster_rcnn_resnet50_muscima_windows"
# $configuration = "faster_rcnn_resnet50_muscima_windows_2"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_with_stafflines_data_augmentation"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_with_stafflines_data_augmentation2"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_with_stafflines_dimension_clustering"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_with_stafflines_dimension_clustering2"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_with_stafflines_dimension_clustering2_rms"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_with_stafflines_dimension_clustering3"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_with_stafflines_dimension_clustering3_rms"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_with_stafflines_dimension_clustering4"
$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_with_stafflines_dimension_clustering4_rms"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_with_stafflines_more_scales_and_ratios"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_with_stafflines_more_scales_and_ratios2"
# $configuration = "rfcn_inception_resnet_v2_atrous_muscima_pretrained"
# $configuration = "rfcn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes"
# $configuration = "rfcn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes2"
# $configuration = "rfcn_resnet50_muscima"
# $configuration = "rfcn_resnet50_muscima_pretrained_no_staff_lines"
# $configuration = "rfcn_resnet50_muscima_pretrained_reduced_classes"
# $configuration = "rfcn_resnet50_muscima_pretrained_reduced_classes_no_staff_lines"
# $configuration = "rfcn_resnet50_muscima_reduced_classes"
# $configuration = "rfcn_resnet50_muscima_reduced_classes_no_staff_lines"
# $configuration = "ssd_inception_v2_muscima_150x300_pretrained"
# $configuration = "ssd_inception_v2_muscima_150x300_pretrained_reduced_classes"
# $configuration = "ssd_inception_v2_muscima_150x300_pretrained_reduced_classes_no_stafflines"
# $configuration = "ssd_mobilenet_v1_muscima_150x300"
# $configuration = "ssd_mobilenet_v1_muscima_150x300_pretrained"
# $configuration = "ssd_inception_v2_muscima_300x600_pretrained_2"


Start-Transcript -path "$($pathToTranscript)/TestModel-$($configuration).txt" -append
echo "Testing with $($configuration) configuration"
python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/checkpoints-$($configuration)-train" --eval_dir="$($pathToSourceRoot)/data/checkpoints-$($configuration)-test"
# Inbetween the tests, change the config to use the weighted Pascal VOC metrics.
# python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/checkpoints-$($configuration)-train" --eval_dir="$($pathToSourceRoot)/data/checkpoints-$($configuration)-test-weighted"
Stop-Transcript
