$pathToGitRoot = "C:/Users/Alex/Repositories/Mensural-Detector"
$pathToSourceRoot = "$($pathToGitRoot)/object_detection"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToSourceRoot

Start-Transcript -path "$($pathToTranscript)/DatasetPreparationTranscript.txt" -append
$number_of_splits = 5

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot);$($pathToGitRoot)/slim"

echo "Testing correct setup"
python builders/model_builder_test.py

echo "Generating data-record in Tensorflow-format"
python generate_mapping.py
python annotation_generator.py
python dataset_splitter.py --destination_directory "..\training_test_splits" --number_of_splits $number_of_splits

For ($index = 1; $index -le 5; $index++) {
    python create_tensorflow_record.py --data_dir="..\training_test_splits\split-$($index)" --set=training --annotations_dir=..\annotations --output_path="..\training-$($index).record" --label_map_path=mapping.txt
    python create_tensorflow_record.py --data_dir="..\training_test_splits\split-$($index)" --set=test --annotations_dir=..\annotations --output_path="..\test-$($index).record" --label_map_path=mapping.txt
}
Stop-Transcript