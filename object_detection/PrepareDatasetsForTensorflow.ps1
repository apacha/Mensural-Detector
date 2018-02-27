$pathToGitRoot = "C:/Users/Alex/Repositories/Mensural-Detector/"
$pathToSourceRoot = "C:/Users/Alex/Repositories/Mensural-Detector/object_detection/"
$pathToTranscript = "$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)DatasetPreparationTranscript.txt" -append

cd $pathToSourceRoot
echo "Appending research folder $($pathToGitRoot) to temporary PYTHONPATH"
$env:PYTHONPATH = $env:PYTHONPATH;"$($pathToSourceRoot)"

echo "Testing correct setup"
python builders/model_builder_test.py

echo "Generating data-record in Tensorflow-format"
python generate_mapping.py
python annotation_generator.py
python dataset_splitter.py

python create_tensorflow_record.py --data_dir=..\training_validation_test  	--set=training 		--annotations_dir=annotations 	--output_path=..\training.record 			--label_map_path=mapping.txt
python create_tensorflow_record.py --data_dir=..\training_validation_test  	--set=validation 	--annotations_dir=annotations 	--output_path=..\validation.record 		--label_map_path=mapping.txt
python create_tensorflow_record.py --data_dir=..\training_validation_test  	--set=test 			--annotations_dir=annotations 	--output_path=..\test.record 				--label_map_path=mapping.txt
Stop-Transcript
