
# Multimodal Emotion Recognition for QAI Hackathon
	- This code is based on the  baseline code for MERC2020
	- This code is based on keras(2.2.4) and tensorflow(1.14).
## Name:
	Yonatan Gizachew
	Bethelhem Nigat

## Installation
	pip3 install -r requirement.txt


## Useful code for data preperation: Run this first

	mp4 to jpg: utils/video_to_image.py
	text emb to npy: utils/text_to_npy.py
	label to npy: utils/label_to.npy.py

## Text dataset preparation

### Root path
	cd model/multimodal

### Usage examples

	1. Load dataset
	python3 load_text.py


## Video dataset preparation

### Root path
	cd model/multimodal

### Usage examples

	1. Get features using VGG face
	CUDA_VISIBLE_DEVICES=0 python3 video_feature_extract.py

	2. To prepare video dataset for test
	CUDA_VISIBLE_DEVICES=0 python3 video_feature_extract_test.py

	2. Load dataset
	python3 load.py

## Multimodal model (without speech dataset)

### Root path
	cd modules/multimodal

### Usage examples 
	1. Get bottleneck features from each single modal
	CUDA_VISIBLE_DEVICES=0 python3 feature_extract.py
	
	2. Train model
	CUDA_VISIBLE_DEVICES=0 python3 train.py

	2. Evaluate model
	CUDA_VISIBLE_DEVICES=0 python3 evaluation.py

## Multimodal model (with speech dataset)

### Root path
	cd modules/multimodal

### Usage examples 
	1. Get bottleneck features from each single modal
	CUDA_VISIBLE_DEVICES=0 python3 feature_extract.py
	
	2. Train model
	CUDA_VISIBLE_DEVICES=0 python3 train_including_speech.py

	2. Evaluate model
	CUDA_VISIBLE_DEVICES=0 python3 evaluation_including_speech.py
	
### Performance
 	- Accuracy: 45.62% (without speech. I didn't have speech dataset)


