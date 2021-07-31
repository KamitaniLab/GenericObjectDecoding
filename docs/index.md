# Generic Object Decoding

## Data (fMRI data and visual features)

The fMRI data for five subjects (training, test_perception, and test_imagery) and visual features (CNN1-8, HMAX1-3, GIST, and SIFT) are avairable from <http://brainliner.jp/data/brainliner/Generic_Object_Decoding>. The fMRI data were saved according to the BrainDecoderToolbox2 format <https://github.com/KamitaniLab/BrainDecoderToolbox2>.

## Visual images

For copyright reasons, we do not make the visual images used in our experiments publicly available on our server. You can directly download the images from URLs listed in the imageURL_training.txt (1-1200 training images)  and imageURL_test.txt (1-50 test images) files, which also can be downloaded from <http://brainliner.jp/data/brainliner/Generic_Object_Decoding>. 

While most of the images can be downloaded from the URLs, some of the images have been deleted from the original URLs. In that case, you need to create your account for the ImageNet (<http://image-net.org/index>)  to download the image set for each synset (2011, fall release; <http://image-net.org/download-images>), and search corresponding images using the synset ID and image ID listed in imageID_training.csv and imageID_test.csv files. The image filename is formatted as xxxx_yyyy, where xxxx represents the WordNet synset ID and yyyy represents the image ID.

For easier access to the stimulus images used in our study, we provide a matlab code (get_expStimImages_from_imagenet.m) for downloading all the training and test images from the ImageNet (while you still need to register to the ImageNet). The code requires your username and accesskey, which are provided from the ImageNet, as its arguments. Using this code, you can automatically collect the images cropped and resized as in our study. Note that it will take about one day for collecting all the experimental stimuli with a single computer. But you can run the code with multiple computers in parallell to accelarate the processings.

Please contact us (<kamitanilab@gmail.com>) if you have trouble with the above method, so that we will help you obtaining the images. 

## Demo codes

Please visit <https://github.com/KamitaniLab/GenericObjectDecoding> for the demo code.
