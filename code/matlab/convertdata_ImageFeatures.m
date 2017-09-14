% convertdata_ImageFeatures    Convert image feature data to BrainDecoderToolbox 2 format
%
% Author: Shuntaro C. Aoki <aoki@atr.jp>
%

clear all;

%% Image features
features = {'cnn1', 'cnn2', 'cnn3', 'cnn4', ...
            'cnn5', 'cnn6', 'cnn7', 'cnn8', ...
            'hmax1', 'hmax2', 'hmax3', 'gist', 'sift'};
%features = {'cnn1'};

workDir = pwd;
dataDir = fullfile(workDir, 'data');

imageListTrain = './data/images/image_training.csv';
imageListTest = './data/images/image_test.csv';
catListTrainFile = './data/images/category_training.csv';
catListTestFile = './data/images/category_test.csv';
catListCandidateFile = './data/images/category_candidate.csv';

saveDir = './data';

trainLabel = 1;
testLabel = 2;
catTestLabel = 3;
catOtherLabel = 4;

%% Create dataSet+metaData
[dataSet, metaData] = initialize_dataset();

for n = 1:length(features)
    featName = features{n};
    
    fTrain = load(fullfile(dataDir, 'feature', 'train', sprintf('%s.mat', featName)));
    fTest = load(fullfile(dataDir, 'feature', 'test', sprintf('%s.mat', featName)));    
    fCatTest = load(fullfile(dataDir, 'feature', 'candidate', 'test', ...
                             sprintf('%s_mu.mat', featName)));
    fCatOther = load(fullfile(dataDir, 'feature', 'candidate', 'others', ...
                              sprintf('%s_mu.mat', featName)));

    numTrain = size(fTrain.(featName), 1);
    numTest = size(fTest.(featName), 1);
    numCatTest = size(fCatTest.(featName), 1);
    numCatOther = size(fCatOther.(featName), 1);

    ds = [fTrain.(featName);
          fTest.(featName);
          fCatTest.(featName);
          fCatOther.(featName)];

    [dataSet, metaData] = add_dataset(dataSet, metaData, ...
                                      ds, ...
                                      featName, ...
                                      sprintf('1 = %s', featName));
end

% Add image id
imgIdListTrain = textread(imageListTrain, '%s');
imgIdListTest = textread(imageListTest, '%s');

% Convert image file names to image IDs
for i = 1:length(imgIdListTrain)
    idcell = strsplit(strrep(strrep(imgIdListTrain{i}, '.JPEG', ''), 'n', '0'), '_');
    imgIdTrainStr{i, 1} = sprintf('%s.%06s', idcell{1}, idcell{2});
end
for i = 1:length(imgIdListTest)
    idcell = strsplit(strrep(strrep(imgIdListTest{i}, '.JPEG', ''), 'n', '0'), '_');
    imgIdTestStr{i, 1} = sprintf('%s.%06s', idcell{1}, idcell{2});
end

imgIdTrain = cellfun(@str2num, imgIdTrainStr);
imgIdTest = cellfun(@str2num, imgIdTestStr);

imageIds = [ imgIdTrain;
             imgIdTest;
             nan(numCatTest, 1);
             nan(numCatOther, 1)];

[dataSet, metaData] = add_dataset(dataSet, metaData, ...
                                  imageIds, ...
                                  'ImageID', ...
                                  'Image ID');

% Add category id
catIdListTrain = textread(catListTrainFile, '%s');
catIdListTest = textread(catListTestFile, '%s');
catIdListCandidate = textread(catListCandidateFile, '%s');

catIdTrain = cellfun(@str2num, catIdListTrain);
catIdTest = cellfun(@str2num, catIdListTest);
catIdCand = cellfun(@str2num, catIdListCandidate);

nImgTrain = 8;

catIds = [ reshape(repmat(catIdTrain', nImgTrain, 1), numel(catIdTrain) * nImgTrain, 1);
           catIdTest;
           catIdTest;
           catIdCand ];

[dataSet, metaData] = add_dataset(dataSet, metaData, ...
                                  catIds, ...
                                  'CatID', ...
                                  'Category ID');

% Add image type labels
typeLabel = [ repmat(trainLabel, numTrain, 1);
              repmat(testLabel, numTest, 1);
              repmat(catTestLabel, numCatTest, 1);
              repmat(catOtherLabel, numCatOther, 1) ];

[dataSet, metaData] = add_dataset(dataSet, metaData, ...
                                  typeLabel, ...
                                  'FeatureType', ...
                                  sprintf('1 = ImageType (1 = training images; 2 = test images; 3 = test category averaged; 4 = novel (candidate) category averaged', featName));

% Save data
save_data(fullfile(saveDir, 'ImageFeatures.mat'), ...
          dataSet, metaData);
