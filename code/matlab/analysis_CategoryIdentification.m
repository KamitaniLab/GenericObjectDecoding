% analysis_CategoryIdentification    Run object category identification
%
% Author: Tomoyasu Horikawa <horikawa-t@atr.jp>, Shuntaro C. Aoki <aoki@atr.jp>
%


clear all;


%% Initial settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data settings
% subjectList  : List of subject IDs [cell array]
% featureList  : List of image features [cell array]
% roiList      : List of RoiList [cell array]

subjectList = {'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'};
featureList = {'cnn1', 'cnn2', 'cnn3', 'cnn4', ...
               'cnn5', 'cnn6', 'cnn7', 'cnn8', ...
               'hmax1', 'hmax2', 'hmax3', 'gist', 'sift'};
roiList     = {'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'};

% Image feature data
imageFeatureFile = 'ImageFeatures.mat';

%% Directory settings
workDir = pwd;
dataDir = fullfile(workDir, 'data');       % Directory containing brain and image feature data
resultsDir = fullfile(workDir, 'results'); % Directory to save analysis results

%% File name settings
predResultFile = fullfile(resultsDir, 'FeaturePrediction.mat');
resultFile = fullfile(resultsDir, 'CategoryIdentification.mat');


%% Analysis Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('%s started\n', mfilename);

%%----------------------------------------------------------------------
%% Initialization
%%----------------------------------------------------------------------

addpath(genpath('./lib'));

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end;

%%----------------------------------------------------------------------
%% Load data
%%----------------------------------------------------------------------

%% Load image features
fprintf('Loading image feature data...\n');

[feat.dataSet, feat.metaData] = load_data(fullfile(dataDir, imageFeatureFile));

%%----------------------------------------------------------------------
%% Create analysis parameter matrix (analysisParam)
%%----------------------------------------------------------------------

analysisParam = uint16(zeros(length(subjectList) * length(roiList) * length(featureList), 3));

c = 1;
for iSbj = 1:length(subjectList)
for iRoi = 1:length(roiList)
for iFeat = 1:length(featureList)
    analysisParam(c, :) = [ iSbj, iRoi, iFeat ];
    c =  c + 1;
end
end
end

%%----------------------------------------------------------------------
%% Analysis loop
%%----------------------------------------------------------------------

featpred = load(predResultFile);

results = [];
for n = 1:size(analysisParam, 1)

    %% Initialization --------------------------------------------------

    % Get data index in the current analysis
    iSbj = analysisParam(n, 1);
    iRoi = analysisParam(n, 2);
    iFeat = analysisParam(n, 3);

    % Set analysis ID and a result file name
    analysisId = sprintf('%s-%s-%s-%s', ...
                         mfilename, ...
                         subjectList{iSbj}, ...
                         roiList{iRoi}, ...
                         featureList{iFeat});

    % Check or double-running
    if checkfiles(resultFile)
        % Analysis result already exists
        fprintf('The analysis is already done and skipped\n');
        continue;
    end

    fprintf('Start %s\n', analysisId);

    %% Get image features ----------------------------------------------
    layerFeat = select_feature(feat.dataSet, feat.metaData, ...
                              sprintf('%s = 1', featureList{iFeat}));
    catIds = get_dataset(feat.dataSet, feat.metaData, 'CatID');
    featType = get_dataset(feat.dataSet, feat.metaData, 'FeatureType');

    %% Load data (unit_all) --------------------------------------------
    ind = strcmp({featpred.results(:).subject}, subjectList{iSbj}) ...
          & strcmp({featpred.results(:).roi}, roiList{iRoi}) ...
          & strcmp({featpred.results(:).feature}, featureList{iFeat});

    % TODO: add index check

    predPercept = featpred.results(ind).predictPercept;
    predImagine = featpred.results(ind).predictImagery;

    categoryPercept = featpred.results(ind).categoryTestPercept;

    %% Object category identification analysis -------------------------

    %% Get category features
    featCatTest = layerFeat(featType == 3, :);
    catIdsCatTest = catIds(featType == 3, :);

    featCatTest = get_refdata(featCatTest, catIdsCatTest, categoryPercept);

    featCatOther = layerFeat(featType == 4, :);
    catIdCatOther = catIds(featType == 4, :);

    %% Pairwise identification
    labels = 1:size(featCatTest, 1);
    candidate = [featCatTest; featCatOther];

    % Seen categories
    simmat = fastcorr(predPercept', candidate');
    correctRate.perception = pwidentification(simmat, labels);

    % Imagined catgories
    simmat = fastcorr(predImagine', candidate');
    correctRate.imagery = pwidentification(simmat, labels);

    fprintf('Correct rate (seen)     = %.f%%\n', mean(correctRate.perception) * 100);
    fprintf('Correct rate (imagined) = %.f%%\n', mean(correctRate.imagery) * 100);

    results(n).subject = subjectList{iSbj};
    results(n).roi = roiList{iRoi};
    results(n).feature = featureList{iFeat};
    results(n).correctRatePercept = correctRate.perception;
    results(n).correctRateImagery = correctRate.imagery;
    results(n).correctRatePerceptAve = mean(correctRate.perception);
    results(n).correctRateImageryAve = mean(correctRate.imagery);
end

%% Save data -----------------------------------------------------------
save(resultFile, 'results', '-v7.3');

fprintf('%s done\n', mfilename);
