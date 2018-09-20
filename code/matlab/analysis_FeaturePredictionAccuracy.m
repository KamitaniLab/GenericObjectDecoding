% analysis_FeaturePredictionAccuracy    Calculate feature prediction accuracy
%
% Author: Tomoyasu Horikawa <horikawa-t@atr.jp>, Shuntaro C. Aoki <aoki@atr.jp>
%


clear all;


%% Initial settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data settings
% subjectList  : List of subject IDs [cell array]
% featureList  : List of image features [cell array]
% roiList      : List of ROIs [cell array]

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
predResultFileNameFormat = @(s, r, f) fullfile(resultsDir, sprintf('%s/%s/%s.mat', s, r, f));
resultFile = fullfile(resultsDir, 'FeaturePrediction.mat');


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
        fprintf('The analysis is already done and skipped\n');
        continue;
    end

    fprintf('Start %s\n', analysisId);

    %% Get image features ----------------------------------------------
    layerFeat = select_feature(feat.dataSet, feat.metaData, sprintf('%s = 1', featureList{iFeat}));
    catIds = get_dataset(feat.dataSet, feat.metaData, 'CatID');
    featType = get_dataset(feat.dataSet, feat.metaData, 'FeatureType');

    %% Aggregate feature units -----------------------------------------
    predResultFile = predResultFileNameFormat(subjectList{iSbj}, roiList{iRoi}, featureList{iFeat});
    res = load(predResultFile);

    predPercept = res.predictPerceptAveraged;
    predImagery = res.predictImageryAveraged;

    categoryPercept = res.categoryTestPercept;
    categoryImagery = res.categoryTestImagery;

    %% Calculate category feature prediction accuracy ------------------

    %% Get test features (images)
    testFeat = layerFeat(featType == 2, :);
    testCatIds = catIds(featType == 2, :);

    testFeat = get_refdata(testFeat, testCatIds, categoryPercept);

    %% Image feature prediction accuracy (profile correlation)
    predAcc.image.perception = nanmean(diag(fastcorr(predPercept, testFeat)));

    %% Get test features (category averaged)
    catTestFeat = layerFeat(featType == 3, :);
    catTestCatIds = catIds(featType == 3, :);

    catTestFeatPercept = get_refdata(catTestFeat, catTestCatIds, categoryPercept);
    catTestFeatImagery = get_refdata(catTestFeat, catTestCatIds, categoryImagery);

    %% Category-average feature prediction accuracy (profile correlation)
    predAcc.category.perception = nanmean(diag(fastcorr(predPercept, catTestFeatPercept)));
    predAcc.category.imagery    = nanmean(diag(fastcorr(predImagery, catTestFeatImagery)));

    results(n).subject = subjectList{iSbj};
    results(n).roi = roiList{iRoi};
    results(n).feature = featureList{iFeat};
    results(n).categoryTestPercept = categoryPercept;
    results(n).categoryTestImagery = categoryImagery;
    results(n).predictPercept = predPercept;
    results(n).predictImagery = predImagery;
    results(n).predaccImagePercept = predAcc.image.perception;
    results(n).predaccCategoryPercept = predAcc.category.perception;
    results(n).predaccCategoryImagery = predAcc.category.imagery;
end

%% Save data -----------------------------------------------------------
save(resultFile, 'results', '-v7.3');

fprintf('%s done\n', mfilename);
