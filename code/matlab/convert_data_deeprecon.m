% Convert feature prediction results fot deep image reconstruction
%
% Result files:
%
% results/decodedfeatures -+- imagery --- matconvnet -+- cnn1 -+- Subject1 -+- FFA --- Mat files
%                          |                          |        |            |
%                          |                          |        |            +- LOC --- Mat files
%                          |                          |        |            |
%                          |                          |        |           ...
%                          |                          |        |
%                          |                          |        +- Subject2 --- ...
%                          |                          |        |
%                          |                          |       ...
%                          |                          |
%                          |                          +- cnn2 --- ...
%                          |                          |
%                          |                         ...
%                          |
%                          +- perception --- ...
%


clear all;


%% Initial settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data settings
% subjectList  : List of subject IDs [cell array]
% roiList      : List of ROIs [cell array]
% featureList  : List of image features [cell array]

subjectList  = {'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'};
roiList      = {'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'};
numVoxelList = { 500,  500,  500,  500,   500,   500,   500,  1000,  1000,  1000};
featureList  = {'cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8'};

%% Directory settings
workDir = pwd;
resultsDir = fullfile(workDir, 'results'); % Directory to save analysis results

%% File name settings
resultFileNameFormat = @(s, r, f) fullfile(resultsDir, sprintf('%s/%s/%s.mat', s, r, f));

%% Model parameters
nTrain = 200; % Num of total training iteration
nSkip  = 200; % Num of skip steps for display info

%--------------------------------------------------------------------------------%
% Note: The num of training iteration (`nTrain`) was 2000 in the original paper. %
%--------------------------------------------------------------------------------%


%% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for isub = 1:length(subjectList)
for iroi = 1:length(roiList)
for ifeat = 1:length(featureList)
    subject = subjectList{isub};
    roi = roiList{iroi};
    feature = featureList{ifeat};

    resultfile = resultFileNameFormat(subject, roi, feature);

    dirDecodedFeatruePercept = fullfile(resultsDir, 'decodedfeatures', 'perception', 'matconvnet', ...
                                        featureList{ifeat}, subjectList{isub}, roiList{iroi});
    dirDecodedFeatrueImagery = fullfile(resultsDir, 'decodedfeatures', 'imagery', 'matconvnet', ...
                                        featureList{ifeat}, subjectList{isub}, roiList{iroi});

    if ~exist(resultfile)
        error(sprintf('Result file not found: %s\n', resultfile));
    end

    res = load(resultfile);

    testPerceptCategory = res.results(1).categoryTestPercept;
    predfeatPerceptAve = [res.results(:).predictPerceptCatAve];
    predfeatImageryAve = [res.results(:).predictImageryCatAve];

    %% Save decoded features for image reconstruction
    setupdir(dirDecodedFeatruePercept);
    setupdir(dirDecodedFeatrueImagery);

    for p = 1:length(testPerceptCategory)
        catid = testPerceptCategory(p);
        
        feat = predfeatPerceptAve(p, :);
        resfilePercept = fullfile(dirDecodedFeatruePercept, ...
                                  sprintf('%s-%s-%s-%s-%s-%s-%d.mat', ...
                                          'decodedfeatures', 'perception', 'matconvnet', ...
                                          featureList{ifeat}, subjectList{isub}, roiList{iroi}, catid));
        save(resfilePercept, 'feat', '-v7.3');

        feat = predfeatImageryAve(p, :);
        resfileImagery = fullfile(dirDecodedFeatrueImagery, ...
                                  sprintf('%s-%s-%s-%s-%s-%s-%d.mat', ...
                                          'decodedfeatures', 'imagery', 'matconvnet', ...
                                          featureList{ifeat}, subjectList{isub}, roiList{iroi}, catid));
        save(resfileImagery, 'feat', '-v7.3');
    end

end
end
end
