% Convert feature prediction results fot deep image reconstruction
%
% Output files:
%
% decodedfeatures -+- imagery --- matconvnet -+- cnn1 -+- Subject1 -+- FFA --- Mat files
%                  |                          |        |            |
%                  |                          |        |            +- LOC --- Mat files
%                  |                          |        |            |
%                  |                          |        |           ...
%                  |                          |        |
%                  |                          |        +- Subject2 --- ...
%                  |                          |        |
%                  |                          |       ...
%                  |                          |
%                  |                          +- cnn2 --- ...
%                  |                          |
%                  |                         ...
%                  |
%                  +- perception --- ...
%


clear all;


%% Initial settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data settings
% subjectList  : List of subject IDs [cell array]
% roiList      : List of ROIs [cell array]
% featureList  : List of image features [cell array]

subjectList  = {'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'};
roiList      = {'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'};
featureList  = {'cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8'};

network = 'matconvnet';

%% Directory settings
workDir = pwd;
resultsDir = fullfile(workDir, 'results'); % Directory to save analysis results
outputDir = fullfile(workDir, 'decodedfeatures');  % Directory to save converted decoded features

%% File name settings
resultFileNameFormat = @(s, r, f) fullfile(resultsDir, sprintf('%s/%s/%s.mat', s, r, f));


%% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for isub = 1:length(subjectList)
for iroi = 1:length(roiList)
for ifeat = 1:length(featureList)
    subject = subjectList{isub};
    roi = roiList{iroi};
    feature = featureList{ifeat};

    resultfile = resultFileNameFormat(subject, roi, feature);

    dirDecodedFeatruePercept = fullfile(outputDir, 'perception', network, ...
                                        featureList{ifeat}, subjectList{isub}, roiList{iroi});
    dirDecodedFeatrueImagery = fullfile(outputDir, 'imagery', network, ...
                                        featureList{ifeat}, subjectList{isub}, roiList{iroi});

    if ~exist(resultfile)
        error(sprintf('Result file not found: %s\n', resultfile));
    end

    res = load(resultfile);

    categoryTestPercept = res.categoryTestPercept;
    categoryTestImagery = res.categoryTestImagery;
    predfeatPerceptAve = res.predictPerceptAveraged;
    predfeatImageryAve = res.predictImageryAveraged;

    %% Save decoded features for image reconstruction
    setupdir(dirDecodedFeatruePercept);
    setupdir(dirDecodedFeatrueImagery);

    subsPercept.type = '()';
    subsPercept.subs = repmat({':'}, 1, ndims(predfeatPerceptAve));
    
    subsImagery.type = '()';
    subsImagery.subs = repmat({':'}, 1, ndims(predfeatImageryAve));
    
    for p = 1:length(categoryTestPercept)
        catid = categoryTestPercept(p);
        subsPercept.subs{1} = p;
        
        feat = squeeze(shiftdim(subsref(predfeatPerceptAve, subsPercept), 1));
        resfilePercept = fullfile(dirDecodedFeatruePercept, ...
                                  sprintf('%s-%s-%s-%s-%s-%s-%d.mat', ...
                                          'decodedfeatures', 'perception', network, ...
                                          featureList{ifeat}, subjectList{isub}, roiList{iroi}, catid));
        save(resfilePercept, 'feat', '-v7.3');
    end
    for p = 1:length(categoryTestImagery)
        catid = categoryTestImagery(p);
        subsImagery.subs{1} = p;

        feat = squeeze(shiftdim(subsref(predfeatImageryAve, subsImagery), 1));
        resfileImagery = fullfile(dirDecodedFeatrueImagery, ...
                                  sprintf('%s-%s-%s-%s-%s-%s-%d.mat', ...
                                          'decodedfeatures', 'imagery', network, ...
                                          featureList{ifeat}, subjectList{isub}, roiList{iroi}, catid));
        save(resfileImagery, 'feat', '-v7.3');
    end

end
end
end
