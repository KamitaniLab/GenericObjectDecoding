% createfigure    Create figures for generic decoding results
%
% Author: Tomoyasu Horikawa <horikawa-t@atr.jp>, Shuntaro C. Aoki <aoki@atr.jp>
%

clear all;

%% Data settings
resultsDir = './results/';
resultsFileFeatPred = fullfile(resultsDir, 'FeaturePrediction.mat');
resultsFileCatIdent = fullfile(resultsDir, 'CategoryIdentification.mat');

%% Figure settings
fontSize = 5;
lineWidth = 2;
subplotMargin = 0.13;

figureProperties = {'Color', 'white'};
axesProperties = {'Box', 'off', ...
                  'TickDir', 'out'};

%% Load results
resFeatPred = load(resultsFileFeatPred);
resCatIdent = load(resultsFileCatIdent);

subjectList = unique({resFeatPred.results(:).subject});
%roiList     = unique({resFeatPred.results(:).roi});
roiList     = {'V1', 'V2', 'V3', 'V4', 'FFA', 'LOC', 'PPA', 'LVC', 'HVC',  'VC'};
featureList = unique({resFeatPred.results(:).feature});

%% Realign results to a 3D array (subject x ROI x feature)
results = [];
for i = 1:length(subjectList)
for j = 1:length(roiList)
for k = 1:length(featureList)
    fInd = strcmp({resFeatPred.results(:).subject}, subjectList{i}) ...
           & strcmp({resFeatPred.results(:).roi}, roiList{j}) ...
           & strcmp({resFeatPred.results(:).feature}, featureList{k});
    cInd = strcmp({resCatIdent.results(:).subject}, subjectList{i}) ...
           & strcmp({resCatIdent.results(:).roi}, roiList{j}) ...
           & strcmp({resCatIdent.results(:).feature}, featureList{k});

    results.featPred.image.perception(i, j, k)    = resFeatPred.results(fInd).predaccImagePercept;
    results.featPred.category.perception(i, j, k) = resFeatPred.results(fInd).predaccCategoryPercept;
    results.featPred.category.imagery(i, j, k)    = resFeatPred.results(fInd).predaccCategoryImagery;

    results.catIdent.perception(i, j, k) = resCatIdent.results(cInd).correctRatePerceptAve;
    results.catIdent.imagery(i, j, k)    = resCatIdent.results(cInd).correctRateImageryAve;
end
end
end

%% Visualize results: feature decoding accuracy
%  Seen image feature & imagined category-average feature decoding accuracy

dataType = {'seen:image', 'seen:category', 'imagined:category'};

% Figure settings
numRow = 14;
numCol = 6;
add = 3;
[plotOrder, numRow, numCol] = get_subplot_order([numRow, numCol], 'lbu', [3, 0]);

% Visualize results
hf = makefigure('fullscreen');
set(hf, figureProperties{:});

cnt = 0;
for iData = 1:length(dataType)

    numSbj = size(results.featPred.image.perception, 1);
    
    switch dataType{iData}
      case 'seen:image'
        col = cmap4('bg4');
        dat = results.featPred.image.perception;
        range = [-0.2, 0.6];
        yax = -0.2:0.2:0.6;
      case 'seen:category'
        col = cmap4('bg4');
        dat = results.featPred.category.perception;
        range = [-0.2,0.6];
        yax = -0.2:0.2:0.6;
      case 'imagined:category'
        col = cmap4('ibg4');
        dat = results.featPred.category.imagery;
        range = [-0.2, 0.4];
        yax = -0.2:0.2:0.4;
    end

    % Calculate mean and confidence interval
    mu = squeeze(mean(dat, 1));
    ci = tinv(0.95, numSbj - 1) .* squeeze(std(dat, [], 1)) ./ sqrt(numSbj);
    
    for ix = 1:length(featureList)
        cnt = 1 + cnt;
        plotIndex = cnt + (iData - 1) * add;
        ha = subplottight(numRow, numCol, plotOrder(plotIndex), subplotMargin);
        set(ha, 'FontSize', fontSize);
        hold on;

        % Draw data
        bar(ha, mu(:, ix), ...
            'facecolor', col{1}, ...
            'edgecolor', 'none', ...
            'LineWidth', lineWidth);
        errorbar_h(ha, mu(:, ix), ci(:, ix), '.k');

        % Draw horizontal lines
        hline(yax, '-k');

        % Draw text
        text(1, -0.1, ...
             sprintf('%s; %s', dataType{iData}, featureList{ix}), ...
             'FontSize', fontSize);

        % x and y axis
        % Draw axis labels only on plots at the bottom of the figure
        if ix == 1 || ix == 9
            draw_axes_label(roiList, 1);
        else
            set(ha, 'XTickLabel', '');
        end
        xlim([0.5, length(roiList) + 0.5]);

        ylabel('Corr. coeff.');
        draw_axes_label(yax, 2, yax);
        ylim(range);

        % Set axes parameters
        set(ha, axesProperties{:});
    end
end

suptitle(sprintf('Seen image feature and seen/imagined category-average feature decoding accuracy'));

savefigure(hf, fullfile(resultsDir, 'FeaturePredictionAccuracy.pdf'));

%% Visualize results: identification accuracy

dataType = {'seen', 'imagined'};

% figure settings
numRow = 14;
numCol = 6;
add = 3;
[plotOrder, numRow, numCol] = get_subplot_order([numRow, numCol], 'lbu', [3, 1]);

% Visualize results
hf = makefigure('fullscreen');
set(hf, figureProperties{:});

cnt = 0;
for iData = 1:length(dataType)

    switch dataType{iData}
        case 'seen'
            col = cmap4('bg4');
            dat = results.catIdent.perception;
        case 'imagined'
            col = cmap4('ibg4');
            dat = results.catIdent.imagery;
    end

    % Calculate mean and confidence interval
    mu = squeeze(mean(dat, 1));
    ci = tinv(0.95, numSbj - 1) .* squeeze(std(dat, [], 1)) ./ sqrt(numSbj);
    
    for ix = 1:length(featureList)
        cnt = 1 + cnt;
        plotIndex = cnt + (iData - 1) * add;
        ha = subplottight(numRow, numCol, plotOrder(plotIndex), subplotMargin);
        set(ha, 'FontSize', fontSize);
        hold on;

        % Draw data
        bar(ha, mu(:, ix) * 100, ...
            'facecolor', col{1}, ...
            'edgecolor', 'none', ...
            'LineWidth', lineWidth);
        errorbar_h(ha, mu(:, ix) * 100, ci(:, ix) * 100, '.k');

        % Draw horizontal lines
        hline(50, '-k')
        hl = hline(60:10:100, '-');
        set(hl, 'Color', col{1});

        % Draw text
        text(1, 95, ...
             sprintf('%s: %s', dataType{iData}, featureList{ix}), ...
             'FontSize', fontSize);

        % x and y axis
        if ix == 1 || ix == 9
            draw_axes_label(roiList, 1);
        else
            set(ha, 'XTickLabel', '');
        end
        xlim([0.5, length(roiList) + 0.5]);

        ylabel('Accuracy (%)');
        ylim([40, 100]);

        
        % Set axes parameters
        set(ha, axesProperties{:});
    end
end % image category

suptitle(sprintf('Seen and imagined category identification accuracy'));

savefigure(hf, fullfile(resultsDir, 'IdentificationAccuracy.pdf'));
