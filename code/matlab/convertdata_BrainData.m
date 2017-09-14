% convertdata_BrainData    Convert GOD data to BrainDecoderToolbox 2 format
%
% Author: Shuntaro C. Aoki <aoki@atr.jp>
%

clear all;

%% Brain data
dataStore = './god2016/data/fmri';
saveDir = './data';

subjects = {'Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5'};
rois = {'V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC'};

%% Label data
imgLabelTrain = './data/images/image_training.csv';
imgLabelTest = './data/images/image_test.csv';

imgTrain = textread(imgLabelTrain, '%s');
imgTest =  textread(imgLabelTest, '%s');

% Convert image file names to image IDs
for i = 1:length(imgTrain)
    idcell = strsplit(strrep(strrep(imgTrain{i}, '.JPEG', ''), 'n', '0'), '_');
    imgIdTrainStr{i, 1} = sprintf('%s.%06s', idcell{1}, idcell{2});
end
for i = 1:length(imgTest)
    idcell = strsplit(strrep(strrep(imgTest{i}, '.JPEG', ''), 'n', '0'), '_');
    imgIdTestStr{i, 1} = sprintf('%s.%06s', idcell{1}, idcell{2});
end

imgIdTrain = cellfun(@str2num, imgIdTrainStr);
imgIdTest = cellfun(@str2num, imgIdTestStr);

% Save label txt
fid = fopen('./data/images/image_training_id.csv', 'wt');
for i = 1:size(imgIdTrain, 1)
    fprintf(fid, '%f,"%s"\n', imgIdTrain(i), imgTrain{i});
end
fclose(fid);

fid = fopen('./data/images/image_test_id.csv', 'wt');
for i = 1:size(imgIdTest, 1)
    fprintf(fid, '%f,"%s"\n', imgIdTest(i), imgTest{i});
end
fclose(fid);

%% Load data
for iSbj = 1:length(subjects)
    sbj = subjects{iSbj};

    saveFileName = fullfile(saveDir, sprintf('%s.mat', sbj));

    if exist(saveFileName, 'file')
        fprintf('%s is already exist\n', saveFileName);
        continue;
    end
    
    dataType = [];
    runs = [];
    labels = [];
    brainData = [];
    voxelXyz  = [];
    volInds = [];
    roiName = {};
    roiFlag = [];
    
    for iRoi = 1:length(rois)
        roi = rois{iRoi};

        % Load data
        dataFileTrain = fullfile(dataStore, sbj, 'train', sprintf('%s.mat', roi));
        dataFileTestPercept = fullfile(dataStore, sbj, 'test_perception', sprintf('%s.mat', roi));
        dataFileTestImagine = fullfile(dataStore, sbj, 'test_imagery', sprintf('%s.mat', roi));

        dTrain = load(dataFileTrain);
        dTestPercept = load(dataFileTestPercept);
        dTestImagine = load(dataFileTestImagine);

        [nSmpTr, nVoxTr] = size(dTrain.D.data);
        [nSmpTeP, nVoxTeP] = size(dTestPercept.D.data);
        [nSmpTeI, nVoxTeI] = size(dTestImagine.D.data);

        if ~isequal(nVoxTr, nVoxTeP, nVoxTeI)
            error('Num voxels inconsistent');
        else
            nVox = nVoxTr;
        end
        
        % ROI flags
        roiName = { roiName{:}, roi };
        roiFlag = [ roiFlag,                  nan(size(roiFlag, 1), nVox);
                    nan(1, size(roiFlag, 2)), ones(1, nVox) ];

        % Voxel xyz
        if ~isequal(dTrain.D.xyz, dTestPercept.D.xyz, dTestImagine.D.xyz)
            error('Voxle coordinates inconsistent');
        else
            voxelXyz = [ voxelXyz, dTrain.D.xyz ];
        end

        % VolInds
        if ~isequal(dTrain.D.volInds, dTestPercept.D.volInds, dTestImagine.D.volInds)
            error('Voxle coordinates inconsistent');
        else
            volInds = [ volInds, dTrain.D.volInds ];
        end

        % DataType
        dataTypeTmp = [ ones(nSmpTr, 1);
                     ones(nSmpTeP, 1) * 2;
                     ones(nSmpTeI, 1) * 3 ];
        if isempty(dataType)
            dataType = dataTypeTmp;
        elseif ~isequal(dataType, dataTypeTmp)
            error('Data type inconsistent');
        end

        % Run
        runsTmp = [ convert_index(dTrain.D.inds_runs)';
                    convert_index(dTestPercept.D.inds_runs)';
                    convert_index(dTestImagine.D.inds_runs)' ];
        if isempty(runs)
            runs = runsTmp;
        elseif ~isequal(runs, runsTmp)
            error('Run number inconsistent');
        end

        % Label
        %labelTrain = dTrain.D.labels;
        %labelTestPercept = dTestPercept.D.labels;
        %labelTestImagine = dTestImagine.D.labels;

        labelsTmp = [ imgIdTrain(dTrain.D.labels);
                      imgIdTest(dTestPercept.D.labels);
                      imgIdTest(dTestImagine.D.labels) ];

        if isempty(labels)
            labels = labelsTmp;
        elseif ~isequal(labels, labelsTmp)
            error('Label inconsistent');
        end

        brainDataTmp = [ dTrain.D.data;
                         dTestPercept.D.data;
                         dTestImagine.D.data ];
        
        brainData = [ brainData, brainDataTmp ];
    end

    % Create dataSet+metaData
    [dataSet, metaData] = initialize_dataset();

    [dataSet, metaData] = add_dataset(dataSet, metaData, ...
                                      dataType, ...
                                      'DataType', ...
                                      '1 = Data type (1 = Training data; 2 = Perception test data; 3 = Imagery test data');
    [dataSet, metaData] = add_dataset(dataSet, metaData, ...
                                      runs, ...
                                      'Run', ...
                                      '1 = Run number');
    [dataSet, metaData] = add_dataset(dataSet, metaData, ...
                                      labels, ...
                                      'Label', ...
                                      '1 = Label (image ID)');
    [dataSet, metaData] = add_dataset(dataSet, metaData, ...
                                      brainData, ...
                                      'VoxelData', ...
                                      '1 = Voxel data');

    metaData = add_voxelxyz(metaData, voxelXyz, 'VoxelData');
    metaData = add_metadata(metaData, 'VolInds', '1 = Volume index', volInds, 'VoxelData');

    for n = 1:length(roiName)
        metaData = add_metadata(metaData, ...
                                sprintf('ROI_%s', roiName{n}), ...
                                sprintf('1 = ROI %s', roiName{n}), ...
                                roiFlag(n, :), ...
                                'VoxelData');
    end

    % Remove redundant voxels
    [dataSet, metaData] = remove_reduncol(dataSet, metaData, ...
                                          {'VoxelData', 'voxel_x', 'voxel_y', 'voxel_z', 'VolInds'});
    
    % Save data
    subject = sbj;
    rawData = '';
    createScript = mfilename;
    createDate = sprintf(datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    createdBy = 'Dpt. Neuroinformatics, ATR';

    save(saveFileName, ...
         'dataSet', 'metaData', ...
         'subject', 'rawData', ...
         'createScript', 'createDate', 'createdBy');
    
end
