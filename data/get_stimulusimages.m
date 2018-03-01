function images = get_stimulusimages(username, accessKey)
% get_stimulusimages    Get stimulus images from ImageNet
% 
% Note:
%
% This script is for downloading all 1200 training and 50 test images of
% experimental stimulus used in Horikawa and Kamitani 2017
% "Generic decoding of seen and imagined objects using hierarchical visual features".
% It will take about one day to complete all processes with single computer,
% but you can accerelate by using multiple computers in parallel.
%
% This function requires `websave`, which is equipped in MATLAB >= 2014b.
%
% Inputs:
%
% - username  [char] : ImageNet user name
% - accessKey [char] : ImageNet access key
%
% Outputs:
%
% - images [struct] : A structure contains image information
%
% Settings:
%
% - imageListFiles    : List of CSV files which contains image IDs and image file names
% - imageType         : List of strings representing image types
% - imageDir          : Directory in which images will be saved
% - imageSize         : The images will be trimmed down with this size (pixel)
% - deleteOriginalImg : If true, remove the downloaded original image files
%


%% Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% List of CSV files which contains image IDs and image file names
imageListFiles = {'./imageID_training.csv', ...
                  './imageID_test.csv'};

% Type of images ('training' or 'test')
% This should correspond to 'imageListFiles'
imageType = {'training', ...
             'test'};

% The directory in which images will be saved
imageDir = './images';

% Image trimming size (500 px)
imageSize = 500;

% Delete the downloaded original images or not
deleteOriginalImg = false;

% Misc options for downloading (websave)
options = weboptions;
options.Timeout = 100;


%% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('%s started\n', mfilename);

urlImageNet = 'http://image-net.org/download/synset';

%% Check inputs ----------------------------------------------------------------
if ~exist('username','var')
    error('Please give your ImageNet username name.');
end

if ~exist('accessKey','var')
    error('Please give your ImageNet access key.');
end

%% Get image lists -------------------------------------------------------------
for i = 1:length(imageListFiles)
    fid = fopen(imageListFiles{i}, 'r');
    imageList = textscan(fid, '%n%s', 'delimiter', ',');
    fclose(fid);

    % Remove quotations from file names
    for j = 1:length(imageList{2})
        imageList{2}{j} = imageList{2}{j}(2:end-1);
    end
    
    images(i).type     = imageType{i};
    images(i).listFile = imageListFiles{1};
    images(i).id       = imageList{1};
    images(i).files    = imageList{2};

    %% Get image files for each synset
    images(i).synsets = [];
    synsetList = {};
    for j = 1:length(images(i).files)
        imgFileName = images(i).files{j};
        synsetId = regexp(imgFileName, '^(n\d+)_\d+\.JPEG$', 'tokens');
        synsetId = synsetId{1}{1};
         
        [ismem, ind] = ismember(synsetId, synsetList);

        if ismem
            images(i).synsets(ind).files{end+1} = imgFileName;
        else
            images(i).synsets(end+1).name = synsetId;
            images(i).synsets(end).files{1} = imgFileName;

            synsetList{end+1} = synsetId;
        end
    end

    images(i).synsetList = unique(synsetList);
end

%% Download and convert images -------------------------------------------------
for i = 1:length(images)
for j = 1:length(images(i).synsets)

    %% Multiprocessing
    imgType = images(i).type;
    synsetId = images(i).synsets(j).name;

    lockFile = sprintf('%s_%s_%s.lock', mfilename, imgType, synsetId);
    if exist(lockFile,'file')
        continue;
    end
    fclose(fopen(lockFile, 'w'));

    fprintf('Image type:\t%s\n', imgType);
    fprintf('Synset:\t\t%s\n', synsetId);
    
    %% Prepare directories
    synsetDir = fullfile(imageDir, imgType, synsetId);
    if ~exist(synsetDir, 'dir')
        fprintf('Create %s\n', synsetDir);
        mkdir(synsetDir);
    end
    
    dlDir = fullfile(imageDir, 'download', synsetId);
    dlFile = fullfile(dlDir, [synsetId '.tar']);

    if exist(dlFile,'file');
        fprintf('%s already exists. Download was skipped.\n', dlFile);
    else
        if ~exist(dlDir, 'dir')
            fprintf('Create %s\n', dlDir);
            mkdir(dlDir);
        end

        %% Download synset images
        fprintf('Downloading %s...', synsetId);
        tic;
        res = websave(dlFile, urlImageNet, ...
                      'wnid', synsetId, ...
                      'username', username, ...
                      'accesskey', accessKey, ...
                      'release', 'latest', ...
                      'src', 'stanford', ...
                      options);
        elapsedTime = toc;
        fprintf('done (%f sec)\n', elapsedTime);

        fprintf('Extracting image files...');
        untar(dlFile, dlDir)
        fprintf('done\n');
    end
    
    %% Copy and trim images
    for k = 1:length(images(i).synsets(j).files)
        imgFile = images(i).synsets(j).files{k};
        sourceFile = fullfile(imageDir, 'download', synsetId, imgFile);
        targetFile = fullfile(synsetDir, imgFile);

        if exist(targetFile,'file')
            continue;
        end
        
        fprintf('Converting %s\n', imgFile);
        
        while ~exist(sourceFile,'file')
            % Wait until 'sourceFile' is created by another process
            pause(1);
        end
        
        img = imread(sourceFile);

        imgTrim = trimImgCenter(img);
        imgResize = imresize(imgTrim, [imageSize, imageSize]);

        imwrite(imgResize, targetFile);

        fprintf('%s saved\n', targetFile);
    end

    delete(lockFile);
end
end

%% Remove the downloaded original images
if deleteOriginalImg
    rmdir(fullfile(imageDir, 'download'), 's');
end

fprintf('Finished\n');

end


%% Subfunctions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [trimmed] = trimImgCenter(img)
% trimimgCenter    Trims an image on the center
%
% Input:
%
% - img: image
%
% Output:
%
% - trimmed: trimmed image

[r, c, d] = size(img);
trimSize = min([r, c]);
lr = floor((c - trimSize) / 2);
ud = floor((r - trimSize) / 2);

trimmed = img((1:min([(trimSize), r - ud])) + ud, ...
              (1:min([(trimSize), c - lr])) + lr, :);

end
