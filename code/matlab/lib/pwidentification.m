function cr = pwidentification(simmat, labels)
% pwidentification    Perform pairwise identification from similarity matrix
%
% Inputs:
%
% - simmat : Similarity matrix [pred x correct]
% - labels : Index matrix
% 
% Outputs:
%
% - cr : Correct rate
% 
% 
% Author: Tomoyasu Horikawa <horikawa-t@atr.jp>, Shuntaro C. Aoki <aoki@atr.jp>
% Created: 2016-09-10
% Modified: 2016-10-20
% 


% Fix labels to a vertical vector
if size(labels, 1) == 1
    labels = labels';
end

% Remove NaN
if any(any(isnan(simmat), 1))
    simmat(:, any(isnan(simmat), 1)) = [];
end

% Get num of candidate
numCandidate = size(simmat, 2) - 1;

% Sort simmat for each prediction
[sortedSimmat, order] = sort(simmat, 2, 'descend');

% Get num of incorrect
[labelList, numIncorrect] = find(~bsxfun(@minus, order, labels));
numIncorrect = numIncorrect - 1;

% Get index of the original label list.
[sortedLabelList, ind] = sort(labelList, 'ascend');

% Calculate correct rate
cr = (numCandidate - numIncorrect(ind)) ./ numCandidate;
% `cr = 1 - numIncorrect(labelInd) ./ numCandidate` caused slight numerical error
% compared with the original code (~ 10^-16).
