%crops out m x m from the center of larger N x N array

function ROI = croppedOut(largeArray,cropSize,varargin)
%optional input specifying center of crop as [center1 center2 center3]

n = size(largeArray);

if nargin > 2
    nc1 = varargin{1}(1);
    nc2 = varargin{1}(2);
    if length(n) > 2
        nc3 = varargin{1}(3);
        nc = [nc1 nc2 nc3];
    else
        nc = [nc1 nc2];
    end
    
else
    nc = round((n+1)/2);
end



if length(cropSize) == 1
    cropSize = repmat(cropSize,length(n),1);
end
for ii = 1:length(n)
    vec = 1:cropSize(ii);
    cropC = round((cropSize(ii)+1)/2);
    cropVec{ii} = single(vec - cropC + nc(ii));
end

if length(n) == 2
    ROI = largeArray(cropVec{1}, cropVec{2});
elseif length(n) == 3
    ROI = largeArray(cropVec{1}, cropVec{2}, cropVec{3});
end
end
    