%%  FourierShellCorrelate %%

%computes Fourier shell correlation (in 3D) or Fourier ring correlation (in
%2D) depending upon the size of the objects to compare
%%inputs:
%%  obj1 - first object to compare
%%  obj2 - second object, should be same size as obj1
%%  numBins - number of spatial frequency bins in which to compute average correlation
%%  pixSize - pixel size, used to display FSC with spatial frequency values that match the data

%%outputs:
%%  corrCoeffs - correlation coefficients
%%  invResInd - "inverse resolution indices", equivalent to normalized spatial frequency
%%  meanIntensity - average intensity at each shell

%% Author: AJ Pryor
%% Jianwei (John) Miao Coherent Imaging Group
%% University of California, Los Angeles
%% Copyright (c) 2015. All Rights Reserved.

function [corrCoeffs invResInd meanIntensity] = FourierShellCorrelate(obj1,obj2,numBins,pixSize)
%calculate FFT
k1 = my_fft(obj1);
k2 = my_fft(obj2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%obj1 k-space indices

nc = size(obj1,2)+1;%central pixel
n2 = size(obj1,2)/2;%array radius

if mod(size(obj1,1),2)==0
ncK1 = size(obj1,1)/2+1;%central pixel
n2K1 = ncK1-1;%max radius
vec1 = (-n2K1:n2K1-1)./n2K1;
elseif size(obj1,1)==1
vec1 = 0;
else
ncK1 = (size(obj1,1)+1)/2;%central pixel
n2K1 = ncK1-1;%max radius
vec1 = (-n2K1:n2K1)./n2K1; 
end

if  mod(size(obj1,2),2)==0
ncK2 = size(obj1,2)/2+1;%central pixel
n2K2 = ncK2-1;%max radius
vec2 = (-n2K2:n2K2-1)./n2K2;
elseif size(obj1,2)==1
vec2 = 0;
else
ncK2 = (size(obj1,2)+1)/2;%central pixel
n2K2 = ncK2-1;%max radius
vec2 = (-n2K2:n2K2)./n2K2; 
end

if  mod(size(obj1,3),2)==0
ncK3 = size(obj1,3)/2+1;%central pixel
n2K3 = ncK3-1;%max radius
vec3 = (-n2K3:n2K3-1)./n2K3;
elseif size(obj1,3)==1
vec3 = 0;
else
ncK3 = (size(obj1,3)+1)/2;%central pixel
n2K3 = ncK3-1;%max radius
vec3 = (-n2K3:n2K3)./n2K3; 
end
[Kx Ky Kz] = meshgrid(vec2,vec1,vec3);%grid of Fourier indices
Kmags = sqrt(Kx.^2+Ky.^2+Kz.^2);%take magnitude
%%%%%%%%%%%%%%%%%%%%%%%%%%%%



invResInd = linspace(0,1,numBins+1);%compute spatial frequency bins
corrCoeffs = zeros(1,length(invResInd)-1);
meanIntensity = zeros(1,length(invResInd)-1);

for i = 1:length(invResInd)-1
    ind = (Kmags>=invResInd(i)&Kmags<invResInd(i+1));%indices of reciprocal voxels within frequency range of interest
    normC = sqrt(sum((abs(k1(ind)).^2)).*sum((abs(k2(ind)).^2))); %denominator of FSC, see http://en.wikipedia.org/wiki/Fourier_shell_correlation
    corrVals = (sum(k1(ind).*conj(k2(ind))))./normC; %FSC value
    corrCoeffs(i) = real(corrVals); %take real part, the imaginary part should be essentially 0
    meanIntensity(i) = mean(abs(k1(ind))+abs(k2(ind))); %average intensity within this frequency
end
invResInd(end) = [];
spacing = invResInd(2)-invResInd(1);halfSpacing = spacing./2;
invResInd = invResInd + halfSpacing; %%center bins
if nargin>3 %if user provided a pixel size, compute actual spatial frequency values
 maxInverseResolution = (1./pixSize)/2;
 invResInd = invResInd.*maxInverseResolution;
end
% figure, plot(invResInd,corrCoeffs,'or'),title('FSC'),ylabel('Correlation Coefficient'),xlabel('Spatial Frequency')