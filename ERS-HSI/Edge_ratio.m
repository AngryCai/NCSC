function [Ratio]=Edge_ratio(img)
 [m,n] = size(img);
%  img =  rgb2gray(img);
 BW = edge(img,'log');
%  figure,imshow(BW)
 ind = find(BW~=0);
 Len = length(ind);
 Ratio = Len/(m*n);
end