% WBC Segmentation using K-Means on LAB color space
% Quick demo for leukemia detection project

clc;
clear;
close all;

% load the blood smear image
[file, path] = uigetfile({'*.jpg;*.tif;*.png', 'Image Files'}, 'Pick a blood image');
if isequal(file, 0)
    disp('No file selected, exiting...');
    return;
end

img = imread(fullfile(path, file));
img = imresize(img, 0.5);

fig = figure('Name', 'WBC Segmentation', 'NumberTitle', 'off', 'Color', 'w');
subplot(2, 3, 1);
imshow(img);
title('Input');
box on;

% convert to LAB and grab a*b* channels
labImg = rgb2lab(img);
ab = double(labImg(:, :, 2:3));
[rows, cols] = size(ab(:, :, 1));
ab_vec = reshape(ab, rows * cols, 2);

subplot(2, 3, 2);
imshow(labImg(:, :, 2), []);
title('a* channel');
box on;
subplot(2, 3, 3);
imshow(labImg(:, :, 3), []);
title('b* channel');
box on;

% kmeans clustering - 3 clusters usually works well
fprintf('Running kmeans... ');
numClusters = 3;
[idx, centers] = kmeans(ab_vec, numClusters, 'Distance', 'sqEuclidean', 'Replicates', 3);
labels = reshape(idx, rows, cols);
fprintf('done\n');

% figure out which cluster is the WBCs (usually darkest in L channel)
L = labImg(:, :, 1);
avgL = zeros(1, numClusters);
for i = 1:numClusters
    avgL(i) = mean(L(labels == i));
end
[minVal, wbcCluster] = min(avgL);
wbcMask = (labels == wbcCluster);

subplot(2, 3, 4);
imshow(labels, []);
title(sprintf('Clusters (WBC=#%d)', wbcCluster));
box on;

% clean up the mask with morphology
se = strel('disk', 3);
cleanMask = imopen(wbcMask, se);
cleanMask = imfill(cleanMask, 'holes');
cleanMask = bwareaopen(cleanMask, 200);

subplot(2, 3, 5);
imshow(cleanMask);
title('Cleaned mask');
box on;

% overlay detected boundaries on original
[boundaries, L2] = bwboundaries(cleanMask, 'noholes');
subplot(2, 3, 6);
imshow(img);
hold on;
for i = 1:length(boundaries)
    b = boundaries{i};
    plot(b(:, 2), b(:, 1), 'y', 'LineWidth', 2);
end
title(sprintf('Found %d WBCs', length(boundaries)));
box on;
hold off;

disp('Done!');
