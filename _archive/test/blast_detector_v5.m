% Blast Cell Detector V5 (K-Means + L1 Features)
% Combines V1's robust K-Means segmentation with V4's L1-tuned classification

clc;
clear;
close all;

% --- Parameters ---
RESIZE_FACTOR = 0.5;
MIN_NUC_AREA = 500;
BLAST_SCORE_CUTOFF = 3.2;  % L1 tuning (increased to reduce false positives)

% --- Load Image ---
[file, path] = uigetfile({'*.jpg;*.tif;*.png;*.bmp', 'Image Files'}, 'Pick a Blood Smear Image');
if isequal(file, 0), return; end
fullPath = fullfile(path, file);
img = imread(fullPath);
img = imresize(img, RESIZE_FACTOR);
[rows, cols, ~] = size(img);

fprintf('Processing: %s\n', file);

% =========================================================================
% SEGMENTATION (K-Means - Proven from V1)
% =========================================================================

labImg = rgb2lab(img);
ab = double(labImg(:, :, 2:3));
ab_vec = reshape(ab, rows * cols, 2);

% K-Means with 3 clusters
numClusters = 3;
[idx, centers] = kmeans(ab_vec, numClusters, 'Distance', 'sqEuclidean', 'Replicates', 3);
labels = reshape(idx, rows, cols);

% Find nucleus cluster (darkest in L channel)
L = labImg(:, :, 1);
avgL = zeros(1, numClusters);
for i = 1:numClusters
    avgL(i) = mean(L(labels == i));
end
[~, nucCluster] = min(avgL);

% Create nucleus mask
mask_nuc = (labels == nucCluster);

% Morphological cleanup
mask_nuc = imopen(mask_nuc, strel('disk', 2));
mask_nuc = imfill(mask_nuc, 'holes');
mask_nuc = bwareaopen(mask_nuc, MIN_NUC_AREA);

% =========================================================================
% FEATURE EXTRACTION & L1-TUNED CLASSIFICATION
% =========================================================================

stats = regionprops(mask_nuc, 'BoundingBox', 'Area', 'Perimeter', 'Eccentricity', 'Image');
grayImg = rgb2gray(img);

figure('Name', 'Blast Detector V5 (K-Means + L1)', 'Color', 'w');
subplot(1, 2, 1); imshow(img); title('Original');
subplot(1, 2, 2); imshow(img); hold on; title('Detection Results');

fprintf('\n--- Nucleus Analysis (L1 Blast Detection) ---\n');
fprintf('ID\tArea\tCirc.\tEcc.\tHom.\tSCORE\tClass\n');
fprintf('--------------------------------------------------------\n');

blast_count = 0;
normal_count = 0;

for i = 1:length(stats)
    area = stats(i).Area;
    perim = stats(i).Perimeter;
    ecc = stats(i).Eccentricity;
    bbox = stats(i).BoundingBox;
    
    % Circularity
    circ = (4 * pi * area) / (perim^2 + eps);
    
    % Texture (GLCM)
    r1 = max(1, floor(bbox(2))); r2 = min(rows, ceil(bbox(2)+bbox(4)));
    c1 = max(1, floor(bbox(1))); c2 = min(cols, ceil(bbox(1)+bbox(3)));
    
    roi_gray = grayImg(r1:r2, c1:c2);
    local_mask = stats(i).Image;
    
    [lr, lc] = size(local_mask);
    [rr, rc] = size(roi_gray);
    if lr ~= rr || lc ~= rc
        roi_gray = roi_gray(1:min(lr,rr), 1:min(lc,rc));
        local_mask = local_mask(1:min(lr,rr), 1:min(lc,rc));
    end
    
    roi_gray(~local_mask) = 0;
    
    try
        glcm = graycomatrix(roi_gray, 'NumLevels', 16, 'Symmetric', true, 'Offset', [0 1]);
        props = graycoprops(glcm, 'Homogeneity');
        homogeneity = props.Homogeneity;
    catch
        homogeneity = 0.5;
    end
    
    % --- L1 SCORING ---
    % L1: Small, Round, Homogeneous
    s_area = min(area / 1500, 1.2);
    s_circ = circ;
    s_tex = homogeneity;
    
    % Weighted for L1
    total_score = (s_area * 1.0) + (s_circ * 1.5) + (s_tex * 1.2);
    
    % Hard filter for irregular shapes
    if ecc > 0.85
        is_blast = false;
        cls_str = 'Debris';
    else
        if total_score > BLAST_SCORE_CUTOFF
            is_blast = true;
            cls_str = 'BLAST (L1)';
            blast_count = blast_count + 1;
        else
            is_blast = false;
            cls_str = 'Normal';
            normal_count = normal_count + 1;
        end
    end
    
    % Visualization
    if is_blast
        rectangle('Position', bbox, 'EdgeColor', 'r', 'LineWidth', 2);
        text(bbox(1), bbox(2)-5, sprintf('BLAST %.1f', total_score), 'Color', 'r', 'FontSize', 8, 'FontWeight', 'bold');
    elseif strcmp(cls_str, 'Normal')
        rectangle('Position', bbox, 'EdgeColor', 'g', 'LineWidth', 1);
    end
    
    fprintf('%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n', ...
        i, area, circ, ecc, homogeneity, total_score, cls_str);
end

hold off;

% Summary
fprintf('\n--- Summary ---\n');
fprintf('Total Cells: %d\n', length(stats));
fprintf('Suspected Blasts: %d\n', blast_count);
fprintf('Normal WBCs: %d\n', normal_count);

if blast_count > 0
    msgbox(sprintf('ALERT: %d Suspected Blast Cells Detected!', blast_count), 'Screening Result', 'warn');
else
    msgbox('No obvious blasts detected.', 'Screening Result', 'help');
end
