
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_hist_match(source, template):
    """
    Perform histogram matching on the source image using the template image.
    Crucial fallback if Macenko normalization fails.
    Ignores [0,0,0] background pixels to prevent skewing.
    """
    old_shape = source.shape
    source_flat = source.reshape(-1, 3)
    template_flat = template.reshape(-1, 3)
    
    matched = np.zeros_like(source_flat)
    
    # Create masks for non-black pixels
    s_mask = np.any(source_flat > 0, axis=1)
    t_mask = np.any(template_flat > 0, axis=1)
    
    # If no foreground, return original
    if not np.any(s_mask) or not np.any(t_mask):
        return source
        
    s_foreground = source_flat[s_mask]
    t_foreground = template_flat[t_mask]
    
    matched_foreground = np.zeros_like(s_foreground)

    for i in range(3):
        # Calculate histograms on FOREGROUND only
        s_values, bin_idx, s_counts = np.unique(s_foreground[:, i], return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(t_foreground[:, i], return_counts=True)
        
        # Calculate CDFs
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        
        # Interpolate
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        matched_foreground[:, i] = interp_t_values[bin_idx]
        
    # Reconstruct: Put matched foreground back into the black background
    matched[s_mask] = matched_foreground
    
    return matched.reshape(old_shape).astype(np.uint8)

def normalize_staining(img, save_file=None, template_img=None, Io=240, alpha=1, beta=0.15):
    """
    Normalize staining appearance of H&E stained images.
    
    Args:
        img: Input image (RGB)
        save_file: Path to vectors.npz file containing reference stain vectors. 
        template_img: (Optional) Template image to use for Histogram Matching fallback.
    """
    if save_file is None:
        if template_img is not None:
            # Fallback to Histogram Matching
            return normalize_hist_match(img, template_img)
        logger.warning("No reference provided (file or template). Returning original.")
        return img

    try:
        # Load reference vectors
        data = np.load(save_file)
        HERef = data['HERef']
        maxCRef = data['maxCRef']
        
        h, w, c = img.shape
        img = img.reshape((-1, 3))

        # Calculate OD
        OD = -np.log((img.astype(np.float32) + 1) / Io)
        
        # Remove data with insufficient OD
        ODhat = OD[~np.any(OD < beta, axis=1)]
        
        # Compute eigenvectors
        if len(ODhat) < 10:
            logger.warning("Image has too little stain content for Macenko. Returning original.")
            return img.reshape(h, w, c)

        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Extract stain vectors (HE)
        # Project on the plane spanned by the eigenvectors corresponding to the two 
        # largest eigenvalues    
        That = ODhat.dot(eigvecs[:, 1:3])
        
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        
        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        
        # Heuristic to order Hematoxylin first, Eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T
            
        # Rows correspond to channels (RGB), columns to stains (H, E)
        Y = np.reshape(OD, (-1, 3)).T
        
        # Determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        
        # Normalize stain concentrations
        maxC = np.percentile(C, 99, axis=1)
        tmp = np.divide(maxC, maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])
        
        # Recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        Inorm[Inorm > 255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
        
        return Inorm

    except Exception as e:
        logger.warning(f"Macenko normalization failed ({e}). Attempting Histogram Matching.")
        if template_img is not None:
            try:
                return normalize_hist_match(img, template_img)
            except Exception as e2:
                logger.error(f"Histogram matching also failed: {e2}")
        return img.reshape(h, w, c)


def resize_with_padding(image, target_size=(224, 224)):
    """
    Resize image to target_size while maintaining aspect ratio using padding.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Use reflect padding for smoother edges
    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT, value=color)
    
    return padded


def preprocess_input(image):
    """
    Standard preprocessing for MobileNetV2
    Input: RGB image (uint8)
    Output: Normalized float32 array (0-1)
    """
    img = image.astype(np.float32)
    img /= 255.0
    return img

def mask_background_kmeans(image, k=2):
    """
    Masks the background using K-Means clustering (K=2) on LAB color space.
    Assumes the cell is the 'darker' or 'more colorful' cluster compared to the background.
    """
    # Convert to LAB for better color/luminance separation
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    pixel_values = lab.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Determine background cluster
        # Heuristic: Background is usually lighter (Higher L value)
        l_centers = centers[:, 0]
        bg_cluster = np.argmax(l_centers)
        cell_cluster = np.argmin(l_centers)
        
        # Create mask
        labels = labels.flatten()
        mask = (labels == cell_cluster).astype(np.uint8) * 255
        mask = mask.reshape(image.shape[:2])
        
        # Morphological cleanup
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Apply mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
        
    except Exception as e:
        logger.error(f"K-Means masking failed: {e}. Returning original.")
        return image
