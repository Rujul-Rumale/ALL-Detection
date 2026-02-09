
import cv2
import numpy as np
import argparse
from pathlib import Path

def compute_stain_vectors(img_path, output_path, Io=240, alpha=1, beta=0.15):
    """
    Compute stain vectors from a reference image and save them.
    Reference: Macenko et al. (2009)
    """
    print(f"Processing reference image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    img = img.reshape((-1, 3))

    # Calculate OD
    OD = -np.log((img.astype(np.float32) + 1) / Io)
    
    # Remove data with insufficient OD
    ODhat = OD[~np.any(OD < beta, axis=1)]
    
    # Compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    # Extract stain vectors (HE)
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
    
    # Determine concentrations of the individual stains
    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    
    # Normalize stain concentrations
    maxC = np.percentile(C, 99, axis=1)
    
    print(f"Stain Vectors (HE):\n{HE}")
    print(f"Max Concentrations:\n{maxC}")
    
    np.savez(output_path, HERef=HE, maxCRef=maxC)
    print(f"Saved reference vectors to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help="Path to reference image")
    parser.add_argument('--output', default='stain_vectors.npz', help="Output file")
    args = parser.parse_args()
    
    compute_stain_vectors(args.image, args.output)
