# White-Matter-Hyperintensity-Segmentation
Complete Process of White Matter Hyperintensities (WMH) Segmentation and Quantitative Analysis
Results Display:
<img width="1205" height="911" alt="1765707632622" src="https://github.com/user-attachments/assets/d7eaf5b8-a3f2-4108-9172-1cc2449f9131" />



1. Installation Process
Key dependency packages:
wmh_seg: Main package for white matter hyperintensities segmentation
torch & torchvision: PyTorch deep learning framework
nibabel: Medical image file (.nii format) processing
torchio: Medical image data augmentation and preprocessing
segmentation-models-pytorch: Image segmentation models
nilearn: Neuroimaging analysis tools

2. Core Computational Process of WMH Segmentation

2.1 Input Data:
T2-FLAIR brain MRI images (.nii format)
Dimensions: 174×240×202 voxels
Voxel size: 1.0 mm³

2.2 Segmentation Algorithm Workflow:
python
# Simplified algorithm workflow
def wmh_segmentation_workflow(nii_image):

    # 1. Preprocessing
    image = normalize_intensity(nii_image)  # Intensity normalization
    image = resize_to_standard(image)       # Resampling to standard space
    
    # 2. Axial Slice Segmentation (UNet architecture)
    for slice_idx in range(image.shape[2]):
        # Using pre-trained UNet model
        axial_slice = extract_slice(image, axis=2, idx=slice_idx)
        prediction = unet_model(axial_slice)  # Deep learning segmentation
        
    # 3. Post-processing
    segmentation = threshold_predictions(predictions)  # Thresholding
    segmentation = remove_small_lesions(segmentation)  # Remove small lesions
    segmentation = fill_holes(segmentation)           # Hole filling
    
    return segmentation

Quantitative Analysis Formulas

3.1 WMH Volume Calculation:
    # Total WMH volume = WMH voxel count × voxel volume
    wmh_volume_mm3 = wmh_voxels * voxel_volume_mm3
    wmh_volume_ml = wmh_volume_mm3 / 1000  # Convert to milliliters
    
3.2 Lesion Count Statistics:    
    # Use connected component analysis to label independent lesions
    from scipy.ndimage import label
    labeled_array, num_features = label(wmh_mask)
    # Lesion size classification criteria: small lesions: <10 voxels (<10 mm³), medium lesions: 10-99 voxels, large lesions: ≥100 voxels.

3.3 Spatial Distribution Analysis:
    # Left and right hemisphere analysis (assuming the image is standardized)
    midline = image_shape[1] // 2
    left_hemisphere_voxels = np.sum(wmh_mask[:, :midline, :])
    right_hemisphere_voxels = np.sum(wmh_mask[:, midline:, :])

4. Interpretation of Output Results

WMH features of the current case:
Total WMH volume: 7.35 ml
Lesion count: 123
Lesion distribution: small lesions: 50 (40.7%), medium lesions: 63 (51.2%), large lesions: 10 (8.1%)
Spatial distribution: left hemisphere: 52.5%, right hemisphere: 47.5%
Clinical significance: WMH percentage: 0.087% (percentage of white matter volume), primary affected region: Z=116 layer (containing 423 WMH voxels), largest lesion: 1081 mm³.

5. Technical Points
Model architecture: backbone network: MIT-B5 (Mix Transformer), segmentation head: UNet decoder
Performance metrics: processing time: ~64 seconds (approximately 1 second per slice)
This workflow demonstrates a complete medical image analysis process, from data preparation and deep learning segmentation to clinical quantitative analysis, providing important indicators for evaluating neurodegenerative diseases (such as cerebrovascular disease, Alzheimer's disease, etc.).
    
