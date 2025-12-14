from wmh_seg import wmh_seg
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

path1 = r"C:\Users\15336\Desktop\T2 flair\T2-2115278_age73-WMH9317\T2_FLAIR\T2_FLAIR_unbiased_brain.nii\T2_FLAIR_unbiased_brain.nii"

# 加载nifti文件
nii_img = nib.load(path1)
# 获取图像数据
nii_data = nii_img.get_fdata()

# # 对整个3D图像进行分割
wmh_3d = wmh_seg(nii_data)

# # 获取第50层的切片
# slice_50 = nii_data[:, :, 50]
# # # 对切片进行分割
# wmh_slice = wmh_seg(slice_50)

def generate_segmentation_report(image_data, seg_data, voxel_size_mm=1.0):
    """
    生成详细的3D分割报告
    """
    from scipy import ndimage
    
    print("=" * 60)
    print("WMH 3D SEGMENTATION REPORT")
    print("=" * 60)
    
    # 基本信息
    print(f"Image dimensions: {image_data.shape}")
    print(f"Voxel size: {voxel_size_mm} mm³")
    
    # WMH统计
    total_voxels = np.prod(seg_data.shape)
    wmh_voxels = np.sum(seg_data > 0)
    wmh_percentage = (wmh_voxels / total_voxels) * 100
    wmh_volume_mm3 = wmh_voxels * (voxel_size_mm ** 3)
    wmh_volume_ml = wmh_volume_mm3 / 1000
    
    print(f"\nWMH Statistics:")
    print(f"  Total voxels: {total_voxels}")
    print(f"  WMH voxels: {wmh_voxels}")
    print(f"  WMH percentage: {wmh_percentage:.3f}%")
    print(f"  WMH volume: {wmh_volume_mm3:.2f} mm³ ({wmh_volume_ml:.2f} ml)")
    
    # 连通区域分析
    labeled_array, num_features = ndimage.label(seg_data > 0)
    print(f"\nLesion Analysis:")
    print(f"  Number of WMH lesions: {num_features}")
    
    if num_features > 0:
        sizes = ndimage.sum(seg_data > 0, labeled_array, range(1, num_features+1))
        
        print(f"  Largest lesion size: {np.max(sizes):.0f} voxels "
              f"({np.max(sizes)*(voxel_size_mm**3):.2f} mm³)")
        print(f"  Smallest lesion size: {np.min(sizes):.0f} voxels "
              f"({np.min(sizes)*(voxel_size_mm**3):.2f} mm³)")
        print(f"  Average lesion size: {np.mean(sizes):.1f} voxels "
              f"({np.mean(sizes)*(voxel_size_mm**3):.2f} mm³)")
        print(f"  Median lesion size: {np.median(sizes):.1f} voxels "
              f"({np.median(sizes)*(voxel_size_mm**3):.2f} mm³)")
        
        # 按大小分类
        small = np.sum(sizes < 10)
        medium = np.sum((sizes >= 10) & (sizes < 100))
        large = np.sum(sizes >= 100)
        
        print(f"\nLesion Size Distribution:")
        print(f"  Small (<10 voxels): {small} lesions ({small/num_features*100:.1f}%)")
        print(f"  Medium (10-99 voxels): {medium} lesions ({medium/num_features*100:.1f}%)")
        print(f"  Large (≥100 voxels): {large} lesions ({large/num_features*100:.1f}%)")
    
    # 空间分布
    print(f"\nSpatial Distribution:")
    
    # Z方向分布
    z_distribution = np.sum(seg_data > 0, axis=(0, 1))
    max_z = np.argmax(z_distribution)
    print(f"  Most affected slice (Z): {max_z} "
          f"({z_distribution[max_z]} voxels)")
    
    # 左右分布
    mid_x = seg_data.shape[0] // 2
    left_hemi = np.sum(seg_data[:mid_x, :, :] > 0)
    right_hemi = np.sum(seg_data[mid_x:, :, :] > 0)
    print(f"  Left hemisphere: {left_hemi} voxels "
          f"({left_hemi/wmh_voxels*100:.1f}%)")
    print(f"  Right hemisphere: {right_hemi} voxels "
          f"({right_hemi/wmh_voxels*100:.1f}%)")
    
    print("\n" + "=" * 60)

# 生成报告
generate_segmentation_report(nii_data, wmh_3d, voxel_size_mm=1.0)


# 保存为 ITK-SNAP 可读格式
wmh_img = nib.Nifti1Image(wmh_3d.astype(np.uint8), nii_img.affine)
nib.save(wmh_img, 'wmh_segmentation.nii.gz')
print("分割结果已保存为 'wmh_segmentation.nii.gz'")
print("可以使用 ITK-SNAP 打开原始图像和分割结果进行查看")