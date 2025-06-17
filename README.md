## OMT-APC Model
  OMT and tensor SVD based deep learning model for segmentation and predicting genetic markers of glioma.

  A key technical challenge in deep learning for tumor region segmentation is the GPU memory limitation caused by the large size of MRI brain images. Traditional approaches, such as random cropping of raw brain images, often risk omitting critical information. We introduced the OMT technique to preserve the global structure of MRI data. OMT transforms MRI brain images into m×m×m tensors with minimal distortion, maintaining the overall 3D structure of the MRI data. The OMT density function enhances the tumor region while preserving the volume of non-tumor regions within the OMT tensor, thereby improving segmentation performance.
