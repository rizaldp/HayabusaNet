# HayabusaNet

**HayabusaNet: Hybrid attention-based multiscale fusion CNN for accurate and efficient brain tumor classification in MRI scans**

This repository provides a TensorFlow/Keras implementation of **HayabusaNet**, a lightweight hybrid attention-based multiscale fusion CNN for multiclass brain tumor MRI classification.

The implementation is associated with the following published article:

- **Title:** HayabusaNet: Hybrid attention-based multiscale fusion CNN for accurate and efficient brain tumor classification in MRI scans
- **Authors:** Rizal Dwi Prayogo, Siti Amatullah Karimah, and Hidetaka Nambo
- **Journal:** Expert Systems With Applications
- **Volume / Article:** 331, 133135
- **Year:** 2026
- **DOI:** https://doi.org/10.1016/j.eswa.2026.133135

> This repository is intended for research and reproducibility purposes only. It is not intended for clinical diagnosis or direct medical decision-making.

---

## Overview

HayabusaNet is designed for efficient brain tumor MRI classification by combining:

- **Depthwise separable convolutions** for computational efficiency.
- **Parallel multiscale branches** with 3×3, 5×5, and 7×7 kernels.
- **Element-wise feature fusion** to combine complementary multiscale representations.
- **Hybrid attention** combining CBAM-based local refinement and lightweight self-attention-based global context.
- **Compact classification head** for binary and multiclass MRI classification.

The model classifies brain MRI images into four categories:

```text
glioma
meningioma
notumor
pituitary
```

---

## Dataset

The main experiment in the published paper uses the public Kaggle Brain Tumor MRI dataset:

- **Dataset:** Brain Tumor MRI Dataset
- **Source:** Nickparvar, M. (2021)
- **Kaggle DOI:** https://doi.org/10.34740/KAGGLE/DSV/2645886

The dataset contains four classes:

```text
glioma
meningioma
notumor
pituitary
```

The notebook assumes that the dataset has already been prepared and arranged in directory-based class folders.

### Important Preprocessing Note

The published study includes preprocessing and augmentation steps such as resizing, normalization, and training-set augmentation/balancing. This notebook assumes that the image folders are already prepared.

Images are loaded using TensorFlow's `image_dataset_from_directory`, resized to `224 × 224`, and normalized inside the model using `Rescaling(1./255)`.

---

## Environment

The published experiments were conducted using TensorFlow/Keras in a GPU environment. A recommended setup is:

```text
Python 3.10
TensorFlow 2.15
Keras 2.15
NumPy
scikit-learn
Matplotlib
Seaborn
Pillow
SciPy
```

GPU users should ensure that their CUDA/cuDNN setup is compatible with their TensorFlow version.

## Main Configuration

The main training configuration follows the published paper:

```python
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCH = 50
l_rate = 0.001
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
```

The model uses:

- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Learning-rate scheduler:** ReduceLROnPlateau
- **Learning-rate decay factor:** 0.5
- **Scheduler patience:** 8 epochs
- **Minimum learning rate:** 1e-6
- **Model selection:** Best validation accuracy

---

## Outputs

The notebook saves generated files in the `outputs/` directory, including:

```text
HayabusaNet_Multiclass_best_weights.keras
HayabusaNet_Multiclass_best_model.keras
HayabusaNet_Multiclass_model_info.txt
HayabusaNet_Multiclass_val_accuracy.pdf
HayabusaNet_Multiclass_val_loss.pdf
HayabusaNet_Multiclass_confusion_matrix.pdf
HayabusaNet_Multiclass_ROC_curves.pdf
```

Depending on the final filename configuration, exact filenames may differ.

---

## Model Architecture Summary

HayabusaNet consists of:

1. **Input and normalization**
   - Input size: `224 × 224 × 3`
   - Pixel rescaling: `[0, 255] → [0, 1]`

2. **Three multiscale branches**
   - Branch 1: 3×3 depthwise separable convolutions
   - Branch 2: 5×5 depthwise separable convolutions
   - Branch 3: 7×7 depthwise separable convolutions
   - Each branch uses five convolution-pooling blocks.

3. **Multiscale feature fusion**
   - Element-wise addition.

4. **Hybrid attention**
   - CBAM-based channel and spatial attention.
   - Lightweight self-attention for global context.
   - Residual-gated fusion.

5. **Classification head**
   - Global average pooling.
   - Dense 512 + ReLU + Dropout.
   - Dense 256 + ReLU + Dropout.
   - Four-class softmax output.

---

## Evaluation

The notebook reports:

- Validation loss and accuracy.
- Macro-averaged precision.
- Macro-averaged recall.
- Macro-averaged F1-score.
- Confusion matrix.
- ROC curves and AUC.
- Parameter count.
- FLOPs.
- Inference latency and throughput.

The final evaluation should be performed using the **best checkpoint selected by validation accuracy**.

---

## Reproducibility Notes

The notebook sets a fixed random seed:

```python
SEED = 42
```

However, exact results may vary depending on:

- GPU model and driver.
- TensorFlow/CUDA/cuDNN versions.
- Dataset split and preprocessing.
- Non-deterministic GPU operations.
- Mixed-precision behavior.

For strict reproducibility, use the same software and hardware environment described in the published study whenever possible.

---

## Citation

If you use this repository or HayabusaNet in your research, please cite the published paper:

```bibtex
@article{Prayogo2026HayabusaNet,
  title   = {HayabusaNet: Hybrid attention-based multiscale fusion CNN for accurate and efficient brain tumor classification in MRI scans},
  author  = {Prayogo, Rizal Dwi and Karimah, Siti Amatullah and Nambo, Hidetaka},
  journal = {Expert Systems With Applications},
  volume  = {331},
  pages   = {133135},
  year    = {2026},
  doi     = {10.1016/j.eswa.2026.133135}
}
```

---

## License

The published article is open access under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

Please include a repository-level `LICENSE` file to specify the license applied to the source code and notebook.

---

## Disclaimer

This code is provided for academic research and reproducibility purposes. It has not been validated for clinical deployment. Any medical use would require further validation, regulatory approval, and expert clinical oversight.

---

## Contact

For questions related to the paper or implementation, please refer to the corresponding author information in the published article.


Please watch this repository or star ⭐ it for future updates.
