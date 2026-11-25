# Brain MRI Segmentation: Proposal

Ajay Uppu, MS Data Science & Analytics, Georgia State University, ML Project 1, November 2026

**Research Question**
Does adding batch-pooled soft Dice loss to binary cross-entropy (BCE) improve positive-slice Dice without increasing false positives on empty slices, compared with BCE alone, when the U-Net and the patient-disjoint data are kept the same?

**Motivation**
The task is highly imbalanced. The training set has 969 tumour slices and 1,794 empty slices, and only 1.06% of pixels are tumour. A 50-epoch BCE pilot run reached a validation pooled Dice of 0.7912, a positive-slice Dice of 0.6061, and 24 false-positive empty slices out of 394. At least one validation patient scored a Dice of 0. These failures suggest a trade-off worth testing; they do not promise an improvement. The pilot also exposed a bug in the data augmentation. It was fixed before the study, so the pilot only motivates the question and is not part of the comparison.

**Planned Method**
I keep the course's patient-level splits: train 77 patients (2,763 slices), validation 17 (598), and test 16 (568). Inputs are native 256×256 images with three MRI sequences as channels and binary masks. Normalization statistics come from the training set only, and augmentation is applied identically to image and mask. The model is a 2D U-Net trained from scratch, with 7,763,041 parameters and no pretrained weights. Every run uses the same settings: 50 epochs, batch size 16, AdamW (learning rate 0.001, weight decay 0.0001), and a cosine learning-rate schedule. Training runs on an NVIDIA L40S GPU in SageMaker Studio (PyTorch 2.8, CUDA 12.9), at about 11 minutes per run, so all 14 runs need roughly 3 to 4 GPU-hours.

**Baseline**
The learned baseline is the same U-Net trained with BCE alone. An always-empty mask prediction serves as a trivial reference.

**Comparison Plan**
The study has 14 fresh runs plus the always-empty reference:
- Main comparison: BCE vs BCE+Dice (λ=1), each with seeds 0, 1 and 2.
- Diagnostics: weighted BCE and BCE + positive-only Dice, each with seeds 0, 1 and 2. They test whether a simpler reweighting, or ignoring empty slices, explains any effect.
- Sensitivity: BCE+Dice with λ=0.5 and λ=2 at seed 0.

All methods and settings are frozen before the test set is evaluated, and the test set is evaluated once, with no tuning on it.

**Evaluation Metrics**
I will report foreground Dice and IoU, positive-slice Dice, empty-slice false positives (count, rate and pixels), and variation across patients. If both prediction and mask are empty, the slice scores 1; if only one is empty, it scores 0. Every run uses the same validation-only selection rule. I pick the best epoch and the decision threshold that maximize pooled validation Dice, searching thresholds from 0.05 to 0.95 in steps of 0.05. I will also examine seed spread, paired per-patient differences, results by lesion size, and failure overlays. The hypothesis is challenged if:
- the gains fall within seed-to-seed variation,
- weighted BCE achieves similar gains, or
- empty-slice false positives increase.

AI tools assisted with coding and documentation, and I verified all claims. This is a course project, not a clinical validation.

References: Ronneberger et al., U-Net, MICCAI 2015. Milletari et al., V-Net, 3DV 2016. Buda et al., LGG Segmentation Dataset (Kaggle), CC BY-NC-SA 4.0.
