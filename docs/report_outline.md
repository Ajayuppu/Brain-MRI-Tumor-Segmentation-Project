# Final report outline (6–8 pages)

Section headings follow the project description, section 7 (2). Each line names the result files to use.

1. **Introduction and question** – research question and what would contradict it (`docs/proposal.md`).
2. **Dataset and preprocessing** – split sizes, positive/empty slices, lesion sizes, channel intensities
   (`tables/data_audit_splits_and_labels.md`, `tables/data_audit_channel_intensities.md`,
   `figures/data_lesion_area_distribution.png`); normalisation, augmentation, mask handling.
3. **Baselines and proposed approach** – always-empty reference, BCE U-Net, BCE + Dice, diagnostics. State what was
   implemented, reused or modified (README "Attribution" and "Relation to the earlier study code").
4. **Experimental setup** – budget, seeds, hardware, parameter count, training time (`tables/experiment_record.md`);
   validation-only selection and the one-time test protocol.
5. **Results** – main comparison, diagnostics, sensitivity, seed stability (`tables/validation_summary_by_condition.md`,
   `tables/test_summary_by_condition.md`, `figures/training_curves.png`, `figures/*_positive_dice_vs_empty_false_positives.png`).
6. **Discussion and conclusions** – paired per-patient differences (`tables/*_paired_patient_dice.md`), lesion-size
   subgroups (`tables/validation_dice_by_lesion_size.md`), failure overlays (`figures/validation_failure_overlays.png`),
   threshold behaviour (`figures/validation_threshold_curves.png`), trade-offs, alternative explanations,
   limitations, negative results.
7. **References and disclosure** – dataset, U-Net, V-Net, libraries, AI assistance.
