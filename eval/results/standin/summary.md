# Evaluation: standin

40 slides from eval/test_subset.txt

| mode | budget | slides | slide_auc | tumour_slides_with_a_hit | lesion_recall | macro_recall | micro_recall | itc_recall | mean_tissue_viewed | fp_tiles_per_slide |
|---|---|---|---|---|---|---|---|---|---|---|
| sliding_window | all | 40 | 0.73125 | 19/20 | 0.899 | 21/21 | 68/78 | 90/162 | 1.0 | 797.6 |
| best_first | 10 | 40 | 0.6975 | 13/20 | 0.222 | 13/21 | 9/78 | 18/162 | 0.072 | 63.0 |
| best_first | 20 | 40 | 0.7325 | 15/20 | 0.364 | 18/21 | 18/78 | 32/162 | 0.143 | 131.4 |
| best_first | 40 | 40 | 0.7025 | 15/20 | 0.455 | 20/21 | 25/78 | 45/162 | 0.287 | 255.9 |
| best_first | 80 | 40 | 0.71375 | 16/20 | 0.616 | 21/21 | 40/78 | 56/162 | 0.541 | 453.4 |
| best_first | 160 | 40 | 0.73125 | 19/20 | 0.808 | 21/21 | 59/78 | 78/162 | 0.821 | 658.5 |
