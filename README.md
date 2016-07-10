# Facebook_check_in


### Main Stuff

Two python sript based on similar methods: working with subset of the whole sample by chopping into grids

Original code : grid_v1.py

Improved code : grid_v2.py

* Add data for periodic time that hit the boundary: make sure 23:00 and 1:00 closer

* Add border_augment, including points near the boundary

### Ensemble

Initial idea to improve: based on the grid_v1. Run several models: KNN, Xgboost, with different feature included(log_accuracy, drop_day, drop accuracy), also with different grid number

Make Ensemble using the make_ensemble.py(Based on MAP@5)

Use gene_sub.py to make final submission. Without any parameter tuning, this can reach around 0.59

### Parameter tuning

Run hyOpt.py to fine tuning the paramter

### Including the RF

Including the RandomForest into final prediction is helpful. Also, we should make ensemble on three models directly based on the probability. grid_v2 use combination of KNN, Xgboost, and RF. Try mix_opt.py to find the best ratio between three models. Should have more sophisticated way to figure out the weight though.

This grid_v2.py should give the score around 0.60078

