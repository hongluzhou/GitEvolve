# GitEvolve
The repository for code in GitEvolve: Predicting the Evolution of GitHub Repositories


# How to run
### models
We provide the following model variants:
- multitask_lstm_user_cluster (proposed model that outputs event type, time and user cluster of the next time step)
- multitask_lstm_user_cluster_true_event_only
- multitask_lstm_branch_et
- multitask_lstm_branch_td
- multitask_lstm_branch_uc
- multitask_random_constraints
- multitask_repeat_last_action
- multitask_repeat_no_action
- multitask_lstm (this model variant outputs only event type and time.)

Please use the main script of the model variant that you selected.



### train
```
python main_multitask_lstm_user_cluster.py train --create_dataset=1  --exp_name=myexp
```
### test (and evaluation)
```
python main_multitask_lstm_user_cluster.py test --create_dataset=1 --load_model_epoch=40  --exp_name=myexp --given_gt=0
```
or
```
python main_multitask_lstm_user_cluster.py test --create_dataset=1 --load_model_epoch=40  --exp_name=myexp --given_gt=1
```

### evaluation (only)
```
python main_multitask_lstm_user_cluster.py eval --load_model_epoch=40  --exp_name=myexp --given_gt=0
```

Please check the create_config.py inside the folder of each model variant, to change the data path, hyper-parameters, and other settings.
