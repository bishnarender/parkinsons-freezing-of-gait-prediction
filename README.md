## parkinsons-freezing-of-gait-prediction
## score at 2nd position is achieved.
![parkinson_submission](https://github.com/bishnarender/parkinsons-freezing-of-gait-prediction/assets/49610834/6220f149-fc71-4820-9cc1-7f3d57b7df7f)


-----

### Start 
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. tdcsfog_train.ipynb
3. defog_train.ipynb
4. parkinson-submission.ipynb

### Data
-----
1. Data is in time series.
2. The tDCS FOG (tdcsfog) dataset, comprising data series collected in the lab.
3. The DeFOG (defog) dataset, comprising data series collected in the subject's home.
4. daily_metadata.csv, events.csv, subjects.csv, tasks.csv have never been used.
5. Series in the notype folder are from the <b>defog</b> dataset but lack event-type annotations.

### tdcsfog
-----
1. Picked each Id.csv from the tdcsfog folder. 
2. Mark each row in these Ids with "Valid" and "Mask" as 1. 
3. Partition each csv into overlapping blocks of size 15552. 
4. Fetch each block data and reshape it to 3D i.e., convert (15552,8) to (864,18,8). Taken 18 timesteps as sequence-length or patch-size. 18 timesteps = (1/7)th part of second.
5. Conversion of (15552,8) to (864,18,8) is known as resolution reduction. 
6. Early 3 values in the 2th dimension correspond to ['AccV', 'AccML', 'AccAP'] and act as our input. The Remaining three ['StartHesitation', 'Turn', 'Walking', 'Valid', 'Mask'] is our target. So, separate the input and target i.e., (864,18,3) will be our input and (864,18,5) our target. 
7. Reshape input to (864,54) from (864,18,3).
8. Reshape the target to (864,5) from (864,18,5) by taking the maximum value along the 1th dimension. Selecting the maximum value from among 18 timesteps [(1/7)th part of second] assumes that in 1 second a person cannot hesitate, turn or walk more than 7 times.
9. ​How is the validation set selected? Count the number of StartHesitation, Turn, Walking events in each subject with scipy.ndimage.label function. Then randomly select a subset of subjects (about 15% subjects). If the subset contains 20 - 30% StartHesitation events and 20 - 30% Walking events, then choose it.
10. Model being designed has used to predict 3 ['StartHesitation', 'Turn', 'Walking'] outputs. Multi-label binary cross-entropy has been implemented as loss function.​

### defog
-----
1. Picked each Id.csv from the defog folder. If Id.csv is not available in defog then it is picked from the "notype" folder.
2. In case of "defog", mark each row in these Ids with "StartHesitation_mask", "Turn_mask", "Walking_mask", "Event_mask" as 1 when ['Task'] & ['Valid'] are True. In case of "notype", mark each row in these Ids with ["StartHesitation_mask", "Turn_mask", "Walking_mask"] as 0; and ["Event_mask"] as 1 (1 when ['Task'] & ['Valid'] are True). 
3. Without Event flag, it is not possible to include "notype" data in the training process. Because the rest three ["StartHesitation_mask", "Turn_mask", "Walking_mask"] are not present in "notype" data.
4. Partition each csv into overlapping blocks of size 12096. 
5. Fetch each block data and reshape it to 3D i.e., convert (12096,11) to (864,14,11). Taken 14 timesteps as sequence-length or patch-size. 14 timesteps = (1/7)th part of second.
6. Early 3 values in the 2th dimension correspond to ['AccV', 'AccML', 'AccAP'] and act as our input. The Remaining eight ['StartHesitation', 'Turn', 'Walking', 'Event', 'StartHesitation_mask', 'Turn_mask', 'Walking_mask', 'Event_mask'] is our target. So, separate the input and target i.e., (864,14,3) will be our input and (864,14,8) our target. 
7. Reshape input to (864,42) from (864,14,3).
8. Reshape the target to (864,8) from (864,14,8) by taking the maximum value along the 1th dimension. Selecting the maximum value from among 14 timesteps [(1/7)th part of second] assumes that in 1 second a person cannot hesitate, turn or walk more than 7 times.
9. ​How is the validation set selected? Count the number of StartHesitation, Turn, Walking events in each subject with scipy.ndimage.label function. Then randomly select a subset of subjects (about 15% subjects). If the subset contains 20 - 30% StartHesitation events and 20 - 30% Walking events, then choose it.
10. Model being designed has used to predict 4 ['StartHesitation', 'Turn', 'Walking', 'Event'] outputs. Multi-label binary cross-entropy has been implemented as loss function.​

### model
-----
1. The model has two important parameters: patch size and sequence length. 864 is sequence length for the transformer.
2. The transformer encoder is to classify events (StartHesitation, Turn or Walking?), and LSTM part provides continuous communication between neighboring tokens.

![model](https://github.com/bishnarender/parkinsons-freezing-of-gait-prediction/assets/49610834/8c193f4e-807e-4ead-90a0-c3ca4ce96026)
