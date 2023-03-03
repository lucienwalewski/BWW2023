## RVI-1 Approach

The features are computed on the following intervals:
- WS: 2021-12-01/2022-05-30
- SA: 2022-04-01/2022-09-30

Date measures: For date x, we compute the number of days between date x and the first date of the inveral observed. 

Features computed:
- start date (first min)
- end date (last min)
- max date (max between first and last min)

- RVI score at start date
- RVI score at end date 
- RVI score at max date 

- harvest date (from csv input)
- RVI score at harvest date 

- RVI correlation 

Model: ExtraTreesRegressor

Grid Search:
- 'max_depth': 20
- 'max_features': 1.0
- 'min_samples_leaf': 1
- 'min_samples_split': 7
- 'n_estimators': 100

R2 scores:
- y_train: 0.75-0.84
- y_test: 0.45-0.55 

Conclusion: overfitting 

- more features ?? 
- different regression ??



## RVI PLS Regression 
