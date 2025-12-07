library(tidyverse)
library(rpart)
library(mlr3)
library(mlr3learners)
library(mlr3verse)
library(mlr3pipelines)
library(ggplot2)
library(xgboost)
library(dplyr)
library(rpart)
library(broom)
library(glmnet)
library(stringr)
library(janitor)

# xgbost learner
lrn_xgboost <- as_learner(
  imputation_pipeline %>>%
    po("encode",  method = "one-hot") %>>%
    lrn("regr.xgboost",
        eta = 0.1,
        max_depth = 3,
        colsample_bytree = 0.8,
        colsample_bylevel = 0.8,
        lambda = 1,
        alpha = 0,
        subsample = 0.8
    )
)

# defining the task
housing_tsk <- as_task_regr(clean_data, target = "q.amazon.use.hh.size.num")
lrn_xgboost$train(housing_tsk)
