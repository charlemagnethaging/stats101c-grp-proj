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
library(mlr3viz)
library(ranger)  # backend for regr.ranger (usually pulled in by mlr3learners, but safe)

load("processed_data_train.RData")  # this should create train_data
## handle missing data
imputation_pipeline <- gunion(list(
  # numeric features: regression tree imputation
  po("select", id = "sel_num", selector = selector_type("numeric")) %>>%
    po("imputelearner", id = "imp_num", learner = lrn("regr.rpart")),
  # factor / ordered features: classification tree imputation
  po(
    "select",
    id = "sel_cat",
    selector = selector_type(c("factor", "ordered"))
  ) %>>%
    po("imputelearner", id = "imp_cat", learner = lrn("classif.rpart")),
  # everything else (IDs, dates, etc.) just passed through
  po(
    "select",
    id = "sel_rest",
    selector = selector_invert(selector_type(c("numeric", "factor", "ordered")))
  )
)) %>>%
  po("featureunion")

# defining the task
housing_tsk <- as_task_regr(train_data, target = "q.amazon.use.hh.size.num")

## spliting training and test data
set.seed(101)
split <- partition(housing_tsk, ratio = 0.8)
train_idx <- split$train
test_idx  <- split$test

# measures
measures <- list(msr("regr.rmse"), msr("regr.mae"), msr("regr.mse"))

# ## lasso
# mod_mat <- model.matrix(~ ., data = preprocessed_data)
# # head(mod_mat, n = 2)
# lasso_housing_tsk <- as_task_regr(mod_mat[, -1], target = "q.amazon.use.hh.size.num")

# lrn_lasso_cv <- lrn("regr.cv_glmnet", 
#   alpha = 1, 
#   nfolds = 10, 
#   s = "lambda.min", 
#   lambda = 10^seq(from = -1.5, to = 1.5, by = 0.1), 
#   standardize = TRUE
# )

# lrn_lasso_cv$train(lasso_housing_tsk)
# coef(lrn_lasso_cv$model)

# ## -----------------------------------------------------------
# # xgbost learner
# lrn_xgboost <- as_learner(
#   imputation_pipeline %>>%
#     po("encode",  method = "one-hot") %>>%
#     lrn("regr.xgboost",
#         eta = 0.1,
#         max_depth = 3,
#         colsample_bytree = 0.8,
#         colsample_bylevel = 0.8,
#         lambda = 1,
#         alpha = 0,
#         subsample = 0.8
#     )
# )

# # training the model
# lrn_xgboost$train(housing_tsk, row_ids = train_idx)
# # testing the model
# preds <- lrn_xgboost$predict(housing_tsk, row_ids = test_idx)
# preds
# preds$score(measures)


# ## -----------------------------------------------------------
# # xgbost learner 2
# lrn_xgboost2 <- as_learner(
#   imputation_pipeline %>>%
#     po("encode",  method = "one-hot") %>>%
#     lrn("regr.xgboost",
#         eta = to_tune(1e-4, 1, logscale =TRUE),
#         max_depth = to_tune(1, 10),
#         colsample_bytree = to_tune(1e-1, 1),
#         colsample_bylevel = to_tune(1e-1, 1),
#         lambda = to_tune(1e-3, 1e3, logscale =TRUE),
#         alpha = to_tune(1e-3, 1e3, logscale =TRUE),
#         subsample = to_tune(1e-1, 1)
#     )
# )

# # 5-fold cross-validation resampling method 
# resampling <- rsmp("cv", folds = 5)

# # terminating condition
# terminator <- trm("evals", n_evals = 20)

# # grid search tuner
# tuner_grid <- tnr("grid_search", resolution = 3) 

# # autoturner
# at_xgboost <- auto_tuner(
#   learner = lrn_xgboost2,
#   resampling = resampling,
#   measure = msr("regr.rmse"),
#   terminator = terminator,
#   tuner = tuner_grid
# )

# # tune & train the model
# at_xgboost$train(housing_tsk, row_ids = train_idx)
# at_xgboost$tuning_result

# # test the model
# preds2 <- at_xgboost$predict(housing_tsk, row_ids = test_idx)
# preds2$score(measures)

# ## -----------------------------------------------------------
# # xgbost learner 3
# lrn_xgboost3 <- as_learner(
#   imputation_pipeline %>>%
#     po("encode",  method = "one-hot") %>>%
#     lrn("regr.xgboost",
#         eta = to_tune(1e-4, 1, logscale =TRUE),
#         max_depth = to_tune(1, 20),
#         colsample_bytree = to_tune(1e-1, 1),
#         colsample_bylevel = to_tune(1e-1, 1),
#         lambda = to_tune(1e-3, 1e3, logscale =TRUE),
#         alpha = to_tune(1e-3, 1e3, logscale =TRUE),
#         subsample = to_tune(1e-1, 1)
#     )
# )

# # tuning instance
# set.seed(101)
# instance <- ti(task = housing_tsk, 
#   learner = lrn_xgboost3,
#   resampling = rsmp("cv", folds = 5),
#   measures = list(msr("regr.rmse"), msr("regr.mae"), msr("regr.mse")),
#   terminator = trm("evals", n_evals = 100)
# )

# # grid search
# tuner <- tnr("grid_search", resolution = 10) 
# tuner$optimize(inst = instance)

# # define learner
# lrn_rpart_tuned <- lrn("regr.rpart")
# # optimal parameters
# instance$result
# optimal_params <- instance$result_learner_param_vals
# # set optimal hyperparamters 
# lrn_rpart_tuned$param_set$values <- optimal_params
# # train the fit parameters
# lrn_rpart_tuned$train(housing_tsk, row_ids = train_idx)
# # test the model
# preds3 <- lrn_rpart_tuned$predict(housing_tsk, row_ids = test_idx)
# preds3
# preds3$score(measures)


# ## random forest learner -----------------------------------------------------
# lrn_rf1 <- as_learner(
#   imputation_pipeline %>>%
#     # drop encode() because ranger handles factors,
#     # but if we want to keep one-hot, uncomment next line:
#     # po("encode", method = "one-hot") %>>%
#     lrn(
#       "regr.ranger",
#       num.trees = to_tune(200, 800),
#       min.node.size = to_tune(1, 10),
#       sample.fraction = 0.8,
#       importance      = "impurity"
#     )
# )

# ## train RF 
# lrn_rf1$train(housing_tsk, row_ids = train_idx)

# ## test RF 
# preds4 <- lrn_rf$predict(housing_tsk, row_ids = test_idx)
# preds4$score(measures)

# ## save RF model + preds -----------------------------------------------------
# saveRDS(lrn_rf, "rf_model.rds")
# saveRDS(preds,  "rf_predictions.rds")

## For Trevor
## rf learner 2 ---------------------------------------------------
lrn_rf2 <- as_learner(
  imputation_pipeline %>>%
    po("encode",  method = "one-hot") %>>%
    lrn("regr.ranger",
        num.trees = to_tune(50, 200),
        min.node.size = to_tune(1, 10)
    )
)

# train model
lrn_rf2$train(task = housing_tsk, row_ids = train_idx)

# test model
preds5 <- lrn_rf2$predict(housing_tsk, row_ids = test_idx)
preds5$score(measures)