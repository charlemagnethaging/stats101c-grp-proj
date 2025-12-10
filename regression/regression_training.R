library(tidyverse)
library(rpart)
library(mlr3)
library(mlr3learners)
library(mlr3verse)
library(mlr3pipelines)
library(ggplot2)
library(xgboost)
library(dplyr)
library(broom)
library(glmnet)
library(ranger)  # backend for regr.ranger (usually pulled in by mlr3learners, but safe)

load("processed_data_train.RData")  # this should create train_data
train_data <- train_data[ , !(names(train_data) %in% c("survey.response.id",
                                                       "total.orders",
                                                       "n.distinct.categories")) ]

load("processed_data_test.RData") # this should create test_data
# store the test id values to merge into submission cv later 
ids <- test_data$survey.response.id
test_data <- test_data[ , !(names(test_data) %in% c("survey.response.id",
                                                       "total.orders",
                                                       "n.distinct.categories")) ]

# defining the task
housing_tsk <- as_task_regr(train_data, target = "q.amazon.use.hh.size.num")

## spliting training and test data
set.seed(101)
split <- partition(housing_tsk, ratio = 0.8)
train_idx <- split$train
test_idx  <- split$test

# measures
measures <- list(msr("regr.rmse"), msr("regr.mse"))

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


## XGBoost Model -----------------------------------------------------------------
# learner
lrn_xgboost1 <- as_learner(
  po("encode",  method = "one-hot") %>>%
    lrn("regr.xgboost",
        eta = to_tune(1e-4, 1, logscale =TRUE),
        max_depth = to_tune(1, 10),
        colsample_bytree = to_tune(1e-1, 1),
        colsample_bylevel = to_tune(1e-1, 1),
        lambda = to_tune(1e-3, 1e3, logscale =TRUE),
        alpha = to_tune(1e-3, 1e3, logscale =TRUE),
        subsample = to_tune(1e-1, 1)
    )
)

# 5-fold cv
resampling1 <- rsmp("cv", folds = 5)
# terminate at 5 evals 
terminator1 <- trm("evals", n_evals = 5)
# grid search tuner
tuner_gs = tnr("grid_search")

#auto tuner 
at_xgboost1 <- auto_tuner(
  learner = lrn_xgboost1,
  resampling = resampling1,
  measure = msr("regr.rmse"), # evaluate by the rmse value
  terminator = terminator1, 
  tuner = tuner_gs
)

# train the model 
# commented out the below lines to prevent knitting from taking too long
# we extracted the best parameters and made a duplicate model with the best paramters set 
# at_xgboost1$train(housing_tsk, row_ids = train_idx)
# store the best parameters 
# best_param1 <- at_xgboost1$tuning_result

tuned_lrn_xgboost1 <- as_learner(
  po("encode",  method = "one-hot") %>>%
    lrn("regr.xgboost",
        eta = exp(-5.116856),
        max_depth = 3,
        colsample_bytree = 0.8,
        colsample_bylevel = 0.6,
        lambda = exp(3.837642),
        alpha = exp(-0.7675284),
        subsample = 1
    )
)

# train the model
set.seed(101)
tuned_lrn_xgboost1$train(housing_tsk, row_ids = train_idx)
# predict on train indecies 
preds1 <- tuned_lrn_xgboost1$predict(housing_tsk, row_ids = test_idx)
# obtain score
preds1$score(measures)
# regr.rmse  regr.mse 
#  1.078237  1.162595

## code to create predictions on the 2000 observations of test data
# test_predict2 <- at_xgboost1$predict_newdata(test_data)
# testing2 <- test_predict2$response
# testing2 <- round(testing2)

# extract variable importance
importance_scores <- at_xgboost1$learner$importance()
# variable importance plot
ggplot(data = data.frame(var = names(importance_scores),
                          value = importance_scores
                        ) |>
  dplyr::arrange(desc(value)) |>
  dplyr::slice(1:15)
) + ggtitle("XGBoost Model Top 15 Variable Importance Plot") + 
  geom_point(aes(x = value, y = reorder(var, value))) +
ylab("variable") + xlab("importance") 

## Random Forest Model 1 -----------------------------------------------------------------
## manually changed the parameters to reduce time waiting to tune the model 
lrn_rf1 <- lrn(
  "regr.ranger",
  num.trees       = 618,
  mtry            = 4,
  min.node.size   = 11,
  sample.fraction = 0.8449092,
  importance      = "impurity"
)

## train the model 
set.seed(101)
lrn_rf1$train(housing_tsk, row_ids = train_idx)
## predict on test indicies 
preds2 <- lrn_rf1$predict(housing_tsk, row_ids = test_idx)
preds2$score(measures)
# regr.rmse  regr.mse 
#  1.074993  1.155609 

# getting importance values
importance <- lrn_rf1$importance()
class(importance) # should be "numeric"
str(importance)

# importance values as a dataframe for plotting
vip_df <- data.frame(
  variable = names(importance),
  importance = as.numeric(importance)
)

# sort by importance, descending
vip_df <- vip_df[order(-vip_df$importance), ]
# plot 
ggplot(vip_df[1:15, ], 
  aes(x = reorder(variable, importance), y = importance)) + 
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(
    title = "Top 15 Most Important Variables (Random Forest)",
    x = "Predictor",
    y = "Impurity-based Importance"
  ) +
    theme_minimal()

## NEW DATA VARIABLES -----------------------------------------------------------------
load("processed_data_train.RData")
train_data <- train_data[ , !(names(train_data) %in% c("survey.response.id", 
                                                       "shipping.address.state", 
                                                       "q.demos.gender", 
                                                       "q.demos.state")) ]

load("processed_data_test2.RData")
ids <- test_data$survey.response.id
test_data <- test_data[ , !(names(test_data) %in% c("survey.response.id", 
                                                    "shipping.address.state", 
                                                    "q.demos.gender", 
                                                    "q.demos.state")) ]

# defining the task
housing_tsk <- as_task_regr(train_data, target = "q.amazon.use.hh.size.num")

## spliting training and test data
set.seed(101)
split <- partition(housing_tsk, ratio = 0.8)
train_idx <- split$train
test_idx  <- split$test

# measures
measures <- list(msr("regr.rmse"), msr("regr.mse"))

## Random Forest Model 2 -----------------------------------------------------------------

## comparing  -----------------------------------------------------------------
learners_to_compare <- list(
  tuned_lrn_xgboost1,
  lrn_rf1
)

rsmp_cv5 <- rsmp("cv", folds = 5)

bmr_design <- benchmark_grid(
  tasks = housing_tsk,
  learners = learners_to_compare,
  resamplings = rsmp_cv5
)

# running the actual benchmark experiment 
set.seed(101)
bmr <- benchmark(design = bmr_design)
bmr_rmse <- bmr$aggregate(measures = msr("regr.rmse"))[, c("learner_id", "regr.rmse")]
bmr_rmse[, c("learner_id", "regr.rmse")]

# visualizing benchmark results 
autoplot(bmr) + scale_y_log10()
