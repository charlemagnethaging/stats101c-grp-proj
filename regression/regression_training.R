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
library(DALEX)
library(iml)

load("processed_data_train.RData")  # this should create train_data
train_data <- train_data[ , !(names(train_data) %in% c("survey.response.id",
                                                       "total.orders",
                                                       "n.distinct.categories")) ]
train_data <- train_data %>% select(-starts_with("spend"))

load("processed_data_test.RData") # this should create test_data
# store the test id values to merge into submission cv later 
ids <- test_data$survey.response.id
test_data <- test_data[ , !(names(test_data) %in% c("survey.response.id",
                                                       "total.orders",
                                                       "n.distinct.categories")) ]
test_data <- test_data %>% select(-starts_with("spend"))

# defining the task
housing_tsk <- as_task_regr(train_data, target = "q.amazon.use.hh.size.num")

## spliting training and test data
set.seed(101)
split <- partition(housing_tsk, ratio = 0.8)
train_idx <- split$train
test_idx  <- split$test

# measures
measures <- list(msr("regr.rmse"), msr("regr.mse"))

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
# test_predict <- tuned_lrn_xgboost1$predict_newdata(test_data)
# testing <- test_predict$response

# submit <- data.frame(ids, testing)
# str(submit)
# colnames(submit) <- c("Survey.ResponseID", "Q.amazon.use.hh.size.num")

# library(data.table)
# write_csv(submit, "submit.csv")

# extract variable importance
importance_scores <- tuned_lrn_xgboost1$importance()
# variable importance plot
ggplot(data = data.frame(var = names(importance_scores),
                          value = importance_scores
                        ) |>
  dplyr::arrange(desc(value)) |>
  dplyr::slice(1:15)
) + ggtitle("XGBoost Model Top 15 Variable Importance Plot") + 
  geom_point(aes(x = value, y = reorder(var, value))) +
ylab("variable") + xlab("importance") 


# lime visual
X_test <- housing_tsk$data(
  rows = test_idx,
  cols = housing_tsk$feature_names
)

y_test <- housing_tsk$data(
  rows = test_idx,
  cols = housing_tsk$target_names
)[[1]]

pred_fun <- function(model, newdata) {
  model$predict_newdata(newdata)$response
}

predictor <- Predictor$new(
  model = tuned_lrn_xgboost1,
  data  = X_test,
  y     = y_test,
  predict.function = pred_fun
)

# pick some interesting test points (rows within X_test)
example_ids <- c(1, 20)

set.seed(101)
# see their predictions for context
tuned_lrn_xgboost1$predict_newdata(X_test[example_ids, ])

set.seed(101)
# LocalModel = LIME-style local surrogate
loc_1 <- LocalModel$new(predictor, x.interest = X_test[example_ids[1], ], k = 5)
loc_2 <- LocalModel$new(predictor, x.interest = X_test[example_ids[2], ], k = 5)

# plots: contribution of top features for each case
plot(loc_1)
plot(loc_2)

tuned_lrn_xgboost1$param_set$values

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
# str(importance)

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

# lime visuals
predictor <- Predictor$new(
  model = lrn_rf1,
  data  = X_test,
  y     = y_test,
  predict.function = pred_fun
)

# pick some interesting test points (rows within X_test)
example_ids <- c(1, 20)

# see their predictions for context
set.seed(101)
lrn_rf1$predict_newdata(X_test[example_ids, ])

set.seed(101)
# LocalModel = LIME-style local surrogate
loc_1 <- LocalModel$new(predictor, x.interest = X_test[example_ids[1], ], k = 5)
loc_2 <- LocalModel$new(predictor, x.interest = X_test[example_ids[2], ], k = 5)

# plots: contribution of top features for each case
plot(loc_1)
plot(loc_2)

## NEW DATA VARIABLES -----------------------------------------------------------------
load("processed_data_train.RData")
data <- train_data |> select(-survey.response.id)
# defining the task
tsk_hhsize <- as_task_regr(data, target = "q.amazon.use.hh.size.num", id="amazon_hh_size")

# Factor Encoding Pipeline
factor_pipeline <- 
  po("collapsefactors", # collapse levels occuring less than 1% in data
      no_collapse_above_prevalence = 0.01) %>>% 
  po("encodeimpact", # impact encoding factors with > 10 lvls
      affect_columns = selector_cardinality_greater_than(10),
      id="high_card_enc") %>>%
  po("encode", method = "one-hot", id="low_card_enc") # one-hot encode low card. features

## Final Model: XGBoost Model 2 -----------------------------------------------------------------
lrn_xgb <- as_learner(
	po("encode", method = "one-hot") %>>%
		lrn(
			"regr.xgboost",
			eta = to_tune(1e-4, 1, logscale = TRUE),
			max_depth = to_tune(1, 15),
			colsample_bytree = to_tune(0.1, 1),
			subsample = to_tune(0.1, 1),
			lambda = to_tune(1e-3, 1e3, logscale = TRUE),
			alpha = to_tune(1e-3, 1e3, logscale = TRUE)
		)
)

set.seed(13)
plan(list(
	tweak("multisession", workers = 3),
	tweak("multisession", workers = 3)
))

instance = ti(
	task = tsk_hhsize,
	learner = lrn_xgb,
	resampling = rsmp("cv", folds = 5),
	measure = msr("regr.rmse"),
	terminator = trm("evals", n_evals = 60)
)

## commenting the below code out to save time for knitting 
# tuner_rs$optimize(instance)
## extract best params, fit and train final model
# best_params <- instance$result_learner_param_vals
xgb_tune4 <- as_learner(
	po("encode", method = "one-hot") %>>%
		lrn(
			"regr.xgboost",
			eta = 0.007055363,
			max_depth = 4,
			colsample_bytree = 0.319908,
			subsample = 0.369098,
			lambda = exp(-1.05846993385),
			alpha = exp(0.73788139955)
		)
)

set.seed(13)
hhsize_split <- partition(tsk_hhsize, ratio=0.7) 
xgb_tune4$train(tsk_hhsize, row_ids = hhsize_split$train)

# var importance plot sanity check
as_tibble(xgb_tune4$importance(), rownames="var") |> ggplot() + 
  geom_point(aes(y = reorder(var, value), x = value))

# evaluate on test set
preds <- xgb_tune4$predict(tsk_hhsize, row_ids = hhsize_split$test)
preds$score(c(msr("regr.rmse"), msr("regr.mae"),  msr("regr.mse")))
# regr.rmse  regr.mae  regr.mse 
# 1.0438824 0.8887537 1.0896904 

# lime visuals
predictor <- Predictor$new(
  model = xgb_tune4,
  data  = X_test,
  y     = y_test,
  predict.function = pred_fun
)

# pick some interesting test points (rows within X_test)
example_ids <- c(1, 20)

# see their predictions for context
set.seed(101)
xgb_tune4$predict_newdata(X_test[example_ids, ])

set.seed(101)
# LocalModel = LIME-style local surrogate
loc_1 <- LocalModel$new(predictor, x.interest = X_test[example_ids[1], ], k = 5)
loc_2 <- LocalModel$new(predictor, x.interest = X_test[example_ids[2], ], k = 5)

# plots: contribution of top features for each case
plot(loc_1)
plot(loc_2)

## comparing models w/ xgb1 -----------------------------------------------------------------
tuned_lrn_xgboost1$id <- "tuned_lrn_xgboost1"
lrn_rf1$id <- "lrn_rf1"
xgb_tune4$id <- "xgb_tune4"
learners_to_compare <- list(tuned_lrn_xgboost1, lrn_rf1, lrn("regr.featureless"), xgb_tune4)

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
bmr_rmse

# visualizing benchmark results 
autoplot(bmr, measure = msr("regr.rmse")) + scale_y_log10()
