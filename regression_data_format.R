library(tidyverse)
library(rpart)
library(janitor)

# Load datasets
purchases <- read_csv("data/regression/amazon-purchases.csv")
survey <- read_csv("data/regression/survey_train_test.csv")

# Full join datasets on Survey Response ID
data <- full_join(
  purchases,
  survey,
  by = join_by(`Survey ResponseID` == Survey.ResponseID)
)

## rename variables to lower.case.dot.space

data <- data |>
  clean_names(sep_out=".")

## Add numeric variable for numeric factors

data <- data |>
  mutate(
    q.amazon.use.howmany = factor(q.amazon.use.howmany, ordered = TRUE),
    q.amazon.use.hh.size = factor(q.amazon.use.hh.size, ordered = TRUE)
  ) |>
  mutate(
    q.amazon.use.howmany.num = as.numeric(q.amazon.use.howmany),
    .after = q.amazon.use.howmany
  )

save(data, file="data/regression/formatted_data.RData")