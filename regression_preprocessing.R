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

# Load datasets
purchases <- read_csv("data/regression/amazon-purchases.csv")
survey <- read_csv("data/regression/survey_train_test.csv")

purchases <- purchases |> clean_names(sep_out=".")
survey <- survey |> clean_names(sep_out=".")

## remove the variables that aren't know at sign up
columns_to_keep <- colSums(is.na(survey)) == 0
columns_to_keep["q.amazon.use.hh.size.num"] <- TRUE
survey <- survey[, columns_to_keep]

# removing rows that are part of the test data
survey <- survey[survey$test == FALSE, ]

purchases$category[purchases$category == ""] <- NA

## Umbrella groups for the category variable
purchases <- purchases %>%
  mutate(
    category = case_when(
      str_detect(
        category,
        regex("BOOK|MUSIC|DVD", ignore_case = TRUE)
      ) ~ "Media",
      str_detect(
        category,
        regex("TOY|GAME|FIGURE|PUZZLE", ignore_case = TRUE)
      ) ~ "Games",
      str_detect(
        category,
        regex(
          "SHIRT|PANTS|DRESS|SHORTS|SWIMWEAR|SWEATER|SWEATSHIRT|BRA|UNDERPANTS|PAJAMAS|TUNIC|LEOTARD|ONE_PIECE_OUTFIT|APPAREL",
          ignore_case = TRUE
        )
      ) ~ "Clothing",
      str_detect(category, regex("PET|ANIMAL", ignore_case = TRUE)) ~ "Pet",
      str_detect(
        category,
        regex(
          "FOOD|SNACK|COFFEE|TEA|JUICE|MILK|BREAD|CHOCOLATE|SAUCE|SUGAR|NOODLE|RICE|PASTRY",
          ignore_case = TRUE
        )
      ) ~ "Food",
      str_detect(
        category,
        regex(
          "BEAUTY|SKIN|HAIR|COSMETIC|FRAGRANCE|LIP|MASCARA|NAIL",
          ignore_case = TRUE
        )
      ) ~ "Beauty",
      str_detect(
        category,
        regex(
          "ELECTRONIC|PHONE|HEADPHONES|CAMERA|MONITOR|FLASH|COMPUTER|TABLET|ACCESSORY|INPUT|DEVICE|DRIVE",
          ignore_case = TRUE
        )
      ) ~ "Electronics",
      str_detect(
        category,
        regex(
          "HOME|FURNITURE|LAMP|LIVING|DECOR|BED|BATH|KITCHEN|STORAGE|CLEANING|APPLIANCE|TOOL|LIGHT",
          ignore_case = TRUE
        )
      ) ~ "Home",
      str_detect(
        category,
        regex("SPORT|EXERCISE|FITNESS|RECREATION", ignore_case = TRUE)
      ) ~ "Fitness",
      str_detect(
        category,
        regex(
          "HEALTH|MEDICATION|SUPPLEMENT|VITAMIN|THERMOMETER|ORTHOPEDIC|PROTECTOR",
          ignore_case = TRUE
        )
      ) ~ "Health",
      str_detect(
        category,
        regex(
          "STATIONERY|PAPER|MARKING|INK|PEN|WRITING|NOTEBOOK",
          ignore_case = TRUE
        )
      ) ~ "Stationery",
      TRUE ~ "Other"
    )
  )

# the top 5 categories for each ID 
top_categories_by_id <- purchases %>%
  group_by(survey.response.id, category) %>%
  summarise(n = n(), .groups = 'drop_last') %>%
  arrange(desc(n)) %>%
  slice_head(n = 5) %>%
  summarise(top.categories = paste(category, collapse = ", ")) %>%
  ungroup()

# the shipping address state for each ID 
state_by_id <- purchases %>%
  group_by(survey.response.id, shipping.address.state) %>%
  summarise(n = n(), .groups = 'drop_last') %>%
  arrange(desc(n)) %>%
  slice_head(n = 1) %>% 
  select(survey.response.id, shipping.address.state) %>%
  ungroup()

# drop irrelevant categories 
purchases <- purchases |>
  select(!c(title, asin.isbn.product.code, category))

# flatten purchases to one row per ID

# summarize the total spent by the user 
# and the average amount spent per order for the user
purchases <- purchases |> 
  group_by(survey.response.id, order.date) |> 
  summarize(
    order.price = sum(purchase.price.per.unit*quantity)
  ) |> 
  group_by(survey.response.id) |> 
  summarize(
    total.spent = sum(order.price),
    avg.order.price = mean(order.price),

  ) |> 
  ungroup()

# merge the top categories summary & shipping address state into purchases
purchases <- purchases |>
  left_join(top_categories_by_id, by = "survey.response.id") |>
  left_join(state_by_id, by = "survey.response.id")

# Full join datasets on Survey Response ID
data <- full_join(
  purchases,
  survey,
  by = join_by(survey.response.id == survey.response.id)
)

## rename variables to lower.case.dot.space
data <- data |>
  clean_names(sep_out=".")

## hot encoding for the multiple top categories variable 
data <- data %>%
  mutate(
    category.media = ifelse(
      str_detect(top.categories, 
        regex("Media", ignore_case = TRUE)), 
      1, 0),
    category.games = ifelse(
      str_detect(top.categories, 
        regex("Games", ignore_case = TRUE)), 
      1, 0),
    category.clothing = ifelse(
      str_detect(top.categories, 
        regex("Clothing", ignore_case = TRUE)), 
      1, 0),
    category.pet = ifelse(
      str_detect(top.categories, 
        regex("Pet", ignore_case = TRUE)), 
      1, 0),
    category.food = ifelse(
      str_detect(top.categories, 
        regex("Food", ignore_case = TRUE)), 
      1, 0),
    category.beauty = ifelse(
      str_detect(top.categories, 
        regex("Beauty", ignore_case = TRUE)), 
      1, 0),
    category.electronics = ifelse(
      str_detect(top.categories, 
        regex("Electronics", ignore_case = TRUE)), 
      1, 0),
    category.home = ifelse(
      str_detect(top.categories, 
        regex("Home", ignore_case = TRUE)), 
      1, 0),
    category.fitness = ifelse(
      str_detect(top.categories, 
        regex("Fitness", ignore_case = TRUE)), 
      1, 0),
    category.health = ifelse(
      str_detect(top.categories, 
        regex("Health", ignore_case = TRUE)), 
      1, 0),
    category.stationery = ifelse(
      str_detect(top.categories, 
        regex("Stationery", ignore_case = TRUE)), 
      1, 0),
    category.other = ifelse(top.categories == "Other", 1, 0)
  )

## drop the column we encoded
data <- data %>%
  select(setdiff(colnames(data),c("top.categories")))

## convert all character and logical variables to factors
data <- data %>%
  mutate(
    across(where(is.character), as.factor),
    across(where(is.logical), as.factor)
  )

# data without the id, test, and the duplicate state variable
preprocessed_data <- data %>% select(setdiff(colnames(data), c("survey.response.id", "test", "q.demos.state")))

# save the data into the directory
save(preprocessed_data, file = "preprocessed_data.RData")

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

