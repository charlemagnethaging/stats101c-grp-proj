## Load data
load("data/regression/joined_data.RData")

## Dropping Duplicate or Similar Variables
# Note: Input names updated to lower case matches
duplicates <- c(
  "title",
  "asin.isbn.product.code",
  "q.demos.state",
  "q.amazon.use.hh.size"
)
data <- data %>% select(setdiff(colnames(data), duplicates))

## Variables Related to Selling Data Questions
selling_variables <- c(
  "q.sell.your.data",
  "q.sell.consumer.data",
  "q.small.biz.use",
  "q.census.use",
  "q.research.society",
  "test"
)

## Create an Encoded Binary Variable Based on if shared accounts
data <- data %>%
  mutate(
    # Renamed from sharedAccount to shared.account
    shared.account = case_when(
      # keep NA as NA
      is.na(q.amazon.use.howmany) ~ NA,

      # anything starting with 1 is FALSE
      grepl("^1", q.amazon.use.howmany) ~ 0,

      # all other values are TRUE
      TRUE ~ 1
    )
  )

## Extract order Year and Month to prepare for encoding
# make sure it's date format (updated Order.Date -> order.date)
data$order.date <- as.Date(data$order.date)

# extract year and month
# Renamed from orderYear/orderMonth to order.year/order.month
data$order.year <- format(data$order.date, "%Y")
data$order.month <- format(data$order.date, "%m")
data$order.year <- factor(data$order.year)
data$order.month <- factor(data$order.month)

## Drop rows where the category column is an empty string ("")
# Renamed Category -> category
data <- subset(data, category != "")

## Keep only rows where category is in the top 500 most frequent categories
freq <- sort(table(data$category), decreasing = TRUE)
top_categories <- names(freq)[1:500]
data <- subset(data, category %in% top_categories)

## Umbrella groups for the category variable
data <- data %>%
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

## Hot encoding for the multiple choice Life Changes variable
data <- data %>%
  mutate(
    # create multi-hot encoded columns (changed underscore to dot)
    change.employment = ifelse(
      str_detect(q.life.changes, regex("Lost a job", ignore_case = TRUE)),
      1,
      0
    ),
    change.relocation = ifelse(
      str_detect(
        q.life.changes,
        regex("Moved place of residence", ignore_case = TRUE)
      ),
      1,
      0
    ),
    change.relationship = ifelse(
      str_detect(q.life.changes, regex("Divorce", ignore_case = TRUE)),
      1,
      0
    ),
    change.family = ifelse(
      str_detect(
        q.life.changes,
        regex("Became pregnant|Had a child", ignore_case = TRUE)
      ),
      1,
      0
    ),
    # No categories selected
    change.none = ifelse(q.life.changes == "", 1, 0)
  )

## Race Variables
data <- data %>%
  mutate(
    # changed underscore to dot
    race.white = ifelse(
      str_detect(q.demos.race, regex("White|Caucasian", ignore_case = TRUE)),
      1,
      0
    ),
    race.black = ifelse(
      str_detect(
        q.demos.race,
        regex("Black|African American", ignore_case = TRUE)
      ),
      1,
      0
    ),
    race.asian = ifelse(
      str_detect(q.demos.race, regex("Asian", ignore_case = TRUE)),
      1,
      0
    ),
    race.native.american = ifelse(
      str_detect(
        q.demos.race,
        regex(
          "American Indian|Native American|Alaska Native",
          ignore_case = TRUE
        )
      ),
      1,
      0
    ),
    race.pacific.islander = ifelse(
      str_detect(
        q.demos.race,
        regex("Native Hawaiian|Pacific Islander", ignore_case = TRUE)
      ),
      1,
      0
    ),
    race.other = ifelse(
      str_detect(q.demos.race, regex("Other", ignore_case = TRUE)),
      1,
      0
    )
  )

## Drop the columns we encoded/prepared for encoding
# Fixed "data$Order.Date" bug (removed 'data$') and lowercased
data <- data %>%
  select(setdiff(
    colnames(data),
    c("q.amazon.use.howmany", "q.life.changes", "q.demos.race", "order.date")
  ))

## Drop unnecessary columns
cols_to_drop <- c(
  "survey.responseid",
  "order.date",
  "shipping.address.state",
  "title",
  "asin.isbn.product.code",
  "q.demos.education",
  "q.demos.gender",
  "q.sexual.orientation",
  "q.demos.state",
  "q.amazon.use.hh.size",
  "q.amazon.use.how.oft",
  "q.substance.use.cigarettes",
  "q.substance.use.marijuana",
  "q.substance.use.alcohol",
  "q.personal.diabetes",
  "q.personal.wheelchair",
  "q.sell.your.data",
  "q.sell.consumer.data",
  "q.small.biz.use",
  "q.census.use",
  "q.research.society",
  "test"
)
cleaned_data <- data %>% select(setdiff(colnames(data), cols_to_drop))


## Convert all character and logical variables to factors
data <- data %>%
  mutate(
    across(where(is.character), as.factor),
    across(where(is.logical), as.factor)
  )


## Handle Missing Data
imputation_pipeline <- gunion(list(
  # numeric features: regression tree imputation
  po("select", id = "sel_num", selector = selector_type("numeric")) %>>%
    po("imputelearner", id = "imp_num", learner = lrn("regr.rpart")),

  # factor / ordered features:classification tree imputation
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

## Encode Variables
# impute variables
full_pipeline <- imputation_pipeline %>>%
  po("encode", method = "one-hot")
# defining the task
# Updated target variable to lower case
housing_tsk <- as_task_regr(data, target = "q.amazon.use.hh.size.num")
# train pipeline on the data
full_pipeline$train(housing_tsk)
# transform the data
encoded_data <- full_pipeline$predict(housing_tsk)$task$data()

# Create learner
lrn_xgb <- as_learner(
  imputation_pipeline %>>%
    ## one-hot encode our categorical variables
    po("encode", method = "one-hot") %>>%

    ## pass cleaned dataset into an XGBoost regression model
    lrn("regr.xgboost")
)