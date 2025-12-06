## Dropping Duplicate Variables 
duplicates <- 
  
  ## Create an encoded binary variable for shared accounts
  data <- data %>%
  mutate(
    sharedAccount = case_when(
      # keep NA as NA
      is.na(Q.amazon.use.howmany) ~ NA, 
      
      # anything starting with 1 is FALSE
      grepl("^1", Q.amazon.use.howmany) ~ 0, 
      
      # all other values are TRUE
      TRUE ~ 1 
    )
  )

## Extract order Year and Month to prepare for encoding 
# make sure it's date format
data$Order.Date <- as.Date(data$Order.Date)
# extract year and month
data$orderYear  <- format(data$Order.Date, "%Y")
data$orderMonth <- format(data$Order.Date, "%m")
data$orderYear  <- factor(data$orderYear)
data$orderMonth <- factor(data$orderMonth)





## Drop rows where the Category column is an empty string ("")
data <- subset(data, Category != "")

## Keep only rows where Category is in the top 500 most frequent categories
freq <- sort(table(data$Category), decreasing = TRUE)
top_categories <- names(freq)[1:500]
data <- subset(data, Category %in% top_categories)

## Umbrella groups for the Category variable
data <- data %>%
  mutate(Category = case_when(
    str_detect(Category, 
               regex("BOOK|MUSIC|DVD", ignore_case = TRUE)) ~ "Media",
    str_detect(Category, 
               regex("TOY|GAME|FIGURE|PUZZLE", ignore_case = TRUE)) ~ "Games",
    str_detect(Category, 
               regex("SHIRT|PANTS|DRESS|SHORTS|SWIMWEAR|SWEATER|SWEATSHIRT|BRA|UNDERPANTS|PAJAMAS|TUNIC|LEOTARD|ONE_PIECE_OUTFIT|APPAREL", ignore_case = TRUE)) ~ "Clothing",
    str_detect(Category, 
               regex("PET|ANIMAL", ignore_case = TRUE)) ~ "Pet",
    str_detect(Category, 
               regex("FOOD|SNACK|COFFEE|TEA|JUICE|MILK|BREAD|CHOCOLATE|SAUCE|SUGAR|NOODLE|RICE|PASTRY", ignore_case = TRUE)) ~ "Food",
    str_detect(Category, 
               regex("BEAUTY|SKIN|HAIR|COSMETIC|FRAGRANCE|LIP|MASCARA|NAIL", ignore_case = TRUE)) ~ "Beauty",
    str_detect(Category, 
               regex("ELECTRONIC|PHONE|HEADPHONES|CAMERA|MONITOR|FLASH|COMPUTER|TABLET|ACCESSORY|INPUT|DEVICE|DRIVE", ignore_case = TRUE)) ~ "Electronics",
    str_detect(Category, 
               regex("HOME|FURNITURE|LAMP|LIVING|DECOR|BED|BATH|KITCHEN|STORAGE|CLEANING|APPLIANCE|TOOL|LIGHT", ignore_case = TRUE)) ~ "Home",
    str_detect(Category, 
               regex("SPORT|EXERCISE|FITNESS|RECREATION", ignore_case = TRUE)) ~ "Fitness",
    str_detect(Category, 
               regex("HEALTH|MEDICATION|SUPPLEMENT|VITAMIN|THERMOMETER|ORTHOPEDIC|PROTECTOR", ignore_case = TRUE)) ~ "Health",
    str_detect(Category, 
               regex("STATIONERY|PAPER|MARKING|INK|PEN|WRITING|NOTEBOOK", ignore_case = TRUE)) ~ "Stationery",
    TRUE ~ "Other"
  ))

## Hot encoding for the multiple choice Life Changes variable 
data <- data %>%
  mutate(
    # create multi-hot encoded columns
    change_employment  = ifelse(str_detect(Q.life.changes, regex("Lost a job", ignore_case = TRUE)), 1, 0),
    change_relocation  = ifelse(str_detect(Q.life.changes, regex("Moved place of residence", ignore_case = TRUE)), 1, 0),
    change_relationship = ifelse(str_detect(Q.life.changes, regex("Divorce", ignore_case = TRUE)), 1, 0),
    change_family      = ifelse(str_detect(Q.life.changes, regex("Became pregnant|Had a child", ignore_case = TRUE)), 1, 0),
    # No categories selected
    change_none = ifelse(Q.life.changes == "", 1, 0)
  )

## 
data <- data %>%
  mutate(
    race_white = ifelse(str_detect(Q.demos.race, regex("White|Caucasian", ignore_case = TRUE)), 1, 0),
    race_black = ifelse(str_detect(Q.demos.race, regex("Black|African American", ignore_case = TRUE)), 1, 0),
    race_asian = ifelse(str_detect(Q.demos.race, 
                                   regex("Asian", ignore_case = TRUE)), 1, 0),
    race_native_american = ifelse(str_detect(Q.demos.race, 
                                             regex("American Indian|Native American|Alaska Native", ignore_case = TRUE)), 1, 0),
    race_pacific_islander = ifelse(str_detect(Q.demos.race, 
                                              regex("Native Hawaiian|Pacific Islander",
                                                    ignore_case = TRUE)), 1, 0),
    race_other = ifelse(str_detect(Q.demos.race, 
                                   regex("Other", ignore_case = TRUE)), 1, 0)
  )

## Drop the columns we encoded/prepared for encoding
data <- data %>% select(setdiff(colnames(data), c("Q.amazon.use.howmany", "Q.life.changes", "Q.demos.race", "data$Order.Date")))

## Drop unnecessary columns 
cols_to_drop <- c("Survey.ResponseID", "Order.Date", "Shipping.Address.State","Title",
                  "ASIN.ISBN..Product.Code.", "Q.demos.education", "Q.demos.gender", 
                  "Q.sexual.orientation", "Q.demos.state", "Q.amazon.use.hh.size", 
                  "Q.amazon.use.how.oft", "Q.substance.use.cigarettes", 
                  "Q.substance.use.marijuana", "Q.substance.use.alcohol", 
                  "Q.personal.diabetes",  "Q.personal.wheelchair", 
                  "Q.sell.YOUR.data", "Q.sell.consumer.data", "Q.small.biz.use", 
                  "Q.census.use", "Q.research.society", "test")
cleaned_data <- data %>% select(setdiff(colnames(data), cols_to_drop))


## Convert all character and logical variables to factors
data <- data %>%
  mutate(across(where(is.character), as.factor),
         across(where(is.logical), as.factor))




## Handle Missing Data
imputation_pipeline <- gunion(list(
  # numeric features: regression tree imputation
  po("select", id = "sel_num", selector = selector_type("numeric")) %>>%
    po("imputelearner", id = "imp_num", learner = lrn("regr.rpart")),
  
  # factor / ordered features:classification tree imputation
  po("select", id = "sel_cat", selector = selector_type(c("factor","ordered"))) %>>%
    po("imputelearner", id = "imp_cat", learner = lrn("classif.rpart")),
  
  # everything else (IDs, dates, etc.) just passed through
  po("select", id = "sel_rest", selector = selector_invert(selector_type(c("numeric","factor","ordered"))))
)) %>>%
  po("featureunion")

## Encode Variables
# impute variables
full_pipeline <- imputation_pipeline %>>%
  po("encode", method = "one-hot")
# defining the task
housing_tsk <- as_task_regr(data, target = "Q.amazon.use.hh.size.num")
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


Before modeling, we performed a few preprocessing steps to ensure the survey and purchase datasets were clean and suitable for regression. First, we removed several variables that were irrelevant to prediction. The variables related to the usage of personal information and the customers' survey identification number did not feel relevant to our goal of being able to predict the household size of a customer using their purchasing behavior. However, we will make another subset of data including the customers' survey identification number for a generative additive model to account for subject variance later on. From our EDA, we were able to decide that the shipping address state, order date, product-specific information, education level, gender, sexual orientation, order frequency, substance use, and personal health-related variables are not helpful for making our prediction. By removing these predictors, it also helps us prevent overfitting from occurring.


We still have the product category variable still to identify different products which we believe would also be more representative of different products. Since there are a lot of potential responses for this variable, we will limit the amount to the top 500 categories to make our modeling process less expensive. Additionally, we have decided the convert the Q.amazon.use.howmany variable into a binary indicator called "Shared_Account" to indicate whether a customer's account is shared by others or is used be a single person. 

Since our dataset contains missing values, we decided to use rpart which can naturally handle missing values. In our pipeline, our numeric and categorical columns are separated, and then a regression and classification tree imputes the missing values respectively. Finally, we recombine all our features and have a clean dataset ready for modeling. 
