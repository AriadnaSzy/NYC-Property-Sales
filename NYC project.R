 
# NYC Property Sales dataset:
# https://www.kaggle.com/new-york-city/nyc-property-sales

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

library(caret)
library(tidyverse)
library(lubridate)
library(rpart)

# Import data
zip_url <- "https://github.com/AriadnaSzy/NYC-Property-Sales/raw/master/nyc-property-sales.zip"
download.file(zip_url, destfile = "nyc-property-sales.zip")
nyc_csv_file <- read.csv(unz("nyc-property-sales.zip", "nyc-rolling-sales.csv"), stringsAsFactors = FALSE)

# Inspect dataset
head(nyc_csv_file)
colnames(nyc_csv_file)

# Preprocessing of data:
# Keep only columns of interest, and rename
sales <- nyc_csv_file %>% select("BOROUGH", "NEIGHBORHOOD", "BUILDING.CLASS.CATEGORY", "LAND.SQUARE.FEET", "GROSS.SQUARE.FEET", "YEAR.BUILT", "SALE.PRICE", "SALE.DATE")
names <- c("Borough", "Neighborhood", "Building_class", "Land_sq_ft", "Gross_sq_ft", "Year_built", "Sale_price", "Sale_date")
colnames(sales) <- names

# Inspect dataset structure for further data cleaning decisions
str(sales)

# Modify Borough values with actual borough names
codes <- data.frame(
  code = c(1, 2, 3, 4, 5), 
  borough = c("Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island")
  )
sales$Borough <- as.character(codes$borough[match(sales$Borough, codes$code)])

# Cleaning of data type - from character to date and numeric, to reflect the variables' characteristics
sales$Sale_date <- as.Date(sales$Sale_date)
numerics <- sales %>% select(Land_sq_ft, Gross_sq_ft, Sale_price) %>% mutate_if(is.character, as.numeric)
numerics[is.na(numerics)] <- 0
sales <- sales %>% select(-Land_sq_ft, -Gross_sq_ft, -Sale_price) %>% cbind(numerics)


# Perform further data cleaning - eliminate NAs, 0 values, outliers, that would influence analysis

sales <- sales %>% filter(!is.na(sales$Sale_price) & sales$Sale_price!=0)
price_out <- min(boxplot(sales$Sale_price)$out)
sales <- sales %>% filter(sales$Sale_price < price_out)

sales <- sales %>% filter(!is.na(sales$Year_built) & sales$Year_built!=0) 
year_out <- max(boxplot(sales$Year_built)$out)
sales <- sales %>% filter(sales$Year_built > year_out)

sales <- sales %>% filter(!is.na(sales$Gross_sq_ft) & sales$Gross_sq_ft!=0)
square_out <- min(boxplot(sales$Gross_sq_ft)$out)
sales <- sales %>% filter(sales$Gross_sq_ft < square_out)

#

# Find out how the 'year of the building' and the 'square footage' influence the 'sales price'
## Year of Building actually has a negative relation with the Sales Price, while the Square Footage has a weak positive correlation.
sales %>% summarize(correlation = cor(Sale_price, Year_built)) %>% pull(correlation)
sales %>% summarize(correlation = cor(Sale_price, Gross_sq_ft)) %>% pull(correlation)

# As per the resulted correlation, a visual exploration also reveals a positive relationship between Sales Price and Square Footage:
sales %>%
  ggplot(aes(x = Gross_sq_ft, y = Sale_price) ) +
  geom_point(pch = 21, size = 1, color = "white", fill = "black", alpha = 0.8) +
  geom_smooth(method = "lm", colour = "black") +
  ggtitle("Correlation between Sales Price and Square Footage")


### Proceed with data analysis through machine learning techniques ###

# Split the data into training set for algorithms modeling, and test set to assess the accuracy of the implemented models 
# Validation set will be 20% of NYC Sales data
set.seed(1)   # for reproducibility
test_index <- createDataPartition(y = sales$Sale_price, times = 1, p = 0.2, list = FALSE)
train_set <- sales[-test_index,]
test_set <- sales[test_index,]


# In analysing the data, linear regression  as a baseline approach can represent a valid method of analysis. 
# This model can be fitted with the below syntax, resulting in the following:
# The intercept (271724) represents the predicted Sale Price when Square Footage is at 0 (which doesn't make much sense in this case), whereas the slope (197) represents the change in Sale Price when Square Footage increases by one unit.
fit_lm <- lm(Sale_price ~ Gross_sq_ft, data = train_set)
fit_lm$coef

# An initial analysis, of average and standard deviation, provides the following:
# Average price of a property is 642132, with a standard deviation of 373826.
params <- train_set %>%
  summarize(avg = mean(Sale_price), sd = sd(Sale_price))
params


## For further estimating of Sales Prices, will be using several predictors instead of just the square footage. ##

# Using the RMSE evaluation metric, will write a function that computes this RMSE for the prices and their corresponding predictors:
RMSE <- function(true_prices, predicted_prices){
  sqrt(mean((true_prices - predicted_prices)^2))
}

# The estimate that minimizes the RMSE is the least squares estimate of the mean, in our case the average of all prices:
avg <- mean(train_set$Sale_price)
avg
# If we predict all unknown prices with this average, we obtain the following RMSE, that is very close to the standard deviation of our prices distribution:
avg_rmse <- RMSE(test_set$Sale_price, avg)
avg_rmse

# A good prediction model should, on average, have better predictions than the simple estimate of the mean for all predictions.
# will therefore test a few algorithms for the purpose of getting better predictions.


#
# use ten-fold cross-validation to estimate the prediction error. 
# specify the resampling scheme, that is, how cross-validation should be performed to find the best values of the tuning parameters:
control <- trainControl(method = "cv", number = 10)


# fit the logistic regression model:
model_glm <- train(Sale_price ~ Gross_sq_ft + Year_built + Borough, 
                 data = train_set, 
                 method = "glm", 
                 metric = "RMSE", 
                 trControl = control)
predictions_glm <- model_glm %>% predict(test_set)
rmse_glm <- RMSE(predictions_glm, test_set$Sale_price)

# fit the regression tree model:
model_rpart <- train(Sale_price ~ Gross_sq_ft + Year_built + Borough, 
                   data = train_set, 
                   method = "rpart", 
                   metric = "RMSE", 
                   tuneGrid = data.frame(cp = seq(0, 0.05, len = 10)),
                   minsplit = 2, 
                   trControl = control)
predictions_rpart <- model_rpart %>% predict(test_set)
rmse_rpart <- RMSE(predictions_rpart, test_set$Sale_price)

# fit the K-nearest neighbors (kNN) model:
model_knn <- train(Sale_price ~ Gross_sq_ft + Year_built + Borough, 
                 data = train_set, 
                 method = "knn", 
                 metric = "RMSE", 
                 tuneGrid = data.frame(k = seq(5,25,2)),
                 trControl = control)
predictions_knn <- model_knn %>% predict(test_set)
rmse_knn <- RMSE(predictions_knn, test_set$Sale_price)


## results:
methods <- c("logistic regression", "regression tree", "K-nearest neighbors")
rmse <- c(rmse_glm, rmse_rpart, rmse_knn)
rmse_results <- tibble(method = methods, RMSE = rmse) 
rmse_results %>% arrange(RMSE) 
 
#######
