---
title: "Final Report"
author: "Adesh Valecha, Sushruti Acharya"
output:
  html_notebook: default
  pdf_document: default
---

## Data Inspection, Data modeling and cleaning

```{r }
# Load packages and data

library(tidyverse)
library(arm)
library(caret)
library(MASS)
library(missForest)
library(rminer)
library(dplyr)
library(RWeka)
library(psych)

#Import train dataset

td <-  read.csv("C:/Users/sushr/Documents/MSIS/Stats & Preds/Final Project/train.csv", stringsAsFactors=FALSE)
#Import test dataset
test <-  read.csv("C:/Users/sushr/Documents/MSIS/Stats & Preds/Final Project/test.csv", stringsAsFactors=FALSE)

#glimpse(td)
#glimpse(test)

td$Alley [is.na(td$Alley)] <- "Unknown"
td$BsmtQual [is.na(td$BsmtQual)] <- "Unknown"
td$BsmtCond [is.na(td$BsmtCond)] <- "Unknown"
td$BsmtExposure [is.na(td$BsmtExposure)] <- "Unknown"
td$BsmtFinType1 [is.na(td$BsmtFinType1)] <- "Unknown"
td$BsmtFinType2 [is.na(td$BsmtFinType2)] <- "Unknown"
td$FireplaceQu [is.na(td$FireplaceQu)] <- "Unknown"
td$GarageType [is.na(td$GarageType)] <- "Unknown"
td$GarageFinish [is.na(td$GarageFinish)] <- "Unknown"
td$GarageQual [is.na(td$GarageQual)] <- "Unknown"
td$GarageCond [is.na(td$GarageCond)] <- "Unknown"
td$PoolQC [is.na(td$PoolQC)] <- "Unknown"
td$Fence [is.na(td$Fence)] <- "Unknown"
td$MiscFeature [is.na(td$MiscFeature)] <- "Unknown"
td$Electrical [is.na(td$Electrical)] <- "Unknown"
td$MasVnrType [is.na(td$MasVnrType)] <- "Unknown"

td$LotFrontage [is.na(td$LotFrontage)] <- round(mean(td$LotFrontage,na.rm=TRUE))
td$MasVnrArea [is.na(td$MasVnrArea)] <- round(mean(td$MasVnrArea,na.rm=TRUE))
td$GarageYrBlt [is.na(td$GarageYrBlt)] <- round(mean(td$GarageYrBlt,na.rm=TRUE))

attributes <- names(td)
attributes <- attributes[attributes != "SalePrice"]

for(i in attributes)
{
  if(is.character(td[[i]]))
  {
    levels <- sort(unique(c(td[[i]])))
    td[[i]] <- factor(td[[i]],levels=levels)
  }
}

for (i in attributes) {
 if(class(levels(td[[i]])) == "character")
  td[[i]] <- seq_along(levels(td[[i]]))[td[[i]]]
}

summary(td)

```
Data cleaning of Test dataset

```{r}
test$Alley [is.na(test$Alley)] <- "Unknown"
test$BsmtQual [is.na(test$BsmtQual)] <- "Unknown"
test$BsmtCond [is.na(test$BsmtCond)] <- "Unknown"
test$BsmtExposure [is.na(test$BsmtExposure)] <- "Unknown"
test$BsmtFinType1 [is.na(test$BsmtFinType1)] <- "Unknown"
test$BsmtFinType2 [is.na(test$BsmtFinType2)] <- "Unknown"
test$FireplaceQu [is.na(test$FireplaceQu)] <- "Unknown"
test$GarageType [is.na(test$GarageType)] <- "Unknown"
test$GarageFinish [is.na(test$GarageFinish)] <- "Unknown"
test$GarageQual [is.na(test$GarageQual)] <- "Unknown"
test$GarageCond [is.na(test$GarageCond)] <- "Unknown"
test$PoolQC [is.na(test$PoolQC)] <- "Unknown"
test$Fence [is.na(test$Fence)] <- "Unknown"
test$MiscFeature [is.na(test$MiscFeature)] <- "Unknown"
test$Electrical [is.na(test$Electrical)] <- "Unknown"
test$MasVnrType [is.na(test$MasVnrType)] <- "Unknown"

test$LotFrontage [is.na(test$LotFrontage)] <- round(mean(test$LotFrontage,na.rm=TRUE))
test$MasVnrArea [is.na(test$MasVnrArea)] <- round(mean(test$MasVnrArea,na.rm=TRUE))
test$GarageYrBlt [is.na(test$GarageYrBlt)] <- round(mean(test$GarageYrBlt,na.rm=TRUE))

attributes <- names(test)
attributes <- attributes[attributes != "SalePrice"]

for(i in attributes)
{
  if(is.character(test[[i]]))
  {
    levels <- sort(unique(c(test[[i]])))
    test[[i]] <- factor(test[[i]],levels=levels)
  }
}

for (i in attributes) {
 if(class(levels(test[[i]])) == "character")
  test[[i]] <- seq_along(levels(test[[i]]))[test[[i]]]
}

summary(test)

rmse <- function(actVal, predVal) {
 sqrt(mean((actVal - predVal)^2))
}

```

##Model and Model Development
We are using five modelling methods in our report to find the strongest predictors. These methods are linear regression, ridge model, lasso model, ridge/lasso mixed model and correlation. We selected the best predictors from each model and built a baseline model (Linear model) to get the strongest predictors across all models.


##1) Linear Model
```{r Linear Model}

lmodel<-lm(SalePrice ~ ., data = td)
varImp(lmodel)
head(sort(abs(lmodel$coefficients),decreasing = TRUE),n=16)

```
Strongest predictors from Linear Model
 
1) Pool quality
2) Type of utilities available
3) Street
4) GarageCars
5) KitchenAbvGr
6) OverallQual
7) ExternalQual
8) Condition2
9) KitchenQual
10) BsmtQual
11) BsmtFullBath
12) RoofMatl
13) LandSlope
14) OverallCond
15) MasVnrType


##2) Ridge Model
```{r Ridge model, warning=FALSE}
ridgeModel <- train(SalePrice ~ ., 
                   data = td,
                   preProcess = c("center", "scale"),
                   method = "glmnet",
                   tuneGrid= expand.grid(
                     alpha=0,
                     lambda = seq(0,10, .1)))
varImp(ridgeModel)
```
Strongest predictor from Ridge Model

1) OverallQual   
2) GrLivArea      
3) X1stFlrSF      
4) BsmtQual       
5) KitchenQual    
6) PoolQC         
7) X2ndFlrSF      
8) GarageCars     
9) ExterQual      
10) PoolArea       
11) TotRmsAbvGrd   
12) MasVnrArea     
13) OverallCond    
14) MSSubClass     
15) YearBuilt

##3) Lasso Model
```{r Lasso Model, warning=FALSE}
lassoModel <- train(SalePrice ~ ., 
                   data = td,
                   preProcess = c("center", "scale"),
                   method = "glmnet",
                   tuneGrid= expand.grid(
                     alpha=1,
                     lambda = seq(0,10, .1)))

varImp(lassoModel)

```
Strongest predictor from Lasso Model

1) GrLivArea
2) OverallQual
3) Pool quality
4) PoolArea
5) GarageCars
6) BsmtQual
7) KitchenQual
8) YearBuilt
9) ExterQual
10) MasVnrArea
11) TotRmsAbvGrd
12) OverallCond
13) MSSubClass
14) LotArea
15) BsmtExposure
  
##4) Ridge and Lasso Mixture Model
```{r, Ridge and Lasso Mixture Model, warning=FALSE}
mixModel <- train(SalePrice ~ ., 
                 data = td,
                 preProcess = c("center", "scale"),
                 method = "glmnet",
                 tuneGrid= expand.grid(
                   alpha=0:1,
                   lambda = seq(0,10, .1)))

varImp(mixModel)

```
Strongest predictor from Ridge and Lasso Mix Model

1) OverallQual
2) GrLivArea
3) X1stFlrSF
4) BsmtQual
5) KitchenQual
6) Pool quality
7) GarageCars
8) X2ndFlrSF
9) ExterQual
10) PoolArea
11) TotRmsAbvGrd
12) MasVnrArea
13) OverallCond
14) MSSubClass
15) BsmtExposure
  
##5) Correlation 
```{r Correlation, warning=FALSE}
correlationModel <- cor(td)
correlation <- correlationModel[,c(0,81)]
head(sort(abs(correlation),decreasing = TRUE),n=16)
```
Strongest relation of the variables with output variable SalePrice

1) OverallQual
2) GrLivArea
3) GarageCars
4) ExterQual
5) GarageArea
6) BsmtQual
7) TotalBsmtSF
8) X1stFlrSF
9) KitchenQual
10) FullBath
11) GarageFinish
12) TotRmsAbvGrd
13) YearBuilt
14) MasVnrArea
15) YearRemodAdd

## Builing the baseline model (Linear model) to find the strongest predictors
```{r,  echo=F, message=F, include=FALSE, warning=FALSE}
#lm model top predictors
Lm_Test1<-lm(SalePrice ~ PoolQC+Utilities+Street+GarageCars+KitchenAbvGr+OverallQual+ExterQual +Condition2+KitchenQual+BsmtQual+BsmtFullBath+RoofMatl+LandSlope+OverallCond+MasVnrType,data=td)
ActualRMSE1<-rmse(td$SalePrice, predict(Lm_Test1))
ActualRMSE1
# Insample RMSE 40465.96

plot(Lm_Test1)

#Ridge model top predictors
Ridge_Test1<-lm(SalePrice ~ OverallQual+GrLivArea+X1stFlrSF+BsmtQual+KitchenQual+PoolQC+GarageCars+ 
X2ndFlrSF+ExterQual+PoolArea+TotRmsAbvGrd+MasVnrArea+OverallCond+MSSubClass+YearBuilt
,data=td)
ActualRMSE2<-rmse(td$SalePrice, predict(Ridge_Test1))
ActualRMSE2
# Insample RMSE 33909.77

plot(Ridge_Test1)

#Lasso model top predictors
Lasso_Test1<-lm(SalePrice ~ GrLivArea+OverallQual+PoolQC+PoolArea+GarageCars+BsmtQual+KitchenQual+YearBuilt+ExterQual+MasVnrArea+TotRmsAbvGrd+OverallCond+MSSubClass+LotArea+BsmtExposure
,data=td)
ActualRMSE3<-rmse(td$SalePrice, predict(Lasso_Test1))
ActualRMSE3
# Insample RMSE 33249.27

plot(Lasso_Test1)

#Ridge-Lasso mix model top predictors
Mix_Test1<-lm(SalePrice ~ GrLivArea+OverallQual+X1stFlrSF+X2ndFlrSF+PoolQC+PoolArea+GarageCars+BsmtQual+KitchenQual+ExterQual+MasVnrArea+TotRmsAbvGrd+OverallCond+MSSubClass+BsmtExposure
,data=td)
ActualRMSE4<-rmse(td$SalePrice, predict(Mix_Test1))
ActualRMSE4
# Insample RMSE 33972.54

plot(Mix_Test1)

#Correlation model top predictors
Cor_Test1<-lm(SalePrice ~ OverallQual+GrLivArea+GarageCars+ExterQual+GarageArea+BsmtQual+TotalBsmtSF+X1stFlrSF+KitchenQual+FullBath+GarageFinish+TotRmsAbvGrd+YearBuilt+GarageYrBlt+YearRemodAdd
,data=td)
ActualRMSE5<-rmse(td$SalePrice, predict(Cor_Test1))
ActualRMSE5
# Insample RMSE 35655.24

plot(Cor_Test1)

```
Comparing the output of all the above models and techniques we reached to a conclusion that the strongest predictors which can be used to develop a model are

1)	Pool quality
2)	GrLivArea
3)	YearBuilt
4)	GarageCars
5)	OverallQual
6)	KitchenQual
7)	BsmtQual
8)	OverallCond
9)	X1stFlrSF
10)	MSSubClass
11)	PoolArea
12)	ExterQual
13)	MasVnrArea
14)	LotArea
15)	TotRmsAbvGrd
16) GarageArea
17) TotalBsmtSF
18) FullBath

##Visualizations
Analysisng Data and effect of different most Significant Predictors (3) that were commonly identified through LM, KNN, GLMNET,Random Forest and correlation with ggplot 

### Impact of the most significant factor OverallQual on Sale Price
```{r}
# Overall Quality is the most significant factor in predicting sale price
td %>% 
  ggplot(aes(OverallQual,SalePrice)) +
  geom_point()+
  theme_bw()
```
### Identifying GrLivingArea Impact on Housing Price
```{r}
## Identifying GrLivingArea Impact on Housing Price
ggplot(td, aes(x=GrLivArea)) +
  geom_histogram(fill='blue',color='white') +
  theme_minimal()
```
### Kitchen Quality also matters in Sale Price
```{r}
ggplot(data=td, aes(x=KitchenQual, y=SalePrice, fill=KitchenQual)) + geom_bar(stat="identity")
  
```

##Different modeling methods were then applied on the above predictors to find the most accurate model.

```{r}
# Linear Model
set.seed(1234)
linear.model <- train(SalePrice ~ PoolQC + GrLivArea  + GarageCars + GarageArea + TotalBsmtSF + FullBath + TotRmsAbvGrd +    OverallQual + KitchenQual + BsmtQual + OverallCond + X1stFlrSF  +
    PoolArea + ExterQual + MasVnrArea + LotArea + 
    MSSubClass + YearBuilt, data = td,
            method = "lm")
linear.model
# Out of sample RMSE: 56456.58 and Rsquared:0.7411109

rmse_linear <- rmse(td$SalePrice, predict(linear.model,newdata = td))
rmse_linear
# In-sample RMSE: 33328.98


# Ridge Model
set.seed(1234)
model.ridge <- train(SalePrice ~ PoolQC + GrLivArea  + GarageCars + GarageArea + TotalBsmtSF + FullBath + TotRmsAbvGrd +    OverallQual + KitchenQual + BsmtQual + OverallCond + X1stFlrSF  +
    PoolArea + ExterQual + MasVnrArea + LotArea + 
    MSSubClass + YearBuilt, data = td,
                preProcess = c("center", "scale"),
                method = "glmnet",
                tuneGrid= expand.grid(
                  alpha=0,
                  lambda = 0:10/20))
model.ridge
# Out of sample RMSE: 37966.41 and Rsquared:0.7703654

rmse.ridge <- rmse(td$SalePrice, predict(model.ridge,newdata = td))
rmse.ridge
# In-sample RMSE: 33552.81

#Lasso Model
set.seed(1234)
model.lasso <- train(SalePrice ~ PoolQC + GrLivArea  + GarageCars + GarageArea + TotalBsmtSF + FullBath + TotRmsAbvGrd +    OverallQual + KitchenQual + BsmtQual + OverallCond + X1stFlrSF  +
    PoolArea + ExterQual + MasVnrArea + LotArea + 
    MSSubClass + YearBuilt, data = td,
                   preProcess = c("center", "scale"),
                   method = "glmnet",
                   tuneGrid= expand.grid(alpha=1,
                     lambda = 0:20/10))
model.lasso
# Out of sample RMSE: 55568.63 and Rsquared:0.7426154

rmse.lasso <- rmse(td$SalePrice, predict(model.lasso,newdata = td))
rmse.lasso
# In-sample RMSE: 33332.57

#kNN
set.seed(1234)
model.knn <- train(SalePrice ~ PoolQC + GrLivArea  + GarageCars + GarageArea + TotalBsmtSF + FullBath + TotRmsAbvGrd +    OverallQual + KitchenQual + BsmtQual + OverallCond + X1stFlrSF  +
    PoolArea + ExterQual + MasVnrArea + LotArea + 
    MSSubClass + YearBuilt, data = td,
      method="knn",
      preProcess=c("center","scale"))
model.knn
# Out of sample RMSE: 36772.19 and Rsquared:0.7803419    

rmse.knn <- rmse(td$SalePrice, predict(model.knn,newdata = td))
rmse.knn
# In-sample RMSE: 31276.88

#Random Forest
set.seed(1234)
model.rf <- train(SalePrice ~ PoolQC + GrLivArea  + GarageCars + GarageArea + TotalBsmtSF + FullBath + TotRmsAbvGrd +    OverallQual + KitchenQual + BsmtQual + OverallCond + X1stFlrSF  +
    PoolArea + ExterQual + MasVnrArea + LotArea + 
    MSSubClass + YearBuilt, data = td,
      method="rf",
      preProcess=c("center","scale")
      )
model.rf
# Out of sample RMSE: 31363.79 and Rsquared:0.8393970 

rmse.rf <- rmse(td$SalePrice, predict(model.rf,newdata = td))
rmse.rf
# In-sample RMSE: 12794.9

#M5P Model
set.seed(1234)
M5p_model<- M5P((SalePrice) ~ PoolQC + GrLivArea  + GarageCars + GarageArea + TotalBsmtSF + FullBath + TotRmsAbvGrd +
    OverallQual + KitchenQual + BsmtQual + OverallCond + X1stFlrSF  +
    PoolArea + ExterQual + MasVnrArea + LotArea + 
    MSSubClass + YearBuilt, data = td)

rmse.m5p <- rmse(td$SalePrice, predict(M5p_model,newdata = td))
rmse.m5p
# In-sample 27532.66

predict_m5p <- predict(M5p_model, td)
mmetric(td$SalePrice, predict_m5p,metric=c("MAE","RMSE","MAPE","RMSPE","RAE","RRSE","COR","R2"))


```

## Model Comparison
This plot compares the RMSEs for all the models and shows the minimum and maximum RMSE of the models. This helped us  finalize the best model.

```{r,echo=F, message=F, include=FALSE, warning=FALSE}
test.rmse <- data.frame( model = c("lm", "ridge", "lasso", "knn", "rf", "m5p"),rmse = c(rmse_linear,rmse.ridge, rmse.lasso, rmse.knn, rmse.rf, rmse.m5p))

test.rmse <- test.rmse[order(test.rmse$rmse, decreasing = TRUE),]

test.rmse$model <- factor(test.rmse$model, levels = test.rmse$model)
```

```{r,fig.cap ="Comparing the models based on their RMSE's", echo=FALSE}
plot(test.rmse, main = "Model Comparison")
```

Looking at the RMSEs of the above models we can say that the Caret model using the random forest method has the lowest RMSE and is the best model.

Checking score on Kaggle
After checking the RMSE score on kaggle for rf and M5P models, we observed that M5P is performing better than rf.

```{r}
ID_data <- test[1]
predict_m5p_test <- predict(M5p_model, test)
dfrm <- data.frame(ID_data = ID_data, predict_m5p_test = predict_m5p_test)
write.table(dfrm, file="kaggle_submission.csv", sep=",", row.names=FALSE, col.names=TRUE)
```

### Kaggle score : 0.14030
### Kaggle Rank :2147

















