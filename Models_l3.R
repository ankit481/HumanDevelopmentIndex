# HDI_Script1
# install.packages("yaImpute")
library(dplyr)
library(plyr)
library(mice)
library(xlsx)
library(yaImpute)

setwd("C:/Dhruv/CMU/F15/DM/Team G/Team G/")
input <- read.csv("HDI_final.csv", stringsAsFactors = F)
input$unemployment_rate <- NULL
row.names(input) <- input$Country

num_vars <- sapply(input, is.character)
num_vars[["Country"]] <- FALSE
num_vars[["HDI_Label"]] <- FALSE
input2 <- input
input2[num_vars] <- lapply(input2[num_vars], as.numeric)
# input2 <- input2[which(input2$Country!="Liechtenstein"),]

complete_data <- input2[complete.cases(input2),]

# knn_inp <- input2
# knn_inp <- complete_equal
# knn_inp <- input2
# knn_inp$HDI_Label <- NULL
# knn_inp$HDI <- NULL
# set.seed(123)
# knn_inp$rand <- runif(nrow(knn_inp))
# knn_inp <- knn_inp[order(knn_inp$rand),]
# knn_inp$rand <- NULL
# knn_inp <- knn_inp[complete.cases(knn_inp),]
# knn_inp$Country <- NULL
# knn_train <- knn_inp[1:120,]
# knn_test <- knn_inp[121:nrow(knn_inp),]
# knn_train_cl <- knn_train$HDI_Label
# knn_test_cl <- knn_test$HDI_Label
# knn_train$HDI_Label <- NULL
# knn_test$HDI_Label <- NULL


library("FNN")
# knn1 <- knn(train=knn_train, test = knn_test, cl = knn_train_cl, k = 5)
# knn1 <- knn(train=knn_train, test = knn_train, cl = knn_train_cl, k = 5)
# lev <- levels(knn1)
# attributes(knn1) <- NULL
# 
# plot(knn1)

input3 <- input[complete.cases(input),]
input3 <- complete_data
input3$HDI <- NULL
input3$Country <- NULL


get_pred_knn <- function (train, test, k, formula=NULL)
{
  last_col <- colnames(train)[ncol(train)]
  # print(last_col)
  output <- as.vector(train[[last_col]])
  # print(output)
  train2 <- train[,1:ncol(train)-1]
  test2 <- test[,1:ncol(test)-1]
  knn_model <- FNN::knn(train = train2, test = test2, k = k, cl = output)
  pred_class <- knn_model
  # attributes(pred_class) <- NULL
  # print(pred_class)
  return_df <- data.frame(prediction = pred_class, true_output=test[[last_col]])
  colnames(return_df) <- c("prediction", "true_output")
  return(return_df)
}

get_pred_dectree <- function (train, test, formula)
{
  last_col <- colnames(train)[ncol(train)]
  # print(last_col)
  output <- as.vector(train[[last_col]])
  # formula <- as.formula(paste(last_col, "~.", sep=""))
  # print(formula)
  # print(output)
  train2 <- train[,1:ncol(train)]
  test2 <- test[,1:ncol(test)]
  # dtree_model <- FNN::knn(train = train2, test = test2, k = k, cl = output)
  dtree_model <- ctree(formula, data=train2)
  pred_class <- predict(dtree_model, newdata=test2)
  # row.names(pred_class) <- row.names(test2)
  # attributes(pred_class) <- NULL
  # print(pred_class)
  return_df <- data.frame(prediction = pred_class, true_output=test[[last_col]])
  row.names(return_df) <- row.names(test2)
  colnames(return_df) <- c("prediction", "true_output")
  return(return_df)
}

get_pred_lm <- function (train, test, formula)
{
  last_col <- colnames(train)[ncol(train)]
  # print(last_col)
  output <- as.vector(train[[last_col]])
  if (formula==FALSE)
    formula <- as.formula(paste(last_col, "~.", sep=""))
  # print(formula)
  # print(output)
  train2 <- train[,1:ncol(train)]
  test2 <- test[,1:ncol(test)]
  # dtree_model <- FNN::knn(train = train2, test = test2, k = k, cl = output)
  lm_model <- lm(formula, data=train2)
  pred_lm <- predict(lm_model, newdata=test2)
  # attributes(pred_class) <- NULL
  # print(pred_class)
  return_df <- data.frame(prediction = pred_lm, true_output=test[[last_col]])
  colnames(return_df) <- c("prediction", "true_output")
  return(return_df)
}

get_pred_rf <- function (train, test, formula)
{
  last_col <- colnames(train)[ncol(train)]
  # print(last_col)
  output <- as.vector(train[[last_col]])
  if (formula==FALSE)
    formula <- as.formula(paste(last_col, "~.", sep=""))
  # print(formula)
  # print(output)
  train2 <- train[,1:ncol(train)]
  test2 <- test[,1:ncol(test)]
  rf <- randomForest(formula=formula, data=train2, ntree=1000, mtry = 5, importance=TRUE, replace=FALSE, na.action = na.omit, keep.forest = T, predict.all=T)
  # lm_model <- lm(formula, data=train2)
  pred_rf <- predict(rf, newdata=test2)
  # attributes(pred_class) <- NULL
  # print(pred_class)
  return_df <- data.frame(prediction = pred_rf, true_output=test[[last_col]])
  colnames(return_df) <- c("prediction", "true_output")
  return(return_df)
}

do_cv_class <- function(df, num_folds, model_name, formula=FALSE)
{
  isnum_isfac <- function(x) {
    return (is.numeric(x) | is.factor(x))
  }
  df <- df[sapply(df, isnum_isfac)]
  #Ordering the dataset using the runif function
  set.seed(123)
  random_sort_vec <- runif(nrow(df))
  df <- df[order(random_sort_vec),]
  
  k <- num_folds
  
  #Calculate number of values in each fold
  split_vals <- floor(nrow(df)/k)
  
  #rem is the last group, or the elements which do not fit in any group because the
  #data might not be a perfect multiple of k
  rem <- rep(k, nrow(df) - split_vals*k)
  
  #Set an index vector that assigns a value of k for every split_vals
  spl_vec <- rep(1:k, each=split_vals)
  
  #Adding the additional rem vector values and converting to a data frame
  spl_vec <- data.frame(split=c(spl_vec, rem))
  
  if (grepl(pattern = "nn", model_name)) {
    model <- "knn"
    knn_k <- as.numeric(gsub(pattern = "nn", replacement = "", x = model_name, ignore.case = TRUE))
  } else {
    model <- model_name
  }
  
  for (i in 1:k)
  {
    keep_rows <- spl_vec != i
    train_temp <- df[keep_rows==TRUE,]
    test_temp <- df[keep_rows!=TRUE,]
    pred_ret_t <- if (model == "knn")
    {
      get_pred_knn(train = train_temp, test=test_temp, k = knn_k, formula=formula)
    } else if (model == "dtree")
    {
      get_pred_dectree(train=train_temp, test=test_temp, formula=formula)
    } else if (model == "lm")
    {
      get_pred_lm(train=train_temp, test=test_temp, formula = formula);
    } else if (model == "rf")
    {
      get_pred_rf(train=train_temp, test=test_temp, formula=formula)
    }
    
    if (!exists("do_cv_op"))
      do_cv_op <- pred_ret_t
    else
      do_cv_op <- rbind.data.frame(do_cv_op, pred_ret_t)
    }
  return(do_cv_op)
}


input4 <- input3
input4$HDI_Label <- as.factor(input4$HDI_Label)
do1 <- do_cv_class(df = input4,num_folds = nrow(input4), model_name = "3nn", formula=NULL)
str(do1)
do1_op <- do1
do1_op$Check <- ifelse(do1_op$prediction == do1_op$true_output, "Match", "No Match")
table(do1_op$Check)
round(table(do1_op$Check)/sum(table(do1_op$Check))*100)
plot(do1_op$prediction)
plot(do1_op$true_output)
colnames(do1_op) <- c("Predicted", "True Output", "Check")

# library(rattle)
# rattle()

mydata <- complete_data
mydata$rand <- runif(nrow(mydata))
mydata <- mydata[order(mydata$rand),]
mydata$rand <- NULL

library(partykit)
tree_inp <- mydata
tree_inp$HDI <- NULL
tree_inp$Country <- NULL
tree_inp$HDI_Label <- as.factor(tree_inp$HDI_Label)
tree_inp$HDI_Label <- factor(tree_inp$HDI_Label, levels = c("Low", "Medium", "High", "Very High"))
tree <- ctree(HDI_Label ~ ., data=tree_inp)
plot(tree)
plot(tree, type="simple", main = "Decision Tree for HDI Classification",cex=25, terminal_panel=node_terminal, 
     tp_args = c (FUN=function(node) c(paste("y=", node$prediction,sep=""), 
                                       paste("n=", node$n, sep=""))), top=8.5, width=5, gp=gpar(fontsize=15))

pred.response <- factor(as.character(predict(tree), tree_inp), levels = c("Low", "Medium", "High", "Very High")) # predict on test data
tree_pred <- data.frame(Original=tree_inp$HDI_Label, Pred = pred.response)
row.names(tree_pred) <- row.names(tree_inp)
tree_pred_table <- table(Original = tree_pred$Original, Predicted = tree_pred$Pred)
tree_pred_table


#Regression_Tree Attempt
tree_inp2 <- mydata
tree_inp2$HDI_Label <- NULL
tree_inp2$Country <- NULL
rtree_test <- ctree(HDI ~ ., data=tree_inp2)

# install.packages("randomForest")
library(randomForest)
rf <- randomForest(HDI_Label ~ ., data=tree_inp, ntree=1000, mtry = 5, importance=TRUE, replace=FALSE, na.action = na.omit)
round(rf$confusion, 2)
temp <- rf$predicted
importance <- data.frame(importance(rf))
varImpPlot(rf, type=2)
importance <- importance
layout(matrix(c(1,2),nrow=1),width=c(4,1))
plot(rf, log="y")
par(mar=c(5,0,4,2)) #No margin on the left side
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(rf$err.rate),col=1:4,cex=0.8,fill=1:4)

rf$frame

rf_tree <- getTree(rf, k = 1, labelVar = TRUE)
rft <- to.dendrogram(rf_tree)
plot(rft, log="y")



#### RF DO CV
rf_input <- complete_data
rf_input$Country <- NULL
rf_input_c <- rf_input
rf_input_c$HDI_Label <- NULL

rf_do_cv_c <- do_cv_class(df = rf_input_c, num_folds = 10, model_name = "rf")
rf_do_cv_c_class <- get_labels(rf_do_cv_c)
get_metrics_acc(rf_do_cv_c_class)


#Trying multinomial logistic regression
# install.packages("nnet")
library(nnet)
mlog_inp <- mydata
mlog_inp$HDI <- NULL
row.names(mlog_inp) <- mlog_inp$Country
mlog_inp$Country <- NULL
mlog_inp$HDI_Label <- as.factor(tree_inp$HDI_Label)
mlog_inp$HDI_Label <- factor(tree_inp$HDI_Label, levels = c("Low", "Medium", "High", "Very High"))

mlog <- multinom(HDI_Label~., data=mlog_inp)
plot(mlog)
predict(mlog, mlog_inp, "probs")
mlog_fit <- cbind.data.frame(mlog_inp, predict(mlog, mlog_inp))
colnames(mlog_fit) <- c(colnames(mlog_inp), "Predicted")
table(mlog_fit$HDI_Label, mlog_fit$Predicted)




#Decision Tree, K fold
tree_k <- do_cv_class(df=tree_inp, num_folds = 10, model_name = "dtree", formula=as.formula("HDI_Label ~ ."))
table(Original= tree_k$true_output, Pred=tree_k$prediction)




#Knn K fold
nn_mod <- do_cv_class(df=tree_inp, num_folds = 10, model_name = "5nn")
table(Original= nn_mod$true_output, Pred=nn_mod$prediction)

#Stepwise
library(MASS)
HDI_Data <- complete_data
HDI_Data$HDI_Label <- NULL
HDI_Data$Country <- NULL
fit_lm<-lm(HDI ~ ., HDI_Data)
  
lm_kf <- do_cv_class(HDI_Data, num_folds = 10, model_name = "lm", formula=HDI ~ 
                       primary_edu + tertiary_edu + mortality_adult_female + mortality_adult_male + life_expectancy +
                       total_population + urban_population + median_age + remittance)

lm_kf <- do_cv_class(HDI_Data, num_folds = 10, model_name = "lm")
head(lm_kf, n=2)


step_fit_lm <- stepAIC(fit_lm, direction="both")

step_fla <- as.formula(HDI ~ 
                         primary_edu + tertiary_edu + mortality_adult_female + mortality_adult_male + life_expectancy +
                         total_population + urban_population + median_age + remittance)

# new_fit_lm<-lm(HDI ~ 
#                  primary_edu + tertiary_edu + mortality_adult_female + mortality_adult_male + life_expectancy +
#                  total_population + urban_population + median_age + remittance, data= HDI_Data)

tile_class <- function(x)
{
  if (x == 4)
    return ("Very High") else if (x == 3)
      return ("High") else if (x == 2)
        return ("Medium") else if (x == 1)
          return ("Low")
}

lm_kf$prediction <- round(lm_kf$prediction, 2)
lm_kf$residuals <- lm_kf$true_output - lm_kf$prediction
lm_kf$Country <- row.names(lm_kf)
lm_kf_op <- merge(lm_kf, subset(complete_data, select = c("Country", "HDI_Label")), by="Country")
lm_kf_op <- lm_kf_op[order(lm_kf_op$prediction, decreasing = TRUE),]
lm_kf_op$PredClass_tile <- ntile(lm_kf_op$prediction, 4)
lm_kf_op$PredClass <- sapply(lm_kf_op$PredClass_tile, tile_class)

table(lm_kf_op$HDI_Label, lm_kf_op$PredClass)

get_labels <- function  (x)
{
  x$prediction <- round(x$prediction, 2)
  x$residuals <- x$true_output - x$prediction
  x$Country <- row.names(x)
  x_op <- merge (x, subset(complete_data, select = c("Country", "HDI_Label")), by="Country")
  x_op <- x_op[order(x$prediction, decreasing=TRUE),]
  x_op$PredClass_tile <- ntile(x_op$prediction, 4)
  x_op$PredClass <- sapply(x_op$PredClass_tile, tile_class)
  return(x_op)
}

get_metrics_acc_reg <- function(x)
{
  table_t <- table(x$HDI_Label, x$PredClass)
  print(table_t)
  acc <- sum(table_t[1,1], table_t[2,2], table_t[3,3], table_t[4,4])/sum(table_t)
  print(paste("Accuracy = ", round(acc,4), sep=""))
  return (c(acc))
}

get_metrics_acc_cls <- function(x)
{
  table_t <- table(x)
  print(table_t)
  acc <- sum(table_t[1,1], table_t[2,2], table_t[3,3], table_t[4,4])/sum(table_t)
  print(paste("Accuracy = ", round(acc,4), sep=""))
  return (c(acc))
}



########
#RUNNING MODELS
########

#CLASSIFICATION
data <- complete_data
data$HDI <- NULL
data$Country <- NULL
data$HDI_Label <- as.factor(data$HDI_Label)

#DECISION TREE
dtree_k <- do_cv_class(df = data, num_folds = 10, model_name = "dtree", formula = as.formula("HDI_Label ~ ."))
print("CLASSIFICATION DTREE")
get_metrics_acc_cls(dtree_k)
dtree_vec <- c("Decision Tree", get_metrics_acc_cls(dtree_k),0)

#RANDOM FOREST
rfor_k <- do_cv_class(df=data, num_folds=10, model_name = "rf", formula = as.formula("HDI_Label ~ ."))
print("CLASSIFICATION RANDOM FOREST")
get_metrics_acc_cls(rfor_k)
rforest_vec <- c("Random Forest", get_metrics_acc_cls(rfor_k),0)

#KNN
knn_k <- do_cv_class(df=data, num_folds = 10, model_name = "5nn", formula = as.formula("HDI_Label ~ ."))
print("CLASSIFICATION KNN")
get_metrics_acc_cls(knn_k)
knn_vec <- c("KNN", get_metrics_acc_cls(knn_k),0)



############
#REGRESSION
############

data <- complete_data
data$HDI_Label <- NULL
data$Country <- NULL

#REGRESSION TREE
rtree_k <- do_cv_class(df=data, num_folds = 10, model_name = "dtree", formula=as.formula("HDI ~ ."))
rtree_k_lab <- get_labels(rtree_k)
rtree_rsq <- cor(rtree_k$prediction, rtree_k$true_output)^2
print("REGRESSION TREE")
get_metrics_acc_reg(rtree_k_lab)
rtree_vec <- c("Regression Tree", get_metrics_acc_reg(rtree_k_lab),rtree_rsq)

#REGRESSION FOREST
rforest_k <- do_cv_class(df=data, num_folds = 10, model_name = "rf", formula=as.formula("HDI ~ ."))
rforest_k <- get_labels(rforest_k)
rforest_rsq <- cor(rforest_k$prediction, rforest_k$true_output)^2
print("REGRESSION FOREST")
get_metrics_acc_reg(rforest_k)
rrforest_vec <- c("Regression Forest", get_metrics_acc_reg(rforest_k),rforest_rsq)

#LINEAR REGRESSION
lm_k <- do_cv_class(df=data, num_folds = 10, model_name = "lm", formula=step_fla)
lm_k <- get_labels(lm_k)
lm_rsq <- cor(lm_k$prediction, lm_k$true_output)^2
print("LINEAR REGRESSION")
get_metrics_acc_reg(lm_k)
lm_vec <- c("Linear Regression", get_metrics_acc_reg(lm_k),lm_rsq)

result <- data.frame(rbind(dtree_vec, rforest_vec, knn_vec,rtree_vec, rrforest_vec, lm_vec))
colnames(result) <- c("Model", "Accuracy", "R-Squared")
row.names(result) <- NULL


temp2 <- getTree(rf, k=600, labelVar = T)







