library(tm)
library(RWeka)
library(magrittr)
library(Matrix)
library(glmnet)
library(ROCR)
library(ggplot2)

# Reading the data
cndjia <- read.csv('Combined_News_DJIA.csv', stringsAsFactors = FALSE)

# Making 'Date' column a Date object
cndjia$Date <- as.Date(cndjia$Date)

# Combining headlines into a single text column for each day and adding sentence separation token
cndjia$all <- paste(cndjia$Top1, cndjia$Top2, cndjia$Top3, cndjia$Top4, cndjia$Top5, cndjia$Top6,
                  cndjia$Top7, cndjia$Top8, cndjia$Top9, cndjia$Top10, cndjia$Top11, cndjia$Top12, 
                  cndjia$Top13, cndjia$Top14, cndjia$Top15, cndjia$Top16, cndjia$Top17, cndjia$Top18,
                  cndjia$Top19, cndjia$Top20, cndjia$Top21, cndjia$Top22, cndjia$Top23, cndjia$Top24,
                  cndjia$Top25, sep=' <s> ')

# Removing b's and backslashes 
cndjia$all <- gsub('b"|b\'|\\\\|\\"', "", cndjia$all)

# Removing punctuations except headline separators
cndjia$all <- gsub("([<>])|[[:punct:]]", "\\1", cndjia$all)

# Keeping the required columns and deleting rest 
cndjia <- cndjia[, c('Date', 'Label', 'all')]

# exclude stopwords and headline tokens
control <- list(removeNumbers = TRUE, tolower = TRUE, 
                stopwords = c(stopwords(kind = 'SMART'), '<s>'))

dtm <- Corpus(VectorSource(cndjia$all)) %>% 
  DocumentTermMatrix(control=control)

split_index <- cndjia$Date <= '2014-12-31'

ytrain <- as.factor(cndjia$Label[split_index])
xtrain <- Matrix(as.matrix(dtm)[split_index, ], sparse=TRUE)

ytest <- as.factor(cndjia$Label[!split_index])
xtest <- Matrix(as.matrix(dtm)[!split_index, ], sparse=TRUE)

# Train the model
glmnet.fit <- cv.glmnet(x=xtrain, y=ytrain, family='binomial', alpha=0)

# Generate predictions
preds <- predict(glmnet.fit, newx=xtest, type='response', s='lambda.min')

# Put results into dataframe for plotting.
results <- data.frame(pred=preds, actual=ytest)

ggplot(results, aes(x=preds, color=actual)) + geom_density()

prediction <- prediction(preds, ytest)
perf <- performance(prediction, measure = "tpr", x.measure = "fpr")

auc <- performance(prediction, measure = "auc")
auc <- auc@y.values[[1]]
auc

roc.data <- data.frame(fpr=unlist(perf@x.values),
tpr=unlist(perf@y.values))

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
geom_ribbon(alpha=0.2) +
geom_line(aes(y=tpr)) +
geom_abline(slope=1, intercept=0, linetype='dashed') +
ggtitle("ROC Curve") +
ylab('True Positive Rate') +
xlab('False Positive Rate')

# Necessary to set this option on Linux machines, otherwise the NGrameTokenizer will cause our 
# DocumentTermMatrix call to hang. 
options(mc.cores=1)

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

control <- list(tokenize=BigramTokenizer,
bounds = list(global = c(20, 500)))

dtm <- Corpus(VectorSource(cndjia$all)) %>%
tm_map(removeNumbers) %>%
tm_map(stripWhitespace) %>%
tm_map(content_transformer(tolower)) %>%
DocumentTermMatrix(control=control)

split_index <- cndjia$Date <= '2014-12-31'

ytrain <- as.factor(cndjia$Label[split_index])
xtrain <- Matrix(as.matrix(dtm)[split_index, ], sparse=TRUE)

ytest <- as.factor(cndjia$Label[!split_index])
xtest <- Matrix(as.matrix(dtm)[!split_index, ], sparse=TRUE)

# Train the model
glmnet.fit <- cv.glmnet(x=xtrain, y=ytrain, family='binomial', alpha=0)

# Generate predictions
preds <- predict(glmnet.fit, newx=xtest, type='response', s="lambda.min")

# Put results into dataframe for plotting.
results <- data.frame(pred=preds, actual=ytest)

ggplot(results, aes(x=preds, color=actual)) + geom_density()

prediction <- prediction(preds, ytest)
perf <- performance(prediction, measure = "tpr", x.measure = "fpr")

auc <- performance(prediction, measure = "auc")
auc <- auc@y.values[[1]]
auc

roc.data <- data.frame(fpr=unlist(perf@x.values),
tpr=unlist(perf@y.values))

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
geom_ribbon(alpha=0.2) +
geom_line(aes(y=tpr)) +
geom_abline(slope=1, intercept=0, linetype='dashed') +
ggtitle("ROC Curve") +
ylab('True Positive Rate') +
xlab('False Positive Rate')

library(randomForest)
set.seed(32) 
rf <-randomForest(y=ytrain, x=as.data.frame(as.matrix(xtrain)), ntree=9, 
                  na.action=na.exclude, importance=T,proximity=T) 

library(caret)
predicted_values = predict(rf, type = "prob", xtest)
head(predicted_values)
threshold <- 0.5 
pred <- factor( ifelse(predicted_values[,2] > threshold, "0", "1") )
head(pred)
levels(ytest)[2]
confusionMatrix(pred, ytest, positive = levels(ytest)[2])

library(ROCR)
predicted_values <- predict(rf, xtest,type= "prob")[,1] 
pred <- prediction(predicted_values, ytest)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc


roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="RF")
ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))
