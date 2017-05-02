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

# set this option on Linux machines or DocumentTermMatrix call may cause hang. 
options(mc.cores=1)

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

control <- list(tokenize=BigramTokenizer,stopwords = c(stopwords(kind = 'SMART'),bounds = list(global = c(20, 500))))

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

roc.data <- data.frame(fpr=unlist(perf@x.values), tpr=unlist(perf@y.values), model="RF")

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))
