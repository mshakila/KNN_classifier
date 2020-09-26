########### ASSIGNMENT KNN on glass dataset

### BUSINESS PROBLEM: To classify the type of glass using KNN algorithm

# importing required libraries
library(caTools) # for splitting the dataset into train and test
library(class)  # to run knn model

# loading the dataset
glass <- read.csv('E:\\KNN\\glass.csv')
names(glass)
str(glass)
#   "RI"   "Na"   "Mg"   "Al"   "Si"   "K"    "Ca"   "Ba"   "Fe"   "Type"
#  there are 214 records and 10 variables, 
# "TYPE" is the dependant variable, which we have to find 
# all variables are numeric in nature. Type is of integer, need to convert it
# to factor
glass$Type <- factor(glass$Type)
str(glass) # now Type is factor and has 6 levels
summary(glass)

attach(glass)
head(glass)
tail(glass)

table(Type)
round(prop.table(table(Type))*100,2)
# there are 6 types of glass 

######## Normalize data: since variable values have different ranges, we have to normalize
normalize <- function(x){
  return(
    (x - min(x)) / (max(x) - min(x))
  )
} 
normalize(c(1,2,3,4,5)) # checking if func works properly

glass_norm <- as.data.frame(lapply(glass[1:9], normalize))
summary(glass_norm) # min is 0 and max is 1, all indep vars have been normalized

############# SPLITTING THE DATASET inti 70:30
set.seed(123)
sample <- sample.split(glass$Type,SplitRatio = 0.70)
train_glass <- subset(glass_norm,sample ==TRUE)
test_glass <- subset(glass_norm,sample==FALSE)

train_glass_labels <- subset(glass$Type,sample==TRUE)
test_glass_labels <- subset(glass$Type, sample==FALSE)

###### Running the model
glass_pred <- knn(train = train_glass, test = test_glass, cl = train_glass_labels, k = 15)
glass_pred
test_glass_labels

# creating  table to check accuracy
tab <- table(glass_pred, test_glass_labels)
tab
table(test_glass_labels)
# from table can see that many have been misclassified
accuracy <- ( sum(test_glass_labels == glass_pred) / NROW(test_glass_labels) ) *100
accuracy

# running model with different values of k

i=1
accuracy=1
for (i in 1:50){
  knn.mod <- knn(train=train_glass, test=test_glass, cl=train_glass_labels, k=i)
  accuracy[i] <- ( sum(test_glass_labels == knn.mod) / NROW(test_glass_labels) ) *100
  k=i
  cat(k,'=',accuracy[i],'')
}

#Accuracy plot
plot(accuracy, type="b", xlab="K- Value",ylab="Accuracy level")
# we are acheiving maximum accuracy 0f 70.77% when k=8

######### Improving the model
##### changing the train:test split to 60:40
set.seed(123)
sample <- sample.split(glass$Type,SplitRatio = 0.60)
train_glass <- subset(glass_norm,sample ==TRUE)
test_glass <- subset(glass_norm,sample==FALSE)

train_glass_labels <- subset(glass$Type,sample==TRUE)
test_glass_labels <- subset(glass$Type, sample==FALSE)

i=1
accuracy=1
for (i in 1:30){
  knn.mod <- knn(train=train_glass, test=test_glass, cl=train_glass_labels, k=i)
  accuracy[i] <- ( sum(test_glass_labels == knn.mod) / NROW(test_glass_labels) ) *100
  k=i
  cat(k,'=',accuracy[i],'')
}

plot(accuracy, type="b", xlab="K- Value",ylab="Accuracy level")
# here highest accuracy score is 68.6 at k=8

##### changing the train:test split to 75:25
set.seed(123)
sample <- sample.split(glass$Type,SplitRatio = 0.75)
train_glass <- subset(glass_norm,sample ==TRUE)
test_glass <- subset(glass_norm,sample==FALSE)

train_glass_labels <- subset(glass$Type,sample==TRUE)
test_glass_labels <- subset(glass$Type, sample==FALSE)

i=1
accuracy=1
for (i in 1:30){
  knn.mod <- knn(train=train_glass, test=test_glass, cl=train_glass_labels, k=i)
  accuracy[i] <- ( sum(test_glass_labels == knn.mod) / NROW(test_glass_labels) ) *100
  k=i
  cat(k,'=',accuracy[i],'')
}

plot(accuracy, type="b", xlab="K- Value",ylab="Accuracy level")
# here highest accuracy score is 71.698 at k=23

####### using standardization method
# using the scale() function to (z-score) standardize a data frame
glass_std <- as.data.frame(scale(glass[1:9]))
summary(glass_std) # getting mean of 0
mean(glass_std$RI)
sd(glass_std$RI) # getting mean of 0 and std devn of 1, data has been standardized

set.seed(123)
sample <- sample.split(glass$Type,SplitRatio = 0.70)
train_glass <- subset(glass_std,sample ==TRUE)
test_glass <- subset(glass_std,sample==FALSE)

train_glass_labels <- subset(glass$Type,sample==TRUE)
test_glass_labels <- subset(glass$Type, sample==FALSE)

i=1
accuracy=1
for (i in 1:30){
  knn.mod <- knn(train=train_glass, test=test_glass, cl=train_glass_labels, k=i)
  accuracy[i] <- ( sum(test_glass_labels == knn.mod) / NROW(test_glass_labels) ) *100
  k=i
  cat(k,'=',accuracy[i],'')
}

plot(accuracy, type="b", xlab="K- Value",ylab="Accuracy level")
# when k=3, accuracy increased to 72.31%

'''
CONCLUSIONS
The glass dataset has 6 types of glasses. we have to classify them based
on 9 continuous and independent variables. 
Here we are using KNN algorithm to classify. we have used different k values, 
ranging from 1 to 50. 
Also used normalization and standardization methods to normalize/ standardize 
the data.
We have also used different splitting ratios (75:25, 70:30, 60:40) to split
the data into train and test data.
We have used for-loop to get the accuracy scores based on different k-values.
All these methods have been employed to get the best model. 
'''
