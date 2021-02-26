---
# title: "Statistical Case Studies - Assignment 2, Report 1"
# author: "Qi Wang"
# date: "2/26/2021"
---

##Construct the data.rda file
library(tidyverse)
######load all the federalistpapers into a data.rda file
`1.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/1.txt --- frequentwords.txt", header=FALSE)
`2.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/2.txt --- frequentwords.txt", header=FALSE)
`3.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/3.txt --- frequentwords.txt", header=FALSE)
`4.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/4.txt --- frequentwords.txt", header=FALSE)
`5.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/5.txt --- frequentwords.txt", header=FALSE)
`6.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/6.txt --- frequentwords.txt", header=FALSE)
`7.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/7.txt --- frequentwords.txt", header=FALSE)
`8.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/8.txt --- frequentwords.txt", header=FALSE)
`9.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/9.txt --- frequentwords.txt", header=FALSE)
`10.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/10.txt --- frequentwords.txt", header=FALSE)
`11.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/11.txt --- frequentwords.txt", header=FALSE)
`12.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/12.txt --- frequentwords.txt", header=FALSE)
`13.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/13.txt --- frequentwords.txt", header=FALSE)
`14.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/14.txt --- frequentwords.txt", header=FALSE)
`15.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/15.txt --- frequentwords.txt", header=FALSE)
`16.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/16.txt --- frequentwords.txt", header=FALSE)
`17.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/17.txt --- frequentwords.txt", header=FALSE)
`18.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/18.txt --- frequentwords.txt", header=FALSE)
`19.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/19.txt --- frequentwords.txt", header=FALSE)
`20.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/20.txt --- frequentwords.txt", header=FALSE)
`21.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/21.txt --- frequentwords.txt", header=FALSE)
`22.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/22.txt --- frequentwords.txt", header=FALSE)
`23.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/23.txt --- frequentwords.txt", header=FALSE)
`24.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/24.txt --- frequentwords.txt", header=FALSE)
`25.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/25.txt --- frequentwords.txt", header=FALSE)
`26.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/26.txt --- frequentwords.txt", header=FALSE)
`27.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/27.txt --- frequentwords.txt", header=FALSE)
`28.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/28.txt --- frequentwords.txt", header=FALSE)
`29.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/29.txt --- frequentwords.txt", header=FALSE)
`30.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/30.txt --- frequentwords.txt", header=FALSE)
`31.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/31.txt --- frequentwords.txt", header=FALSE)
`32.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/32.txt --- frequentwords.txt", header=FALSE)
`33.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/33.txt --- frequentwords.txt", header=FALSE)
`34.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/34.txt --- frequentwords.txt", header=FALSE)
`35.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/35.txt --- frequentwords.txt", header=FALSE)
`36.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/36.txt --- frequentwords.txt", header=FALSE)
`37.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/37.txt --- frequentwords.txt", header=FALSE)
`38.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/38.txt --- frequentwords.txt", header=FALSE)
`39.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/39.txt --- frequentwords.txt", header=FALSE)
`40.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/40.txt --- frequentwords.txt", header=FALSE)
`41.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/41.txt --- frequentwords.txt", header=FALSE)
`42.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/42.txt --- frequentwords.txt", header=FALSE)
`43.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/43.txt --- frequentwords.txt", header=FALSE)
`44.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/44.txt --- frequentwords.txt", header=FALSE)
`45.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/45.txt --- frequentwords.txt", header=FALSE)
`46.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/46.txt --- frequentwords.txt", header=FALSE)
`47.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/47.txt --- frequentwords.txt", header=FALSE)
`48.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/48.txt --- frequentwords.txt", header=FALSE)
`49.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/49.txt --- frequentwords.txt", header=FALSE)
`50.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/50.txt --- frequentwords.txt", header=FALSE)
`51.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/51.txt --- frequentwords.txt", header=FALSE)
`52.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/52.txt --- frequentwords.txt", header=FALSE)
`53.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/53.txt --- frequentwords.txt", header=FALSE)
`54.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/54.txt --- frequentwords.txt", header=FALSE)
`55.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/55.txt --- frequentwords.txt", header=FALSE)
`56.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/56.txt --- frequentwords.txt", header=FALSE)
`57.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/57.txt --- frequentwords.txt", header=FALSE)
`58.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/58.txt --- frequentwords.txt", header=FALSE)
`59.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/59.txt --- frequentwords.txt", header=FALSE)
`60.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/60.txt --- frequentwords.txt", header=FALSE)
`61.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/61.txt --- frequentwords.txt", header=FALSE)
`62.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/62.txt --- frequentwords.txt", header=FALSE)
`63.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/63.txt --- frequentwords.txt", header=FALSE)
`64.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/64.txt --- frequentwords.txt", header=FALSE)
`65.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/65.txt --- frequentwords.txt", header=FALSE)
`66.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/66.txt --- frequentwords.txt", header=FALSE)
`67.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/67.txt --- frequentwords.txt", header=FALSE)
`68.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/68.txt --- frequentwords.txt", header=FALSE)
`69.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/69.txt --- frequentwords.txt", header=FALSE)
`70.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/70.txt --- frequentwords.txt", header=FALSE)
`71.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/71.txt --- frequentwords.txt", header=FALSE)
`72.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/72.txt --- frequentwords.txt", header=FALSE)
`73.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/73.txt --- frequentwords.txt", header=FALSE)
`74.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/74.txt --- frequentwords.txt", header=FALSE)
`75.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/75.txt --- frequentwords.txt", header=FALSE)
`76.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/76.txt --- frequentwords.txt", header=FALSE)
`77.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/77.txt --- frequentwords.txt", header=FALSE)
`78.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/78.txt --- frequentwords.txt", header=FALSE)
`79.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/79.txt --- frequentwords.txt", header=FALSE)
`80.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/80.txt --- frequentwords.txt", header=FALSE)
`81.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/81.txt --- frequentwords.txt", header=FALSE)
`82.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/82.txt --- frequentwords.txt", header=FALSE)
`83.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/83.txt --- frequentwords.txt", header=FALSE)
`84.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/84.txt --- frequentwords.txt", header=FALSE)
`85.txt.....frequentwords` <- read.csv("~/Desktop/SCS/Semester2/federalistpapers/85.txt --- frequentwords.txt", header=FALSE)

data<-bind_rows(`1.txt.....frequentwords`, `2.txt.....frequentwords`, `3.txt.....frequentwords`,
                `4.txt.....frequentwords`, `5.txt.....frequentwords`, `6.txt.....frequentwords`,
                `7.txt.....frequentwords`, `8.txt.....frequentwords`, `9.txt.....frequentwords`,
                `10.txt.....frequentwords`, `11.txt.....frequentwords`, `12.txt.....frequentwords`,
                `13.txt.....frequentwords`, `14.txt.....frequentwords`, `15.txt.....frequentwords`,
                `16.txt.....frequentwords`, `17.txt.....frequentwords`, `18.txt.....frequentwords`,
                `19.txt.....frequentwords`, `20.txt.....frequentwords`, `21.txt.....frequentwords`,
                `22.txt.....frequentwords`, `23.txt.....frequentwords`, `24.txt.....frequentwords`,
                `25.txt.....frequentwords`, `26.txt.....frequentwords`, `27.txt.....frequentwords`,
                `28.txt.....frequentwords`, `29.txt.....frequentwords`, `30.txt.....frequentwords`,
                `31.txt.....frequentwords`, `32.txt.....frequentwords`, `33.txt.....frequentwords`, 
                `34.txt.....frequentwords`, `35.txt.....frequentwords`, `36.txt.....frequentwords`, 
                `37.txt.....frequentwords`, `38.txt.....frequentwords`, `39.txt.....frequentwords`,
                `40.txt.....frequentwords`, `41.txt.....frequentwords`, `42.txt.....frequentwords`,
                `43.txt.....frequentwords`, `44.txt.....frequentwords`, `45.txt.....frequentwords`,
                `46.txt.....frequentwords`, `47.txt.....frequentwords`, `48.txt.....frequentwords`,
                `49.txt.....frequentwords`, `50.txt.....frequentwords`, `51.txt.....frequentwords`,
                `52.txt.....frequentwords`, `53.txt.....frequentwords`, `54.txt.....frequentwords`,
                `55.txt.....frequentwords`, `56.txt.....frequentwords`, `57.txt.....frequentwords`,
                `58.txt.....frequentwords`, `59.txt.....frequentwords`, `60.txt.....frequentwords`,
                `61.txt.....frequentwords`, `62.txt.....frequentwords`,
                `63.txt.....frequentwords`, `64.txt.....frequentwords`, `65.txt.....frequentwords`,
                `66.txt.....frequentwords`, `67.txt.....frequentwords`, `68.txt.....frequentwords`,
                `69.txt.....frequentwords`, `70.txt.....frequentwords`, `71.txt.....frequentwords`,
                `72.txt.....frequentwords`, `73.txt.....frequentwords`, `74.txt.....frequentwords`,
                `75.txt.....frequentwords`, `76.txt.....frequentwords`, `77.txt.....frequentwords`,
                `78.txt.....frequentwords`, `79.txt.....frequentwords`, `80.txt.....frequentwords`,
                `81.txt.....frequentwords`, `82.txt.....frequentwords`, `83.txt.....frequentwords`,
                `84.txt.....frequentwords`, `85.txt.....frequentwords`)
save(data,file="data.rda")

#load data.rda & fpauthors.txt
library(tidyverse)
fpauthors <- read.table("~/Desktop/SCS/Semester2/fpauthors.txt",skip=0,sep=',',header=FALSE)
fp<- as.data.frame(t(fpauthors))
data_fp<-bind_cols(data,fp)
data_known<-filter(data_fp,V110>0) # data with known papers
data_1<-filter(data_fp,V110=="1") # data with numebr 1
data_2<-filter(data_fp,V110=="2") # data with numebr 2
data_3<-filter(data_fp,V110=="3") # data with numebr 3
data_unknown<-filter(data_fp,V110=="-1") # data with unknown papers
data_joint<-filter(data_fp,V110=="-2") # data with joint papers

Data_Hamilton<-data_1 %>%                        # adding the function word count for Hamilton
  replace(is.na(.), 0) %>%
  summarise_all(sum)
Data_Jay<-data_2 %>%                        # adding the function word count for Jay
  replace(is.na(.), 0) %>%
  summarise_all(sum)
Data_Madison<-data_3 %>%                        # adding the function word count for Madison
  replace(is.na(.), 0) %>%
  summarise_all(sum)
Alldata<- rbind(Data_Hamilton, Data_Jay, Data_Madison)
Alldata[Alldata < 1] <- 1
training_set<- as.matrix(Alldata[, -72])     # training set for 3x71 matrix
test_set<- as.matrix(data_unknown[, -72])

###Find the accuracy and error rate, using Cross Validation
set.seed(1) #initialise random number generator
library(mvtnorm) #for multivariate normal density
library(class)
library(e1071)
library(caret)
data_known <- as.data.frame(data_known)
##### Cross validation
#To use 10-fold cross validation
trControl <- trainControl(method  = "cv",
                          number  = 10)
#To evaluate the accuracy of the KNN classifier with different values of k by cross validation
fit <- train(V110 ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:10),
             trControl  = trControl,
             data       = data_known)
#shows the results and plot the accuracies for different values of k
fit
fit$results
plot(fit$results$k,fit$results$Accuracy,
     main="Accuracies for different values of k",
     ylab="accuracies",
     xlab="k")
#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was k = 1.

#######Find the accuracy for discriminant analysis
set.seed(1) #initialise random number generator
library(mvtnorm) #for multivariate normal density
library(class)

data_known <- as.data.frame(data_known)
C <- 72 #index of class label
trainindices <- seq(3,nrow(data_known),by=3)
train <- as.matrix(data_known[-trainindices,-C])
test <- as.matrix(data_known[trainindices,-C])
trainlabel <- data_known[-trainindices,C]
testlabel <- data_known[trainindices,C]

A <- nrow(train)
B <- nrow(test)
prior <-rep(1/A, B*A)
dim(prior) <- c(B,A)
thetas <- train
for (i in 1:nrow(thetas)) {
  thetas[i,]<- thetas[i,]/sum(thetas[i,])
  
  posterior <- matrix(0:0, nrow = B, ncol = A)
  for (j in 1:B){
    for (i in 1:A) {
      test[j,]<-test[j,]
      prior[j,i] <- prior[j,i]
      posterior[j,i]<-log(prior[j,i]) + dmultinom(test[j,], prob=thetas[i,], log=TRUE)
    }}
}
discriminantpreds <- apply(posterior,1,which.max)  #allocate to class for which posterior is maxmised
discriminantpreds <-as.matrix(discriminantpreds)

D<-nrow(discriminantpreds)
label <-matrix(0:0, nrow = D, ncol = 1)
for (y in 1:D){
  label[y,1]<-data_known[discriminantpreds[y,],72]
}

label<-as.numeric(label)
discriminantaccuracy<- sum(label==testlabel)/length(testlabel)
discriminantaccuracy

###Predictions for the unknown papers for Discriminant Analysis
###Discriminant Analysis
A <- nrow(training_set)
prior <-rep(1/A, 12*A)
dim(prior) <- c(12,3)
thetas <- training_set
for (i in 1:nrow(thetas)) {
  thetas[i,]<- thetas[i,]/sum(thetas[i,])
}

B <- nrow(test_set)
posterior <- matrix(0:0, nrow = B, ncol = A)
for (j in 1:B){
  for (i in 1:A) {
    test_set[j,]<-test_set[j,]
    prior[j,i] <- prior[j,i]
    posterior[j,i]<-log(prior[j,i]) + dmultinom(test_set[j,], prob=thetas[i,], log=TRUE)
  }}
posterior # the biggest number is the authors' paper

### Predicting the unknown papers 
### KNN analysis (concatenated)
#convert counts to proportions
for (i in 1:nrow(training_set)) {
  training_set[i,] <- training_set[i,]/sum(training_set[i,])
}
for (j in 1:nrow(test_set)) {
  test_set[j,] <- test_set[j,]/sum(test_set[j,])
}
#normalise features
mus <- apply(training_set,2,mean)
sds <- apply(training_set,2,sd)

for (i in 1:nrow(training_set)) {
  training_set[i,] <- (training_set[i,]-mus)/sds
}

for (j in 1:nrow(test_set)) {
  test_set[j,] <- (test_set[j,]-mus)/sds
}

#compute L2 norm (euclidean distance) for each author
dists <- matrix(0:0, nrow = B, ncol = A)
for (j in 1:B) {
  for (i in 1:A) {
    dists[j,i] <-sqrt(sum((training_set[i,] - test_set[j,])^2))
  }}
dists # the lowest distance in each row is the author's paper

###KNN analysis (non-concatenated) 
rm(list=ls()) #eliminate the data before and run again
#load data.rda & fpauthors.txt
library(tidyverse)
fpauthors <- read.table("~/Desktop/SCS/Semester2/fpauthors.txt",skip=0,sep=',',header=FALSE)
fp<- as.data.frame(t(fpauthors))
data_fp<-bind_cols(data,fp)
data_known<-filter(data_fp,V110>0) # data with known papers
data_unknown<-filter(data_fp,V110=="-1")

training_set<- as.matrix(data_known[, -72])
test_set<- as.matrix(data_unknown[, -72])
### KNN
#convert counts to proportions
for (i in 1:nrow(training_set)) {
  training_set[i,] <- training_set[i,]/sum(training_set[i,])
}
for (j in 1:nrow(test_set)) {
  test_set[j,] <- test_set[j,]/sum(test_set[j,])
}
#normalise features
mus <- apply(training_set,2,mean)
sds <- apply(training_set,2,sd)

for (i in 1:nrow(training_set)) {
  training_set[i,] <- (training_set[i,]-mus)/sds
}

for (j in 1:nrow(test_set)) {
  test_set[j,] <- (test_set[j,]-mus)/sds
}

#compute L2 norm (euclidean distance) for each author
A <- nrow(training_set)
B <- nrow(test_set)
dists <- matrix(0:0, nrow = B, ncol = A)
for (j in 1:B) {
  for (i in 1:A) {
    dists[j,i] <-sqrt(sum((training_set[i,] - test_set[j,])^2))
  }}
dists
paper <- apply(dists,1,which.min)
paper
paper <-as.matrix(paper)
D<-nrow(paper)
label <-matrix(0:0, nrow = D, ncol = 1)
for (y in 1:D){
  label[y,1]<-data_known[paper[y,],72]
}

label<-as.numeric(label)
label


#### Calculate the Brier Score for knn 
#(as on the lecture, the probability of 1 to the class it chose for an observation, and a probability of 0 to all other classes)
rm(list=ls()) #eliminate the data before and run again
#load data.rda & fpauthors.txt
library(tidyverse)
fpauthors <- read.table("~/Desktop/SCS/Semester2/fpauthors.txt",skip=0,sep=',',header=FALSE)
fp<- as.data.frame(t(fpauthors))
data_fp<-bind_cols(data,fp)
data_known<-filter(data_fp,V110>0)
a<-as.data.frame(data_known[,72])
b<-cbind(matrix(0:0,70,3),a)

attach(b)
names(b)[1]<-paste("H")
names(b)[2]<-paste("J")
names(b)[3]<-paste("M") 
names(b)[4]<-paste("index") 
b<-as.data.frame(b)
b$H[which(b$index == "1")] = "1"
b$J[which(b$index == "2")] = "1"
b$M[which(b$index == "3")] = "1"


b$y[which(b$index == "1")] = "0"
b$y[which(b$index == "2")] = "1"
b$y[which(b$index == "3")] = "4"

z<-as.matrix(b[,5])
z<-as.numeric(z)
sum(z)/70

