
library(ramify)
library(jpeg)
library(caret)
library(OpenImageR)

##DIRECTORY CONTAINING IMAGES DOWNSCALED TO 100X100.
##EACH CLASS CONTAIN 13 Images of different fonts.
KPath = "C:/Users/91828/Documents/Rlab/CharRecog/Ks/"
NPath = "C:/Users/91828/Documents/Rlab/CharRecog/Ns/"
list.files(KPath)
list.files(NPath)
accuracies = data.frame()
ModelTest <- function(p,actual){
  # p<-output$net.result
  preds <- ifelse(p>0.5,1,0)
  tabular<-table(preds,actual)
  print(tabular)
  accuracy <- sum((preds==actual))/length(actual) * 100
  print(accuracy)
  accurateK <- sum(preds==actual & preds==1)/sum(actual==1) *100  
  accurateN <- sum(preds==actual & preds==0)/sum(actual==0) *100
  KPreds <- ifelse(preds==1,"K","N")
  KActual <- ifelse(actual==1,"K","N")
  print(data.frame(predicted = KPreds , actual = KActual, Result = actual==preds))
  print(data.frame("Accuracy in predicting K" = accurateK , "Accuracy in predicting N" = accurateN,"overallAccuracy"=accuracy))
  assign("accuracies", rbind(accuracies,data.frame(Accuracy=accuracy,AccurateK = accurateK, AccurateN= accurateN)),env=.GlobalEnv)
  }

##HELPER FUNCTION TO PLOT IMAGES OF DIRECTORY
plotImage <- function(tst){
  
  if(exists("rasterImage")){
       plot(1:2, type='n')
       rasterImage(tst,1,1,2,2)
  }
}

##INITIALISING DATA FRAME
dataset = data.frame()

##FILENAMES OF font K 
Kfilenames = list.files(KPath)
# par(mfrow=c(4,4))
for(i in Kfilenames[1:length(Kfilenames)]){
  txt <- paste(KPath,i,sep="")
  
  #Read image
  tst <- readJPEG(txt)
  tst <- resizeImage(tst, w = 50, h = 50)
  
  #Converting values to 1 or 0
  tst <- ifelse(tst>0.5,1,0)
  
  #taking only Black pixel values
  #(since image is black and white taking any pixel value will do)
  tst <- tst[,,1]
  plotImage(tst)
  #flattening array from (100,100) to (10000,1)
  tst <- ramify::flatten(tst)
  
  #Adding label to the data, '1' for K and '0' for N.
  tst<-c(1,tst)
  
  #Adding rows to dataset
  dataset<- rbind(dataset,tst)
}
##Below code block does the same for font N(second class).

Nfilenames = list.files(NPath)
# par(mfrow=c(4,4))
for(i in Nfilenames[1:length(Nfilenames)]){
  txt <- paste(NPath,i,sep="")
  tst <- readJPEG(txt)
  tst <- resizeImage(tst, w = 50, h = 50)
  tst <- ifelse(tst>0.5,1,0)
  tst <- tst[,,1]
  plotImage(tst)
  tst <- ramify::flatten(tst)
  tst<-c(0,tst)
  dataset<- rbind(dataset,tst)
}
View(dataset)

#######################
##Splitting data into test and train.
library(caTools)
data1 = sample.split(dataset,SplitRatio=0.6)
train = na.omit(subset(dataset,data1==TRUE),header=FALSE)
test = na.omit(subset(dataset,data1==FALSE),header=FALSE)


#########################
#NeuralNetwork 1 using "rprop+" algorithm with 25 hidden layers
#2 repetitions, error metric "cross-entropy"
library(neuralnet)
set.seed(103)
n <- neuralnet(X1~.,
               data = train,
               hidden = 25,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               rep = 4,
               algorithm = "rprop+",
               stepmax = 100000)
output <- neuralnet::compute(n,rep=4, test)
print("rprop+ with 25layers, 2reps")
ModelTest(output$net.result,test$X1 )
#############################
#NeuralNetwork 1 using "rprop-" algorithm with 30 hidden layers
#2 repetitions, error metric "cross-entropy"
set.seed(103)
n <- neuralnet(X1~.,
               data = train,
               hidden = 30,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               rep = 2,
               algorithm = "rprop-",
               stepmax = 100000)
output <- neuralnet::compute(n,rep=2, test)

print("rprop- with 10layers, 2reps")
ModelTest(output$net.result,test$X1 )
#############################
#NeuralNetwork 1 using "sag" algorithm with 30 hidden layers
#2 repetitions, error metric "cross-entropy"
set.seed(103)
n <- neuralnet(X1~.,
               data = train,
               hidden = 30,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               rep = 2,
               algorithm = "sag",
               stepmax = 100000)
output <- neuralnet::compute(n,rep=2, test)

print("sag with 30layers, 2reps")
ModelTest(output$net.result,test$X1)
###########################
#NeuralNetwork 1 using "backprop" algorithm with 15 hidden layers
#1 repetitions, error metric "cross-entropy"
#learning rate = 0.01
set.seed(103)
n <- neuralnet(X1~.,
               data = train,
               hidden = 15,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               rep = 3,
               algorithm = "backprop",learningrate = 0.01,
               )
output <- neuralnet::compute(n,rep=3, test)

print("backprop with 15layers, 1reps, lr=0.01")
ModelTest(output$net.result,test$X1)

###########################

#NeuralNetwork 1 using "backprop" algorithm with 15 hidden layers
#1 repetitions, error metric "cross-entropy"
#learningrate = 1e-1

set.seed(103)

n <- neuralnet(X1~.,
               data = train,
               hidden = 15,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               rep = 3,
               algorithm = "backprop",learningrate = 1e-2,
)
output <- neuralnet::compute(n,rep=1, test)

print("backprop with 15layers, 1reps, lr= 1e-2")
ModelTest(output$net.result,test$X1 )

#
# ###########################
#NeuralNetwork 1 using "backprop" algorithm with 25 hidden layers
#1 repetitions, error metric "cross-entropy"
#learningrate = 1e-2

set.seed(103)
#
n <- neuralnet(X1~.,
               data = train,
               hidden = 25,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               rep = 3,
               algorithm = "backprop",learningrate = 1e-2,
)
output <- neuralnet::compute(n,rep=3, test)

print("backprop with 25 layers, 3reps, lr=1e-2")
ModelTest(output$net.result,test$X1 )

plot(1:6,accuracies$Accuracy,xlab="Models",ylab="Accuracies",col="red",type="l",ylim=c(20,100))
plot(1:6,accuracies$AccurateK,xlab="Models",ylab="Accuracy in K",col="green",type="l",ylim=c(20,100))
plot(1:6,accuracies$AccurateN,xlab="Models",ylab="Accuracy in N",col="blue",type="l",ylim=c(20,100))

