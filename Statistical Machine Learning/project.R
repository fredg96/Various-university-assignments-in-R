# Compare different methods k-fold cross validation.
set.seed(1)
library(ggplot2)
library(class)
library(MASS)
library(randomForest)

# Load data.
song.data <- read.csv('training_data.csv', header=T)
N <- nrow(song.data)

# Creating matrix with index of songs, each row in the matrix corresponds to a segment.
k.fold = 10
segment.index <- matrix(sample(N),k.fold, N/k.fold)

#
# Creating vectors/matrices with error.

# Logistic regression
# Creating vector with r-values to check. Creating matrix with error, with every validation
# data set (i) at the columns and every r at the rows.
glm.r = (1:99)/100
glm.error.i <- matrix(0, length(glm.r), k.fold)

# k-NN
# Creating vector with k-values to check. Creating matrix with errors, with every validation
# data set (i) at the columns and every k at the rows.
knn.k = 1:200
knn.error.i <- matrix(0, length(knn.k), k.fold)

# LDA
lda.error.i <- c()

# QDA
qda.error.i <- c()

# Random forest
rf.ntree <- 4*(0:50)+1
rf.error.i <- matrix(0, length(rf.ntree), k.fold)


# The loop. Using each segment as validation data, one at a time.
for (i in 1:k.fold) {
  song.val.data <- song.data[segment.index[i,],]
  song.train.data <- song.data[-segment.index[i,],]
  
  # Training model and calculation error of validation data for every method
  
  # Logistic regression
  glm.fit <- glm(formula = label ~ danceability +
                       loudness + speechiness + acousticness +
                       liveness + valence + duration +
                       time_signature,
                     data = song.train.data, family = binomial)
  glm.probs <- predict(object = glm.fit, newdata = song.val.data, type = "response")
  for (j in 1:length(glm.r)) { # Calculating error for every choise of r.
    glm.pred <- rep("dislike",length(glm.probs))
    glm.pred[glm.probs > glm.r[j]] <- "like"
    glm.error.i[j,i] <- mean(glm.pred != song.val.data$label)
    }
  
  # k-NN
  knn.train.X <- scale(as.matrix(song.train.data[c("danceability","loudness",
                                                       "speechiness","acousticness",
                                                       "liveness","valence","duration",
                                                       "time_signature" )]))
  knn.train.Y <- as.matrix(song.train.data[c("label")])
  knn.val.X <- scale(as.matrix(song.val.data[c("danceability","loudness",
                                                   "speechiness","acousticness",
                                                   "liveness","valence","duration",
                                                   "time_signature" )]))
  for (j in 1:length(knn.k)) { # Calculating error for every choise of k.
    knn.pred <- knn(train = knn.train.X, test = knn.val.X, cl = knn.train.Y, k = knn.k[j])
    knn.error.i[j,i] <- mean(knn.pred != song.val.data$label)
    }
  
  # LDA
  lda.fit <- lda(formula = label ~ danceability +
                       loudness + speechiness + acousticness +
                       duration + liveness + valence +
                       time_signature,
                     data = song.train.data, family = binomial)
  lda.val.data <- predict(object = lda.fit, newdata = song.val.data)
  lda.pred <- lda.val.data$class
  lda.error.i[i] <- mean(lda.pred != song.val.data$label)
  
  # QDA
  qda.fit <- qda(formula = label ~ danceability +
                       loudness + speechiness + acousticness +
                       instrumentalness + liveness + valence +
                       time_signature,
                     data = song.train.data, family = binomial)
  qda.val.data <- predict(object = qda.fit, newdata = song.val.data)
  qda.pred <- qda.val.data$class
  qda.error.i[i] <- mean(qda.pred != song.val.data$label)
  
  # Random forest
  for (j in 1:length(rf.ntree)) {
    rf.fit = randomForest(formula = label ~ danceability +
                                loudness + speechiness + acousticness +
                                instrumentalness + liveness + valence +
                                time_signature,
                              data = song.train.data, ntree = rf.ntree[j], mtry = 3, importance = TRUE)
    rf.pred <- predict(rf.fit, song.val.data, type = "class")
    rf.error.i[j, i] <- mean(rf.pred != song.val.data$label)
  }
  
}

# Calculating misclassification error for each method and viasualizing the result.

# Logistic regression
glm.error <- rowMeans(glm.error.i)
qplot(x=glm.r, y=glm.error, geom = "line", xlab = "r", ylab = "Misclassification error")

# k-NN
knn.error <- rowMeans(knn.error.i)
qplot(x=knn.k, y=knn.error, geom = "line", xlab = "k", ylab = "Misclassification error")

# LDA
lda.error <- mean(lda.error.i)
print(paste("The misclassification error for lda is:", lda.error))

# QDA
qda.error <- mean(qda.error.i)
print(paste("The misclassification error for qda is:", qda.error))

# Random forest
rf.error <- rowMeans(rf.error.i)
qplot(x=rf.ntree, y=rf.error, geom = "line", xlab = "ntree", ylab = "Misclassification error")