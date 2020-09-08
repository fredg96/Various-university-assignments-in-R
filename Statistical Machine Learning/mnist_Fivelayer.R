
rm(list=ls())
library(tensorflow)

# Load mnist data
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("data", one_hot = TRUE, reshape=FALSE)

# The sign L specifyies that it is an integer
Xim <- tf$placeholder(dtype = tf$float32, shape = shape(NULL, 28L, 28L, 1L))

X1 <- tf$reshape(tensor = Xim, shape = shape(-1L, 784L))

# The placeholder of the output. Hotfix encoding of the class
# [1 0 0 0 0 0 0 0 0 0] is a "0", [0 1 0 0 0 0 0 0 0 0] is a "1" and so forth.
Y <- tf$placeholder(dtype = tf$float32, shape = shape(NULL, 10L))



M1 <- 4L # Number of hidden units
M2 <- 8L
M3 <- 12L
M4 <- 30

W1 <-  tf$Variable(initial_value =
                     tf$truncated_normal(shape(5L,5L,1L, M1),stddev = 0.1))
b1 <- tf$Variable(initial_value = tf$ones(shape(M1))/10)
W2 <- tf$Variable(initial_value =
                    tf$truncated_normal(shape(5L,5L,M1, M2),stddev = 0.1))
b2 <- tf$Variable(initial_value = tf$ones(shape(M2))/10)
W3 <-  tf$Variable(initial_value =
                     tf$truncated_normal(shape(4L,4L,M2, M3),stddev = 0.1))
b3 <- tf$Variable(initial_value = tf$ones(shape(M3))/10)

W4 <- tf$Variable(tf$truncated_normal(shape(588, M4),stddev = 0.1))
b4 <- tf$Variable(initial_value = tf$ones(shape(M4))/10)

W5 <- tf$Variable(tf$truncated_normal(shape(M4, 10L),stddev = 0.1))
b5 <- tf$Variable(initial_value = tf$ones(shape(10L))/10)

# The model
stride1 <- 1
H1 <- tf$nn$relu(tf$nn$conv2d(Xim, W1, strides =
                                shape(1L, stride1, stride1, 1L), padding='SAME') + b1)

stride2 <- 2
H2 <- tf$nn$relu(tf$nn$conv2d(H1, W2, strides =
                                shape(1L, stride2, stride2, 1L), padding='SAME') + b2)
H3 <- tf$nn$relu(tf$nn$conv2d(H2, W3, strides =
                                shape(1L, stride2, stride2, 1L), padding='SAME') + b3)
H3flat <- tf$reshape(H3, shape=shape(-1L,588L))
H4 <- tf$nn$relu(tf$matmul(H3flat, W4) + b4)
Z <- tf$matmul(H4, W5) + b5
Q <- tf$nn$softmax(Z)

# The loss function: Compute the cross-entropy
cross_entropy <-
  tf$nn$softmax_cross_entropy_with_logits(logits = Z, labels = Y)
cross_entropy <- tf$reduce_mean(cross_entropy)
#cross_entropy <- -tf$reduce_mean(tf$reduce_sum(Y * log(Q), reduction_indices=1L))

# Define the training
gammaMin <-  0.0001 # The learning rate
gammaMax <-  0.003
gamma <-  tf$placeholder(dtype = tf$float32)

train_step <- tf$train$AdamOptimizer(gamma)$minimize(cross_entropy)

prediction <- tf$argmax(Q, 1L) # In each row of Q, determine the index of the column with highest probability
true <- tf$argmax(Y, 1L) 
correct_prediction <- tf$equal(true, prediction) # Compare with true labels. TRUE if correct, FALSE if not.
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# Create session and initialize  variables
sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Initialize the test and training error resutls
test.accuracy <- c()
test.cross_entropy <- c()
test.iter <- c()
train.accuracy <- c()
train.cross_entropy <- c()
train.iter <- c()

# The main training loop
for (t in 1:10000) {
  gamma <- gammaMin+(gammaMax-gammaMin)*exp(-t/2000)
  # Get 100 training data points (called a mini-batch)
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  
  # Do one gradient setp
  sess$run(train_step,
           feed_dict = dict(Xim = batch_xs, Y = batch_ys))
  
  # Evaluate the performance on training data

  if (t%%10==0 || t==1) {
    train.tmp <- sess$run(c(accuracy,cross_entropy),
                          feed_dict = dict(Xim = batch_xs, Y = batch_ys))
    train.accuracy <- c(train.accuracy,train.tmp[[1]])
    train.cross_entropy<- c(train.cross_entropy,train.tmp[[2]])
    train.iter <- c(train.iter, t)
    cat('\n', t)
    cat(" train accuracy", sprintf("%.2f", train.tmp[[1]]))
    cat(" train cross-entropy", sprintf("%.4f", train.tmp[[2]]))  
  }
  
  # Evaluate the performance on training data at every 100th itertation

  if (t%%100==0 || t==1) {
    # Evaluate on test data
    test.tmp <- sess$run(c(accuracy,cross_entropy),
             feed_dict = dict(Xim = mnist$test$images, Y = mnist$test$labels))
    test.accuracy <- c(test.accuracy,test.tmp[[1]])
    test.cross_entropy <- c(test.cross_entropy,test.tmp[[2]])
    test.iter <- c(test.iter, t)
    
    # Also print iteration number and prediction accuracy
    cat('\n', "***************************************************")
    cat(" test accuracy", sprintf("%.4f", test.tmp[[1]]))
    cat(" test cross-entropy", sprintf("%.4f", test.tmp[[2]]))
  }
}

# The rest of the code will launch the plots! You don't have to go though them if you don't want to!

# Plot 100 test images
batch.test <- mnist$test$next_batch(100L)
batch.test.xs <- batch.test[[1]]
batch.test.ys <- batch.test[[2]]

testbatch.pred <- sess$run(prediction, feed_dict = dict(Xim = batch.test.xs))
testbatch.true <- sess$run(true, feed_dict = dict(Y = batch.test.ys))

# Sort the data such that we have the missclassifications first
result <- sort(testbatch.true==testbatch.pred,index.return=TRUE)

# Do the plot
.pardefault <- par(no.readonly = T) # Save default plotting parameters
par( mfrow = c(10,10), mai = c(0,0,0,0))
for(i in result$ix){
  im = t(as.matrix(batch.test.xs[i,,,1]))
  image( im[,nrow(im):1], axes = FALSE, col = gray(255:0 / 255)) #
  if (testbatch.pred[i]==testbatch.true[i]){
    text( 0.2, 0, testbatch.pred[i] , cex = 3, col = 1, pos = c(3,4))
  } else {
    text( 0.2, 0, testbatch.pred[i] , cex = 3, col = 2, pos = c(3,4))
  }
}
par(.pardefault)

# Plot the training/test cross-entropy
plot(x=train.iter,y=train.cross_entropy, col = "blue",type="l",ylim=c(0.0,tail(test.cross_entropy[!is.nan(test.cross_entropy)],n=1)*2),xlab="Iteration",ylab="Cross-entropy", main="Cross-entropy")
grid(NA,ny = NULL,lwd=2)
lines(x=test.iter,y=test.cross_entropy, col="red",lwd=3)
legend(x="topright",legend=c("Training data (mini-batch)","Test data"),col=c("blue","red"),lwd=c(1,3))

# Plot the training accuracy
plot(x=train.iter,y=train.accuracy, col = "blue",type="l",ylim=c(max(1-(1-tail(test.accuracy,n=1))*2,0),1.00),xlab="Iteration",ylab="Prediction accuracy", main="Prediction accuracy")
grid(NA,ny = NULL,lwd=2)
lines(x=test.iter,y=test.accuracy, col="red",lwd=3)
legend(x="bottomright",legend=c("Training data (mini-batch)","Test data"),col=c("blue","red"),lwd=c(1,3))

# Close the session
sess$close()

