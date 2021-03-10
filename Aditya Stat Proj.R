## SVHN Dataset Classification using Artificial Neural Networks::

# 1 - Installing required packages::
devtools::install_github("rstudio/keras")  #skip if already installed
library(keras)
install_keras()
library(tensorflow)
install_tensorflow(version = "nightly") #skip if already installed
library(keras)

require(R.matlab)
require(reticulate)

# 2 - Loading the dataset:: (remember to change below path to your directory)
data_train <- readMat("/Users/pramodhreddy/Downloads/svhn datasets/train_32x32.mat")
data_test <- readMat("/Users/pramodhreddy/Downloads/svhn datasets/test_32x32.mat")

x_train <- data_train$X
y_train <- data_train$y

x_test <- data_test$X
y_test <- data_test$y

# 3 - Preprocessing data suitable for CNN::
d <- dim(x_train)
train <- array(dim=c(d[4], d[1], d[2], d[3]))
for (i in 1:d[4])
{ train[i,,,] <- x_train[,,,i]
}

d = dim(x_test)
test = array(dim=c(d[4], d[1], d[2], d[3]))
for (i in 1:d[4])
{ test[i,,,] <- x_test[,,,i]
}

x_trainCNN <- array_reshape(train, c(dim(x_train)[4], 32, 32, 3))
x_testCNN <- array_reshape(test, c(dim(x_test)[4], 32, 32, 3))


# 3.3 - Rescaling Data::
x_trainCNN <- x_trainCNN / 255
x_testCNN <- x_testCNN / 255


# 3.3 - Organising the labels suitable for Classification::
y_train <- to_categorical(y_train-1, 10)
y_test <- to_categorical(y_test-1, 10)


## 4 - CNN architecture using Adaptive Gradient Descent::

model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(5,5), activation = 'relu',
                input_shape = c(32, 32, 3), padding="same") %>% 
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(5,5), activation = 'relu', padding="same") %>% 
  layer_average_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(5,5), activation = 'relu', padding="same") %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 10, activation = 'softmax')

summary(model)  

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

history <- model %>% 
  fit(
    x_trainCNN, y_train,
    epochs = 20,verbose = 1,
    batch_size = 256, validation_split = 0.2
  )

# evaluate accuracy on test set::
model %>% evaluate(x_testCNN, y_test, batch_size = 256, verbose = 1, sample_weight = NULL)

# prediction
model %>% predict_classes(x_testCNN)

## 5 - CNN Using SGD Optimizer::

modelSGD <- keras_model_sequential()
modelSGD %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(32, 32, 3), padding="same") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu', padding="same") %>% 
  layer_average_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu', padding="same") %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 10, activation = 'softmax')


modelSGD %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_sgd(),
  metrics = c('accuracy')
)

historySGD <- modelSGD %>% 
  fit(
    x_trainCNN, y_train,
    epochs = 20,verbose = 1,
    batch_size = 64, validation_split = 0.2
  )

# evaluate accuracy on test set::
modelSGD %>% evaluate(x_testCNN, y_test)

# prediction
modelSGD %>% predict_classes(x_testCNN)


##6. Comparing the 2 models::

library(tidyr)
library(tibble)
library(dplyr)
library(ggplot2)

compare_cx <- data.frame(
  Adadelta_train = history$metrics$loss,
  Adadelta_val = history$metrics$val_loss,
  
  SGD_train = historySGD$metrics$loss,
  SGD_val = historySGD$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")
