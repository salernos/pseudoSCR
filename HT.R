#===============================================================================
#
#  FILE: HT.R
#
#  AUTHOR: Stephen Salerno (ssalerno@fredhutch.org)
#
#  PURPOSE: Script for hypertuning deep neural network
#
#  NOTES: This script is called in "EXAMPLE_HT.R" and is needed for hypertuning
#         the downstream deep neural network (called by "tuning_run()")
#
#  UPDATED: 2024.06.29
#
#===============================================================================

#--- SOURCE NECESSARY PACKAGES -------------------------------------------------

library(keras)

#--- DEFAULT VALUES FOR HYPERPARAMETERS ----------------------------------------

FLAGS <- flags(

  flag_numeric('dropout1', 0.3),

  flag_integer('neurons1', 4),

  flag_integer('neurons2', 4),

  flag_numeric('l2', 0.001),

  flag_numeric('lr', 0.001)
)

#--- ARCHITECTURE --------------------------------------------------------------

build_model <- function() {

  model <- keras_model_sequential()

  model %>%

    layer_dense(units = FLAGS$neurons1, activation = 'relu',

      kernel_regularizer = regularizer_l2(l = FLAGS$l2),

      input_shape = dim(x_train)[2]) %>%

    layer_dropout(FLAGS$dropout1) %>%

    layer_dense(units = FLAGS$neurons2, activation = 'relu') %>%

    layer_dense(units = 1, activation = "sigmoid")

  model %>% compile(

    loss = "binary_crossentropy",

    optimizer = optimizer_rmsprop(learning_rate = FLAGS$lr),

    metrics = list("binary_crossentropy"))

  model
}

#--- EVALUATE ------------------------------------------------------------------

model <- build_model()

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25)

epochs <- 100

history <- model %>% fit(

  x_train,

  y_train,

  epochs = epochs,

  validation_split = 0.2,

  verbose = 0,

  callbacks = list(early_stop)
)

plot(history)

score <- model %>% evaluate(x_test, y_test, verbose = 0)

#--- STORE RESULTS -------------------------------------------------------------

save_model_hdf5(model, 'model.h5')

#=== END =======================================================================
