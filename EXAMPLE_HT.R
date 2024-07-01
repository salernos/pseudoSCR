#===============================================================================
#
<<<<<<< HEAD
#  FILE: EXAMPLE_HT.R
=======
#  FILE: EXAMPLE_PSEUDO.R
>>>>>>> d03f4959a5d4b9a0215611f79bb5921e05f081f6
#
#  AUTHOR: Stephen Salerno (ssalerno@fredhutch.org)
#
#  PURPOSE: Example run on simulated dataset with hypertuning on a separate,
#           simulated dataset
#
#  UPDATED: 2024.06.29
#
#===============================================================================

#=== INITIALIZATION ============================================================

#--- SOURCE NECESSARY PACKAGES -------------------------------------------------

library(pacman)

p_load(MASS, Matrix, Rcpp, truncnorm, survival, survivalmodels, keras, tfruns,

  reticulate, pec, tidyverse, update = F)

#--- HELPER FUNCTIONS ----------------------------------------------------------

source("FUNCS.R")

sourceCpp("THETA_FINE.cpp")

#--- GLOBAL PARAMETERS ---------------------------------------------------------

n <- 500

theta_true <- 0.5

cens <- 0.5

time <- seq(0.2, 1, by = 0.2)

train_prop <- 0.8

#=== EXAMPLE RUN ON SIMULATED DATA =============================================

#--- HYPERTUNING ---------------------------------------------------------------

cat("Starting Hypertuning\n")

set.seed(1)

simu_ht <- sim_dat(n, theta_true, cens, risk = "PH-L")

simdat_ht <- simu_ht[[1]]

theta_fine_ht <- theta_loi(simdat_ht, 8:10)

pseudocalc_ht <- pseudo_scr(simdat_ht, time, theta_fine_ht)

pseudodat_ht <- pseudocalc_ht[[1]]

train_id <- sample(1:n, floor(n*train_prop))

test_id  <- setdiff(1:n, train_id)

pseudodat_train_ht <- pseudodat_ht |> filter(ID %in% train_id)

pseudodat_test_ht  <- pseudodat_ht |> filter(ID %in% test_id)

x_train <- model.matrix(~ factor(TIME) + X1 + X2 + X3 - 1,

  data = pseudodat_train_ht)

x_test  <- model.matrix(~ factor(TIME) + X1 + X2 + X3 - 1,

  data = pseudodat_test_ht)

x_test_t1 <- x_test[which(x_test[,5] == 1),]

y_train <- matrix(pseudodat_train_ht$PSEUDO, ncol = 1)

y_test  <- matrix(pseudodat_test_ht$PSEUDO,  ncol = 1)

y_test_t1 <- y_test[which(x_test[,5] == 1),]

par <- list(

  dropout1 = seq(0.1, 0.4, 0.1),

  neurons1 = 2^(2:6),

  neurons2 = 2^(2:5),

  l2 = 10^(-(3:1)),

  lr = 10^(-(4:1))
)

runs <- tuning_run('HT.R',

  runs_dir = paste("_tuning_ex", n, theta_true, cens, "PH-L", sep = "_"),

  sample = 0.25, flags = par)

ls_runs(order = "metric_binary_crossentropy", decreasing= F,

  runs_dir = paste("_tuning_ex", n, theta_true, cens, "PH-L", sep = "_"))

best_run <- ls_runs(order = "metric_binary_crossentropy", decreasing= F,

  runs_dir = paste("_tuning_ex", n, theta_true, cens, "PH-L", sep = "_"))[1,]

cat("Finished Hypertuning\n")

#--- EXAMPLE ANALYSIS ----------------------------------------------------------

cat("Starting Example Analysis\n")

early_stopping <- callback_early_stopping(patience = 25)

set.seed(2)

simu <- sim_dat(n, theta_true, cens, "PH-L")

simdat <- simu[[1]]

theta_fine <- theta_loi(simdat, 8:10)

pseudocalc <- pseudo_scr(simdat, time, theta_fine)

pseudodat <- pseudocalc[[1]]

train_id <- sample(1:n, floor(n*train_prop))

test_id  <- setdiff(1:n, train_id)

simdat_train <- simdat |> mutate(ID = 1:n()) |> filter(ID %in% train_id)

simdat_test  <- simdat |> mutate(ID = 1:n()) |> filter(ID %in% test_id)

pseudodat_train <- pseudodat |> filter(ID %in% train_id)

pseudodat_test  <- pseudodat |> filter(ID %in% test_id)

x_train <- model.matrix(~ factor(TIME) + X1 + X2 + X3 - 1,

  data = pseudodat_train)

x_test  <- model.matrix(~ factor(TIME) + X1 + X2 + X3 - 1,

  data = pseudodat_test)

x_test_t1 <- x_test[which(x_test[,5] == 1),]

y_train <- matrix(pseudodat_train$PSEUDO, ncol = 1)

y_test  <- matrix(pseudodat_test$PSEUDO,  ncol = 1)

y_test_t1 <- y_test[which(x_test[,5] == 1),]

fit_dnn <- keras_model_sequential()

fit_dnn |>

  layer_dense(units = best_run$flag_neurons1, activation = 'relu',

    kernel_regularizer = regularizer_l2(best_run$flag_l2),

    input_shape = dim(x_train)[2]) |>

  layer_dropout(best_run$flag_dropout1) |>

  layer_dense(units = best_run$flag_neurons2, activation = 'relu') |>

  layer_dense(units = 1, activation = "sigmoid")

fit_dnn |> compile(

  loss = "binary_crossentropy",

  optimizer = optimizer_adam(learning_rate = best_run$flag_lr))

fit_dnn |> fit(

  x_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2,

  callbacks = list(early_stopping), verbose = 1)

x1_test_t1 <- x_test_t1; x1_test_t1[,6] <- 1

y1_pred_fit <- fit_dnn |> predict(x1_test_t1)

x0_test_t1 <- x_test_t1; x0_test_t1[,6] <- 0

y0_pred_fit <- fit_dnn |> predict(x0_test_t1)

ate_dnn <- mean(y1_pred_fit - y0_pred_fit)

cat("True ATE:", simu[[2]], "\nEstimated ATE:", ate_dnn, "\n")

cat("End of example.\n")

#=== END =======================================================================
