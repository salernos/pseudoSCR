# pseudoSCR

Code for "A Pseudo-Value Approach to Causal Deep Learning of Semi-Competing Risks."

## Overview

This project proposes a deep learning approach for estimating the causal effect of treatment on non-fatal outcomes in the presence of dependent censoring and complex covariate relationships. Our three-stage approach involves:

1. Estimating the non-fatal event's marginal survival function using an Archimedean copula representation
2. Constructing jackknife pseudo-values that estimate pseudo-survival probabilities for the non-fatal event at fixed time points
3. Fitting a deep neural network (S-learner) to estimate survival average causal effect of treatment

This repository contains the code necessary to implement this approach, as well as two `R` scripts which provide example calculations on simulated data. In this directory are the following files:

- `FUNCS.R`: Contains the `R` functions necessary for steps (1) and (2) of our approach, namely the estimation of the non-fatal event's marginal survival function and construction of the pseudo-outcomes
- `THETA_FINE.cpp`: Contains a `C++` function that implements the concordance-based copula parameter estimator of Fine et al. (2001), 
*"On semi-competing risks data"*
- `HT.R`: An example script to hypertune the deep neural network in step (3). This is called in the file `EXAMPLE_HT.R`
- `EXAMPLE_PSEUDO.R`: An example script to show the calculation of the pseudo-values (steps 1 and 2) on simulated data
- `EXAMPLE_HT.R`: An example script to show the implementation of the entire procedure, including hypertuning of the step (3) model, on simulated data

**Note**: For homogeneous data, the `C++` function `THETA_FINE.cpp` provides the point estimate and optional analytic standard errors. Our function for the "leave-one-in" approach for heterogenous data only utilizes the point estimates.

**Note**: The copula estimation code was graciously adapted from Orenti et al. (2022), *"A pseudo-values regression model for non-fatal event free survival in the presence of semi-competing risks."*

## Example Workflow

We work through an example analysis on simulated data. Note that the code below is also provided in `EXAMPLE_HT.R`. First, we load the necessary packages and functions from this repository:

```
#--- LOAD NECESSARY PACKAGES ---------------------------------------------------

library(pacman)

p_load(MASS, Matrix, Rcpp, truncnorm, survival, survivalmodels, keras, tfruns,

  reticulate, pec, tidyverse, update = F)

#--- LOAD HELPER FUNCTIONS -----------------------------------------------------

source("FUNCS.R")

sourceCpp("THETA_FINE.cpp")
```

We provide two functions for generating data in `FUNCS.R` based on the simulation settings described in the main paper. These are `sim_dat()`, which generates data under the assumed Clayton copula, and `sim_gumbel()`, which generates data under a misspecified Gumbel copula. We can generate data for hypertuning as follows:

```
#--- GLOBAL PARAMETERS ---------------------------------------------------------

n <- 500

theta_true <- 0.5

cens <- 0.5

#--- GENERATE DATA FOR HYPERTUNING ---------------------------------------------

set.seed(1)

simu_ht <- sim_dat(n, theta_true, cens, risk = "PH-L")

simdat_ht <- simu_ht[[1]]

```

Here, we generate a dataset of 500 observations (`n`), with a true copula dependence parameter, $\theta = 0.5$, and an approximate censoring rate of 50%. We then estimate this copula parameter with our proposed "leave-one-in" approach (see the main text for more information) using the function `theta_loi()`, where the `x_ind` argument specifies which columns of `simdat_ht` are covariates:

```
theta_fine_ht <- theta_loi(simdat_ht, x_ind = 8:10)
```

We then estimate our pseudo-values at time points $t$ = 0.2, 0.4, 0.6 and 0.8 using the function `pseudo_scr()` and our estimated theta value, `theta_fine_ht`:

```
time <- seq(0.2, 1, by = 0.2)

pseudocalc_ht <- pseudo_scr(simdat_ht, time, theta_fine_ht)

pseudodat_ht <- pseudocalc_ht[[1]]
```

Hyperparameters needed to fully specify the neural network architecture include the number of hidden layers and number of nodes per hidden layer, the dropout fraction, and learning rate. In practice, these quantities are optimized over a Cartesian grid search based on predictive performance. We provide a script, `HT.R` and the following code to assist with hypertuning. We do this on one set of simulated data based on an 80/20 training/testing split before later applying the final architecture to our simulated analytic data:

```
#--- HYPERTUNING ---------------------------------------------------------------

cat("Starting Hypertuning\n")

train_prop <- 0.8

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

```

Note that we chose to randomly evaluate a 25% subset of all of the different hyperparameter combinations stored in `par` with the `tuning_runs()` argument `sample = 0.25`. This can be omitted, or the total number of possible combinations can be reduced or increased based on your specific context. Also, note that each of the tuning runs are stored in a sub-directory, `/_tuning_ex`, and can be inspected further. 

We next exemplify the procedure on a separate set of generated data, where we repeat the above procedure using the optimal hyperparameter values stored in `best_run`:

```
#--- EXAMPLE ANALYSIS ----------------------------------------------------------

cat("Starting Example Analysis\n")

early_stopping <- callback_early_stopping(patience = 25)

#-- SIMULATE DATA

set.seed(2)

simu <- sim_dat(n, theta_true, cens, "PH-L")

simdat <- simu[[1]]

#-- ESTIMATE THETA

theta_fine <- theta_loi(simdat, 8:10)

#-- CALCULATE PSEUDO VALUES

pseudocalc <- pseudo_scr(simdat, time, theta_fine)

pseudodat <- pseudocalc[[1]]

#-- SPLIT DATA INTO TRAINING AND TESTING SETS

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

#-- FIT DNN

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
  
#-- ESTIMATE ATE

x1_test_t1 <- x_test_t1; x1_test_t1[,6] <- 1

y1_pred_fit <- fit_dnn |> predict(x1_test_t1)

x0_test_t1 <- x_test_t1; x0_test_t1[,6] <- 0

y0_pred_fit <- fit_dnn |> predict(x0_test_t1)

ate_dnn <- mean(y1_pred_fit - y0_pred_fit)

cat("True ATE:", simu[[2]], "\nEstimated ATE:", ate_dnn, "\n")

cat("End of example.\n")
```

## Contact

For questions or suggestions, please do not hesitate to contact us at: ssalerno@fredhutch.org
