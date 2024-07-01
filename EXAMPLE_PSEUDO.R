#===============================================================================
#
#  FILE: EXAMPLE_PSEUDO.R
#
#  AUTHOR: Stephen Salerno (ssalerno@fredhutch.org)
#
#  PURPOSE: Example pseudo-value calculation for illustration
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

#=== EXAMPLE PSEUDO-VALUES =====================================================

set.seed(12345)

simu <- sim_dat(n, theta_true, cens, risk = "PH-L")

simdat <- simu[[1]]

pseudocalc <- pseudo_scr(simdat, time, theta_true)

pseudodat <- pseudocalc[[1]]

bind_rows(

  pseudodat |>

    filter(ID == which(simdat$Di1 == 1 & simdat$Di2 == 1)[2]) |>

    dplyr::select(ID, TIME, Yi1, Di1, Yi2, Di2, X1, X2, PSEUDO),

  pseudodat |>

    filter(ID == which(simdat$Di1 == 0 & simdat$Di2 == 0)[2]) |>

    dplyr::select(ID, TIME, Yi1, Di1, Yi2, Di2, X1, X2, PSEUDO)) |>

  gt::gt()

#=== END =======================================================================
