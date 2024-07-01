#===============================================================================
#
#  FILE: FUNCS.R
#
#  AUTHOR: Stephen Salerno (ssalerno@fredhutch.org)
#
#  PURPOSE: R functions to implement method and simulate data:
#
#    - sim_dat: Generate simulated data based on Clayton copula
#    - sim_gumbel: Generate simulated data based on Gumbel copula
#    - KM_time: Calculate survival probabilities based on Kaplan-Meier method
#    - clayton_psi: Clayton copula generator function
#    - clayton_ipsi: Inverse of Clayton copula generator function
#    - surv_fine: Marginal non-fatal survival function
#    - pseudo_scr: Pseudo-values
#
#  NOTES:
#
#    1. Copula estimation code graciously adapted from Orenti et al. (2022)
#
#         "A pseudo-values regression model for non-fatal event free survival
#          in the presence of semi-competing risks"
#
# pseudo_scr
#
#  UPDATED: 2024.06.29
#
#===============================================================================

#--- DATA GENERATION -----------------------------------------------------------
#
#  FUNCTION:
#    sim_dat
#
#  PURPOSE:
#    Simulates bivariate survival data from the Clayton copula as described in
#    the manuscript for a given sample size, copula parameter, censoring rate,
#    and risk function of covariates.
#
#  ARGUMENTS:
#    n (int): Sample size
#    theta (float): Copula dependence parameter (must be > 0)
#    cens (float): Approximate censoring rate (currently either 0 or 0.5)
#    risk (string): Risk function of generated covariates. Can take values:
#
#     - "PH-L": Proportional Hazards Model with Linear Risk Function
#     - "PH-NL": Proportional Hazards Model with Non-Linear Risk Function
#
#  RETURNS:
#    list: A list containing the following elements:
#
#      - dat (data.frame): Simulated data with n observations of 10 variables:
#
#         * Yi1: Observed first event time; min(T1, T2, C)
#         * Di1: Event indicator for non-terminal event; I(T1 < min(T2, C))
#         * Yi2: Observed terminal event time; min(T2, C)
#         * Di2: Event indicator for terminal event; I(T2 < C)
#         * YiS: Observed progression-free survival time
#         * DiS: Event indicator for progression-free survival
#         * Ci:  (Right) censoring times
#         * X1:  Binary covariate (i.e., treatment, 'Z')
#         * X2:  Normal covariate
#         * X3:  Normal covariate
#
#      - ATE (float): Empirical average treatment effect
#
#-------------------------------------------------------------------------------

sim_dat <- function(n, theta, cens, risk) {

  #-- Errors

  e1 <- log(-log(runif(n)))

  ec <- ((runif(n)^(-theta / (1 + theta)) - 1) *

           exp(theta * exp(e1)) + 1)^(-1 / theta)

  e2 <- log(-log(ec))

  #-- Marginal Models

  if (risk == "PH-L") {

    #-- Proportional Hazards, Linear Risk

    #- Covariates

    X1 <- rbinom(n, 1, 0.5)

    X2 <- rtruncnorm(n, a = 0, b = 2, mean = 1, sd = sqrt(0.5))

    X3 <- rtruncnorm(n, a = 0, b = 2, mean = 1, sd = sqrt(0.5))

    #- Event Times

    Ti1 <- 3*exp(-(1.0 * X1 + 1.0 * X2 + 1.0 * X3) + e1)

    Ti2 <- 3*exp(-(0.2 * X1 + 0.2 * X2 + 0.2 * X3) + e2)

    #- ATE

    ATE <- mean(exp(-3*exp(-(X2 + X3))) - exp(-3*exp(-(1 + X2 + X3))))

  } else if (risk == "PH-NL") {

    #-- Proportional Hazards, Non-Linear Risk

    #- Covariates

    Sigma <- toeplitz(0.5^(0:2))

    X <- MASS::mvrnorm(n, rep(0, 3), Sigma)

    X1 <- ifelse(X[,1] <= 0, 0, 1)

    X2 <- X[,2]

    X3 <- X[,3]

    #- Event Times

    Ti1 <- 3*exp(-(1.0 * X1 + 1.0 * X2^2 + 1.0 * X3^2) + e1)

    Ti2 <- 3*exp(-(0.2 * X1 + 0.2 * X2^2 + 0.2 * X3^2) + e2)

    # ATE

    ATE <- mean(exp(-3*exp(-(X2^2 + X3^2))) - exp(-3*exp(-(1 + X2^2 + X3^2))))
  }

  #-- Censoring Time

  if (cens == 0) {

    Ci <- runif(n, 100, 101)

  } else if (cens == 0.5) {

    Ci <- ifelse(rbinom(n, 1, 0.2) == 1, runif(1, 0, 1), runif(1, 1, 1.2))
  }

  #-- Observed Times

  Yi2 <- pmin(Ti2, Ci)

  Yi1 <- pmin(Ti1, Ti2, Ci)

  YiS <- pmin(Ti1, Ti2, Ci)

  #-- Event Indicators

  Di2 <- ifelse(Ti2 <= Ci, 1, 0)

  Di1 <- ifelse(Ti1 <= pmin(Ti2, Ci), 1, 0)

  DiS <- ifelse(pmin(Ti1, Ti2) < Ci, 1, 0)

  #-- Data

  dat <- data.frame(Yi1, Di1, Yi2, Di2, YiS, DiS, Ci, X1, X2, X3)

  return(list(dat, ATE))
}

#--- DATA GENERATION FOR GUMBEL COPULA -----------------------------------------
#
#  FUNCTION:
#    sim_gumbel
#
#  PURPOSE:
#    Simulates bivariate survival data from the Gumbel copula as described in
#    the manuscript for a given sample size, copula parameter, censoring rate,
#    and risk function of covariates.
#
#  ARGUMENTS:
#    n (int): Sample size
#    theta (float): Copula dependence parameter (must be > 1)
#    cens (float): Approximate censoring rate (currently either 0 or 0.5)
#
#  RETURNS:
#    list: A list containing the following elements:
#
#      - dat (data.frame): Simulated data with n observations of 10 variables:
#
#         * Yi1: Observed first event time; min(T1, T2, C)
#         * Di1: Event indicator for non-terminal event; I(T1 < min(T2, C))
#         * Yi2: Observed terminal event time; min(T2, C)
#         * Di2: Event indicator for terminal event; I(T2 < C)
#         * YiS: Observed progression-free survival time
#         * DiS: Event indicator for progression-free survival
#         * Ci:  (Right) censoring times
#         * Z:   Binary covariate (i.e., treatment)
#         * X1:  Normal covariate
#         * X2:  Normal covariate
#
#      - ATE (float): Empirical average treatment effect
#
#-------------------------------------------------------------------------------

sim_gumbel <- function(n, theta, cens) {

  Z <- rbinom(n, 1, 0.5)

  X1 <- rtruncnorm(n, a = 0, b = 2, mean = 1, sd = sqrt(0.5))

  X2 <- rtruncnorm(n, a = 0, b = 2, mean = 1, sd = sqrt(0.5))

  u1 <- runif(n)

  e1 <- log(-log(u1))

  ec <- exp(-((- log(runif(n)))^(theta) + (- log(u1))^(theta))^(1/theta))

  e2 <- log(-log(ec))

  Ti1 <- 3*exp(-(1*Z + 1*X1 + 1*X2) + e1)

  Ti2 <- 3*exp(-(0.2*Z + 0.2*X1 + 0.2*X2) + e2)

  if (cens == 0) {

    Ci <- Inf

  } else if (cens == 0.5) {

    Ci <- ifelse(rbinom(n, 1, 0.2) == 1, runif(1, 0, 1), runif(1, 1, 1.2))
  }

  Yi2 <- pmin(Ti2, Ci)

  Yi1 <- pmin(Ti1, Ti2, Ci)

  YiS <- pmin(Ti1, Ti2, Ci)

  Di2 <- ifelse(Ti2 <= Ci, 1, 0)

  Di1 <- ifelse(Ti1 <= pmin(Ti2, Ci), 1, 0)

  DiS <- ifelse(pmin(Ti1, Ti2) < Ci, 1, 0)

  dat <- data.frame(Yi1, Di1, Yi2, Di2, YiS, DiS, Ci, Z, X1, X2)

  ATE <- mean(exp(-3*exp(-(1 + 1*X1 + 1*X2))) - exp(-3*exp(-(1*X1 + 1*X2))))

  return(list(dat, ATE))
}

#--- SURVIVAL TIMES ------------------------------------------------------------
#
#  FUNCTION:
#    KM_time
#
#  PURPOSE:
#     Computes survival probabilities at specified time points using
#     Kaplan-Meier survival estimates
#
#  ARGUMENTS:
#    x (vector): Numeric vector of time points to compute survival probabilities
#    KMx (survfit): A Kaplan-Meier object containing:
#
#      - KMx$time (vector): Numeric vector of time points where survival
#                           probabilities are estimated
#      - KMx$surv (vector/matrix): A matrix or vector of survival probabilities
#                                  corresponding to the time points in KMx$time
#
#  RETURNS:
#    matrix: Matrix of survival probabilities, where each row corresponds to a
#            time point in x and each column corresponds to a survival
#            probability from the Kaplan-Meier object. The dimensions of the
#            output matrix depend on the input x and the KMx$surv component.
#
#-------------------------------------------------------------------------------

KM_time <- function(x, KMx){

  row_n <- ifelse(is.null(dim(KMx$surv)), 1, dim(KMx$surv)[2])

  res_x <- matrix(NA, row_n, length(x))

  for(i in 1:length(x)) {

    if(x[i] == 0 | (which.min(KMx$time <= x[i]) == 1 & max(KMx$time) > x[i])) {

      res_x[,i] <- rep(1, row_n)

    } else {

      if(max(KMx$time) <= x[i]) {

        if(row_n == 1) {

          res_x[,i] <- min(KMx$surv)

        } else {

          res_x[,i] <- apply(KMx$surv,2,min)
        }

      } else {

        if(row_n == 1) {

          res_x[,i] <- KMx$surv[which.min(KMx$time <= x[i]) - 1]

        } else {

          res_x[,i] <- KMx[,]$surv[which.min(KMx$time <= x[i]) - 1,]
        }
      }
    }
  }

  t(res_x)
}

#--- LEAVE-ONE-IN COPULA PARAMETER ESTIMATE ------------------------------------
#
#  FUNCTION:
#    theta_loi
#
#  PURPOSE:
#     Computes the 'leave-one-in' estimate of the Clayton copula parameter
#
#  ARGUMENTS:
#    dat (data.frame): Data frame with semi-competing outcomes and covariates
#    x_ind (vector): Numeric vector of indices for covariate columns
#    tol (float): Tolerance to select number of neighbors (defaults to 0.001)
#
#  RETURNS:
#    float: Leave-one-in estimate of the copula dependence parameter
#
#-------------------------------------------------------------------------------

theta_loi <- function (dat, x_ind, tol = 0.001) {

  n <- nrow(dat)

  distmat <- as.matrix(dist(dat[, x_ind]))

  k <- 2

  theta_k_prev <- theta_k <- Inf

  while (k <= n - 1) {

    nearmat <- apply(distmat, 2, function(i) {

      rank(i)[1:k] })

    thetas_loi_mat <- apply(nearmat, 2, function(i) {

      with(dat[i,], theta_fine_cpp(Yi2, YiS, Di2, DiS))})

    thetas_fine_loi <- na.omit(unlist(thetas_loi_mat))

    theta_k <- mean(thetas_fine_loi[is.finite(thetas_fine_loi)], na.rm = T)

    if (abs(theta_k - theta_k_prev) <= tol) {

      break

    } else {

      theta_k_prev <- theta_k

      k <- k + 1
    }
  }

  return(theta_k)
}

#--- COPULA GENERATOR FUNCTIONS ------------------------------------------------
#
#  FUNCTIONS:
#    clayton_psi
#    clayton_ipsi
#
#  PURPOSE:
#     Computes generator (clayton_psi) or inverse generator (clayton_ipsi)
#     function for the Clayton copula
#
#  ARGUMENTS:
#    time (vector): Time points to compute the generator function
#    theta (float): Copula dependence parameter
#
#  RETURNS:
#    vector: Generator or inverse generator values
#
#-------------------------------------------------------------------------------

clayton_psi  <- function(time, theta) {

  (1 / theta) * (time^(-theta) - 1)
}

clayton_ipsi <- function(time, theta) {

  ((1 + theta * time)^(-1 / theta))
}

#--- MARGINAL NON-FATAL SURVIVAL FUNCTION --------------------------------------
#
#  FUNCTION:
#    surv_fine
#
#  PURPOSE:
#     Computes the survival function for the non-fatal event under an
#     assumed Clayton copula
#
#  ARGUMENTS:
#    theta (float): Copula dependence parameter
#    time (vector): Time points to compute the survival probabilities
#    Fz (survfit): A Kaplan-Meier object for progression-free survival
#    Fy (survfit): A Kaplan-Meier object for death
#
#  RETURNS:
#    (vector): Marginal non-fatal survival function at input time points
#
#-------------------------------------------------------------------------------

surv_fine <- function(theta, time, Fz, Fy) {

  arg <- clayton_psi(KM_time(time, Fz), theta) -

    clayton_psi(KM_time(time, Fy), theta)

  for(i in 2:length(time)) {

    arg[i] <- ifelse(arg[i] < 0, arg[i - 1],

      ifelse(arg[i] < arg[i - 1], arg[i - 1], arg[i]))
  }

  clayton_ipsi(arg, theta)
}

#--- PSEUDO-VALUES -------------------------------------------------------------
#
#  FUNCTION:
#    pseudo_scr
#
#  PURPOSE:
#     Computes pseudo-values for the non-fatal survival probability at given
#     time points based on the leave-one-out method under the Clayton copula
#
#  ARGUMENTS:
#    dat (data.frame): Data frame containing semi-competing outcome variables:
#
#      - YiS: Observed progression-free survival time
#      - DiS: Event indicator for progression-free survival
#      - Yi2: Observed terminal event time; min(T2, C)
#      - Di2: Event indicator for terminal event; I(T2 < C)
#
#    time (vector): Time points to compute the pseudo-values
#    theta_fine (float): Copula dependence parameter
#
#  RETURNS:
#    (list): A list containing two elements:
#
#      - pseudo (data.frame): A data frame including the original data, the
#                             computed pseudo-values (PSEUDO), and additional
#                             columns for the time points (TIME), and subject
#                             identifier (original row number; ID)
#      - theta (float): The adjusted copula parameter used in our formulation,
#                       i.e., theta_fine - 1
#
#-------------------------------------------------------------------------------

pseudo_scr <- function(dat, time, theta) {

  n <- nrow(dat)

  Ssn <- survfit(Surv(YiS, DiS) ~ 1, data = dat)

  S2n <- survfit(Surv(Yi2, Di2) ~ 1, data = dat)

  S1n <- surv_fine(theta, time, Ssn, S2n)

  pseudo <- map_dfr(1:n, function(i) {

    dat_i <- dat[-i,]

    Ssi <- survfit(Surv(YiS, DiS) ~ 1, data = dat_i)

    S2i <- survfit(Surv(Yi2, Di2) ~ 1, data = dat_i)

    S1i <- surv_fine(theta, time, Ssi, S2i)

    pseudo_i <- n*S1n - (n - 1)*S1i

    df_i <- data.frame(cbind(time, i, pseudo_i, dat[i,]))

    colnames(df_i) <- c("TIME", "ID", "PSEUDO", colnames(dat))

    df_i
  })

  return(list(pseudo, theta))
}

#=== END =======================================================================
