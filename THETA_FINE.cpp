#include <Rcpp.h>
using namespace Rcpp;

//--- CONCORDANCE-BASED COPULA PARAMETER ESTIMATE ------------------------------
//
//  FUNCTION:
//    theta_fine_cpp
//
//  PURPOSE:
//    Computes the concordance-based copula parameter estimate and variance
//    for the Clayton copula based on the method of Fine et al. (2001)
//
//  ARGUMENTS:
//    time_d (vector): A numeric vector of the fatal event times
//    time_r (vector): A numeric vector of the non-fatal event times
//    status_d (vector): An integer vector for the status of the fatal event
//                       (1 for event, 0 for censored)
//    status_rd (vector): An integer vector for the status of either event
//                        (1 for event, 0 for censored)
//    weights (bool): A boolean flag for weighted version of estimate
//                    (defaults to true)
//    variance (bool): A boolean flag for whether to compute the variance of
//                     the theta parameter (defaults to false)
//
//  RETURNS:
//    (list): A list containing two elements:
//
//      - theta (float): Estimated theta parameter
//      - var_theta (float): Variance of theta (if variance = true, else NA)
//
//  NOTES: Code graciously adapted from Orenti et al. (2022)
//
//         "A pseudo-values regression model for non-fatal event free survival
//          in the presence of semi-competing risks"
//
//------------------------------------------------------------------------------

// [[Rcpp::export]]
List theta_fine_cpp(NumericVector time_d, NumericVector time_r,

  IntegerVector status_d, IntegerVector status_rd,

  bool weights = true, bool variance = false) {

  int n = time_d.length();

  double numerator = 0.0, denominator = 0.0;

  NumericVector y(n), x(n), c(n);

  NumericMatrix DD(n, n), delta(n, n), WW(n, n);

  for (int i = 0; i < n; i++) {

    y[i] = status_d[i]  == 1 ? time_d[i] : time_d[i] + 1e-10;
    x[i] = status_rd[i] == 1 ? time_r[i] : time_d[i] + 1e-10;
    c[i] = status_d[i]  == 0 ? time_d[i] : time_d[i] + 1e-10;
  }

  for (int i = 0; i < n - 1; i++) {

    for (int j = i + 1; j < n; j++) {

      double x_t = fmin(x[i], x[j]); // x tilde variable
      double y_t = fmin(y[i], y[j]); // y tilde variable
      double c_t = fmin(c[i], c[j]); // c tilde variable

      DD(i, j) = (x_t < y_t && y_t < c_t) ? 1 : 0; // Dij

      if (DD(i, j) == 1) {

        delta(i, j) = ((x[i] - x[j]) * (y[i] - y[j]) > 0) ? 1 : 0; // delta_ij
      }

      double s_t = fmin(fmin(x_t, y_t), c_t); // s tilde variable
      double r_t = fmin(y_t, c_t);            // r tilde variable

      double ind = 0.0;

      for(int k = 0; k < n; k++) {

        if((time_r[k] >= s_t) && (time_d[k] >= r_t)) {

          ind++;
        }
      }

      WW(i, j) = weights ? n / ind : 1;

      numerator   += WW(i, j) * DD(i, j) * delta(i, j);

      denominator += WW(i, j) * DD(i, j) * (1 - delta(i, j));
    }
  }

  double theta = numerator / denominator;

  double var_theta = NA_REAL;

  if (variance) {

    NumericMatrix QQ(n, n);

    for (int i = 0; i < n - 1; i++) {

      for (int j = i + 1; j < n; j++) {

        QQ(i, j) = WW(i, j) * DD(i, j) * (delta(i, j) - theta / (1 - theta));
      }
    }

    double II = 0.0;

    for (int i = 0; i < n; i++) {

      for (int j = i+1; j < n; j++) {

        double w = WW(i,j);
        double d = DD(i,j);

        II += w * d / pow(1 - theta, 2);
      }
    }

    II *= 1 / pow(n, 2);

    double sum_klm = 0;

    for (int k = 0; k < n - 2; k++) {

      for (int l = k + 1; l < n - 1; l++) {

        for (int m = l + 1; m < n; m++) {

          sum_klm += QQ(k, l)*QQ(k, m) + QQ(k, l)*QQ(l, m) + QQ(l, m)*QQ(k,m);
        }
      }
    }

    double JJ = 2 / pow(n, 3) * sum_klm;

    var_theta = pow(II, -2) * JJ / n;
  }

  // Note: Final theta estimate corrected by theta - 1 for our formulation

  List result;

  result.push_back(theta - 1,     "theta");
  result.push_back(var_theta, "var_theta");

  return result;
}
