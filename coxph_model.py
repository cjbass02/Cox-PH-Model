import numpy as np
from scipy.optimize import minimize

class CoxPHModel:
    def __init__(self):
        self.coef_ = None
        self.baseline_hazard_ = None
        self.baseline_survival_ = None
        self.event_times_ = None

    def _partial_likelihood(self, coef, X, T, E):
        """
        Defines the partial likelihood cost function for the model
        Parameters: The coefficients, covariates, time intervals, and event indicators
        Returns: The negative log partial likelihood value
        """
        risk_scores = np.dot(X, coef)
        exp_risk = np.exp(risk_scores)

        # Sort by time 
        sorted_indices = np.argsort(-T)
        E_desc = E[sorted_indices]
        exp_risk_desc = exp_risk[sorted_indices]

        # Cumulative sums from the back for risk sets
        cumsum_risk = np.cumsum(exp_risk_desc)
        log_part = 0.0
        # For each event, log_part += log of that subject's risk - log of sum of risk in risk set
        event_counter = 0
        for i, e in enumerate(E_desc):
            if e == 1:
                # i-th subject is an event, add risk score and subtract log(sum of risk in at-risk set)
                log_part += np.log(exp_risk_desc[i]) - np.log(cumsum_risk[i])
                event_counter += 1

        return -log_part

    def fit(self, X, T, E):
        """
        Fits the Cox proportional hazards model to the data
        Parameters: Covariates, time intervals, and event indicators
        Returns: None, assigns coefficients and baseline hazard to the model
        """
        initial_coef = np.zeros(X.shape[1])
        result = minimize(
            self._partial_likelihood,
            initial_coef,
            args=(X, T, E),
            method="L-BFGS-B"  
        )
        self.coef_ = result.x
        self. baseline_hazard_ = self._compute_baseline_hazard(X, T, E)

    def _compute_baseline_hazard(self, X, T, E):
        """
        Computes the baseline hazard function from the data using the breslow method
        """
        # Compute baseline hazard at each event time
        risk_scores = np.dot(X, self.coef_)
        exp_risk = np.exp(risk_scores)

        # Sort by time
        sorted_indices = np.argsort(T)
        T_sorted = T[sorted_indices]
        E_sorted = E[sorted_indices]
        exp_risk_sorted = exp_risk[sorted_indices]

        # Unique event times
        event_times = T_sorted[E_sorted == 1]
        self.event_times_ = event_times

        baseline_hazard = []
        # For each event time, hazard is # events at t / sum risk of those at risk at t
        for t in event_times:
            # Events at time t
            at_event = (T_sorted == t) & (E_sorted == 1)
            # At risk set: those with time >= t
            at_risk = (T_sorted >= t)
            numerator = np.sum(at_event)
            denominator = np.sum(exp_risk_sorted[at_risk])
            baseline_hazard.append(numerator / denominator)

        self.baseline_hazard_ = np.array(baseline_hazard)
        self.baseline_survival_ = np.cumprod(1 - self.baseline_hazard_)

    def predict_risk(self, X):
        return np.dot(X, self.coef_)

    def predict_expected_survival_time(self, X):

        risk_scores = self.predict_risk(X)
        exp_risk = np.exp(risk_scores)

        # Compute intervals between event times
        # Start from t=0 to the first event time, then from each event time to the next.
        intervals = np.diff(np.concatenate(([0], self.event_times_)))

        expected_times = []
        for exp_r in exp_risk:
            # Adjusted survival at each event time
            adjusted_survival = self.baseline_survival_ ** exp_r
            # Numerically approximate expected survival time via a Riemann sum:
            expected_time = np.sum(adjusted_survival * intervals)
            expected_times.append(expected_time)

        return np.array(expected_times)
