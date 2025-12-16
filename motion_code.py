import time
from tqdm import tqdm
import numpy as np

from data_processing import load_data, process_data_for_motion_codes, split_train_test_forecasting
from motion_code_utils import optimize_motion_codes, classify_predict_helper
from sparse_gp import sigmoid, q
from utils import accuracy, RMSE

class MotionCode:
    """
    Class for motion code model

    Attributes
    ----------
    m: int
        Number of inducing points
    Q: int
        Number of kernel components
    latent_dim: int
        Dimension of motion codes
    sigma_y: float
        Noise of the target variable
    model_path: str
        Path to save/load the model
    num_motion: int
        Number of stochastic processes the model considered
    X_m: numpy.ndarray
        The common transformation for all stochastic processes underlying collections of time series data
    Z: numpy.ndarray
        Stacked motion codes
    Sigma: numpy.ndarray
        Stacked kernel parameters for the exponents of all stochastic processes
    W: numpy.ndarray
        Stacked kernel parameters for the scales of all stochastic processes
    kernel_params:
        Stacked pairs of kernel parameters (Sigma, W) for all stochastic processes
    mu_ms: numpy.ndarray
        Stacked mean prediction over the inducing points for all stochastic processes
    A_ms: numpy.ndarray
        Stacked covariance of mu_ms for all stochastic processes
    K_mm_invs: numpy.ndarray
        Stacked inverses of the kernel over inducing points for all stochastic processes
        
    Methods
    -------
    fit(X_train, Y_train, labels_train, model_path)
        Train model on `(X_train, Y_train)` time series with collection labels `labels_train`
        and then save model to `model_path`

    load(model_path='')
        Load model at `model_path`
    classify_predict(X_test, Y_test):
        Predict the label for a single time series `(X_test, Y_test)` in classification problem.

    classify_predict_on_batches(X_test_list, Y_test_list, true_labels)
        Predict the labels for a list of time series `(X_test_list, Y_test_list)`
        and then compare against `true_labels` and return accuracy

    forecast_predict(self, test_time_horizon, label)
        Predict future values on `test_time_horizon` for stochastic process with label `label`
    
    forecast_predict_on_batches(self, test_time_horizon, Y_test_list, labels)
        Predict future (mean) values on `test_time_horizon` for all stochastic processes
        and then use the results to compare against the time series list `Y_test_list` with labels `labels`
        for RMSE errors.

    """
    def __init__(self, m=10, Q=1, latent_dim=2, sigma_y=0.1, R=3, lambda_reg=0.0, lambda_weight_reg=0.0):
        self.m = m # Num inducing pts
        self.Q = Q # Num of kernel components
        self.latent_dim = latent_dim # Dim of motion code
        self.sigma_y = sigma_y # Noise of target
        self.R = R # Number of global motion codes (experts)
        self.lambda_reg = lambda_reg # L2 regularization coefficient for global motion codes Z
        self.lambda_weight_reg = lambda_weight_reg # Regularization coefficient for pairwise cosine similarity between expert weight vectors

    def fit(self, X_train, Y_train, labels_train, model_path):
        start_time = time.time()
        self.model_path = model_path
        optimize_motion_codes(X_train, Y_train, labels_train, model_path=model_path, 
              m=self.m, Q=self.Q, latent_dim=self.latent_dim, sigma_y=self.sigma_y, R=self.R, 
              lambda_reg=self.lambda_reg, lambda_weight_reg=self.lambda_weight_reg)
        self.train_time = time.time() - start_time

    def load(self, model_path=''):
        if len(model_path) == 0 and self.model_path is not None:
            model_path = self.model_path
        params = np.load(model_path + '.npy', allow_pickle=True).item()
        self.X_m, self.Z, self.Sigma, self.W = params.get('X_m'), params.get('Z'), params.get('Sigma'), params.get('W') 
        self.mu_ms, self.A_ms, self.K_mm_invs = params.get('mu_ms'), params.get('A_ms'), params.get('K_mm_invs')
        # R is stored in saved model, or use shape of Z to infer
        if 'R' in params:
            self.R = params.get('R')
        else:
            self.R = self.Z.shape[0]  # Infer from Z shape (backward compatibility)
        # num_motion is based on number of processes (Sigma/W shape), not R
        self.num_motion = self.Sigma.shape[0]
        # Load expert weights if available, otherwise use uniform weights
        if 'expert_weights' in params:
            self.expert_weights = params.get('expert_weights')
        else:
            # Fallback to uniform weights (backward compatibility)
            self.expert_weights = np.ones((self.num_motion, self.R)) / self.R
        self.kernel_params = []
        for k in range(self.num_motion):
            self.kernel_params.append((self.Sigma[k], self.W[k]))

    def classify_predict(self, X_test, Y_test):
        return classify_predict_helper(X_test, Y_test, self.kernel_params, 
                                       self.X_m, self.Z, self.mu_ms, self.A_ms, self.K_mm_invs, 
                                       R=self.R, expert_weights=self.expert_weights)
    
    def classify_predict_on_batches(self, X_test_list, Y_test_list, true_labels):
        # Predict each trajectory/timeseries in the test dataset
        num_predicted = 0
        pred = []; gt = []
        if isinstance(Y_test_list, list):
            num_test = len(Y_test_list)
        else:
            num_test = Y_test_list.shape[0]
        pbar = tqdm(zip(X_test_list, Y_test_list), total=num_test, leave=False)
        num_predicted = 0
        for X_test, Y_test in pbar:
            # Get predict and ground truth motions
            pred_label = self.classify_predict(X_test, Y_test)
            gt_label = true_labels[num_predicted]
            pbar.set_description(f'Predict: {pred_label}; gt: {gt_label}')
            # Append results to lists for final evaluation
            pred.append(pred_label); gt.append(gt_label)
            num_predicted += 1

        # Accurary evaluation
        return accuracy(pred, gt)
    
    def forecast_predict(self, test_time_horizon, label):
        k = label
        # Use mixture of experts: compute predictions from all R experts and weight them using learned static weights
        expert_predictions = []
        
        for r in range(self.R):
            X_m_r = sigmoid(self.X_m @ self.Z[r])
            if isinstance(self.mu_ms[k], list):
                mean, covar = q(test_time_horizon, X_m_r, 
                               self.kernel_params[k], self.mu_ms[k][r], self.A_ms[k][r], self.K_mm_invs[k][r])
            else:
                # Backward compatibility: single expert
                mean, covar = q(test_time_horizon, X_m_r, 
                               self.kernel_params[k], self.mu_ms[k], self.A_ms[k], self.K_mm_invs[k])
            expert_predictions.append((mean, covar))
        
        # Use learned static weights for this class
        if hasattr(self, 'expert_weights') and self.expert_weights is not None:
            weights = self.expert_weights[k]  # Shape (R,)
        else:
            # Fallback to uniform weights if not available (backward compatibility)
            weights = np.ones(self.R) / self.R
        
        # Weight and combine predictions
        weighted_mean = np.zeros_like(expert_predictions[0][0])
        weighted_covar = np.zeros_like(expert_predictions[0][1])
        
        for r, (mean, covar) in enumerate(expert_predictions):
            weighted_mean += weights[r] * mean
            weighted_covar += weights[r] * covar
        
        return weighted_mean, weighted_covar
    
    def forecast_predict_on_batches(self, test_time_horizon, Y_test_list, labels):
        # Average prediction for each type of motion.
        mean_preds = []
        for k in range(self.num_motion):
            mean, _ = self.forecast_predict(test_time_horizon, label=k)
            mean_preds.append(mean)
        
        all_errors = [[] for _ in range(self.num_motion)]
        
        for i in range(len(Y_test_list)):
            label = labels[i]
            all_errors[label].append(RMSE(mean_preds[label], Y_test_list[i]))

        errs = np.zeros(self.num_motion)
        for i in range(self.num_motion):
            errs[i] = np.mean(np.array(all_errors[i]))
        
        return errs

## Convenient functions that combine train, load, test, report errors.
def motion_code_classify(model, name,
                        X_train, Y_train, labels_train,
                        X_test, Y_test, labels_test,
                        load_existing_model=False):
    model_path = 'saved_models/' + name + '_classify'
    if not load_existing_model:
        model.fit(X_train, Y_train, labels_train, model_path)
    model.load(model_path)
    acc = model.classify_predict_on_batches(X_test, Y_test, labels_test)
    return acc

def motion_code_forecast(model, name, X_train, Y_train, labels,
                         test_time_horizon, Y_test, load_existing_model=False):
    model_path = 'saved_models/' + name + '_forecast'
    if not load_existing_model:
        model.fit(X_train, Y_train, labels, model_path)
    model.load(model_path)
    err = model.forecast_predict_on_batches(test_time_horizon, Y_test, labels)

    return err