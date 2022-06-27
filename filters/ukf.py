# Unscented Kalman filter for neuronal state estimation
import numpy as np
from scipy.stats.distributions import chi2
from scipy.linalg import sqrtm
from tqdm import tqdm


class Ukf:
    """Unscented Kalman Filter (UKF) estimator"""
    
    def __init__(self, dynamics, measurements, inputs, slow_states, x0, P0, Q0, R0, kappa=0, sigma=0.5, robust=False):
        """
        Estimator variables:
            plant: (nonlinear) plant dynamics F, e.g. neuron dynamics
            p: NeuronType parameters
            theta: index of parameters to estimate (i.e. slow states)
            L: dimension of state
            x: estimated state (output)
            P: estimated covariance (output)
            x0: intitial state
            P0: initial covariance matrix
            y: observations
            u: inputs
            num_sp: number of sigmapoints (2*L + 1, where L = dim(x))
            alphas: sigmapoint weights
            Q: process noise covariance matrix
            R: measurement noise covariance matrix
        """
        self.plant = dynamics
        self.theta = slow_states
        self.L = x0.shape[1]
        self.y = measurements
        self.u = inputs
        self.t_sim = len(measurements)
        self.robust = robust

        # Robust UKF parameters
        if self.robust:
            self.lambda0 = 0.2     # lambda0 \in (0, 1)
            self.a = 5.0
            self.delta0 = 0.2      # delta0 \in (0, 1)
            self.b = 5.0
            self.threshold = chi2.ppf(1 - sigma, df=self.y.shape[1])

        # Sigmapoint parameters
        self.num_sp = 0
        self.kappa = kappa
        self.alphas = self.get_weights()

        # I.C.
        self.initialize(x0, P0, Q0, R0)

    def initialize(self, x0, P0, Q0, R0):
        """(Re)Initialize simulation"""

        if self.theta:
            self.p = np.repeat(np.array([list(self.plant.p.values())]), self.num_sp, axis=0)
        else:
            self.p = np.array([list(self.plant.p.values())])

        self.x = np.zeros((self.t_sim,self.L))
        self.x[:] = np.nan
        self.x[[0]] = x0
        self.P = np.zeros((self.t_sim*self.L,self.L))
        self.P[:] = np.nan
        self.P[:self.L,:] = P0

        self.Q = Q0
        self.R = R0
        self.Q_diag = np.zeros((self.t_sim*self.L,self.L))
        self.Q_diag[:] = np.nan
        self.Q_diag[:self.L,:] = np.diag(self.Q)[None]

        self.R_diag = np.zeros((self.t_sim*len(self.R),len(self.R)))
        self.R_diag[:] = np.nan
        self.R_diag[:len(self.R),:] = np.diag(self.R)[None]

        self.phi = np.zeros([self.t_sim])
        self.phi[:] = np.nan
        self.mu = np.zeros([self.t_sim])
        self.mu[:] = np.nan

    def get_weights(self):
        """Weights applied to the transformed sigmapoints to obtain predicted beliefs"""

        if self.kappa != 0:
            alphas = 0.5 / (self.L + self.kappa) * np.ones((2 * self.L + 1, 1))
            alphas[0] = self.kappa / (self.L + self.kappa)
            self.num_sp = 2 * self.L + 1
        else:
            alphas = 0.5 / self.L * np.ones((2 * self.L, 1))
            self.num_sp = 2 * self.L
        return alphas

    def unscented_transform(self, P, x):

        P = (P + P.T) / 2.0
        sigma_root = np.linalg.cholesky((self.L + self.kappa) * P)
        sigmapoints = x.T * np.ones((2 * self.L, self.L)) + np.concatenate((sigma_root.T, -sigma_root.T)) # shape: 2*L x L
        
        if self.kappa != 0:
            sigmapoint0 = x.T
            sigmapoints = np.concatenate((sigmapoint0, sigmapoints))
        
        return sigmapoints

    def run_estimation(self, int_factor=1, resample=False, no_progress=False):
        """Recursive state estimation"""

        N_slow = len(self.theta)    # number of parameters to estimate
        N_fast = self.L - N_slow    # number of states to estimate
        try:
            for k in tqdm(range(1, self.t_sim), disable=no_progress):
                ## PREDICTION STEP
                # Sample sigmapoints by passing in P_k_1 and x_k_1
                z_ = self.unscented_transform(self.P[self.L*(k-1):self.L*k, :], self.x[[k-1]].T)
                x_check_z = np.zeros_like(z_) # transformed sigmapoints
                # Update parameters
                if self.theta:
                    self.p[:, self.theta] = z_[:, N_fast:]

                # Pass the sigmapoints through the motion model
                x_check_tmp = self.plant.forward(z_[:, :N_fast], self.u[k, 0], self.p, int_factor)
                x_check_z[:, :N_fast] = x_check_tmp[:, :N_fast]
                x_check_z[:, N_fast:] = z_[:, N_fast:]

                # Alpha weighted sum of transformed sigmapoints --> predicted belief (x_check, P_check)
                x_check = x_check_z.T @ self.alphas
                x_cov_diff = x_check_z - x_check.T
                P_check = (self.alphas * x_cov_diff).T @ x_cov_diff + self.Q
                P_check = (P_check + P_check.T) / 2.0

                ## CORRECTION STEP
                if not np.isnan(self.y[k]):
                    if resample:
                        z = self.unscented_transform(P_check, x_check)
                        y_check_z = self.plant.observe(z)
                    else:
                        y_check_z = self.plant.observe(x_check_z) # NOTE: is this a mistake?
                        # y_check_z = self.plant.observe(z_)

                    # Alpha weighted sum of transformed sigmapoints
                    mu_y = np.sum(self.alphas * y_check_z, axis=0, keepdims=True)
                    yy_cov_diff = y_check_z - mu_y
                    sigma_yy = (self.alphas * yy_cov_diff).T @ yy_cov_diff + self.R
                    sigma_xy = (self.alphas * x_cov_diff).T @ yy_cov_diff

                    # Kalman gain and innovation calculation
                    # if not np.isnan(self.y[k]):
                    y_k = self.y[k]

                    K = sigma_xy @ np.linalg.inv(sigma_yy)
                    # innovation = self.y[k] - mu_y
                    innovation = y_k - mu_y
                    norm_innovation = np.linalg.inv(sqrtm(sigma_yy)) * innovation
                    # norm_sigma_yy = norm_innovation @ norm_innovation.T
                    # self.mu[k] = innovation[0]
                    self.mu[k] = norm_innovation[0]

                    # Corrected beliefs
                    x_hat = x_check + K @ innovation
                    P_hat = P_check - K @ sigma_xy.T
                    P_hat = (P_hat + P_hat.T) / 2.0

                    # Adjust covariance matrices Q and R
                    if self.robust:
                        phi = (innovation @ np.linalg.inv(sigma_yy) @ innovation.T)
                        ## phi = (norm_innovation @ norm_innovation.T)
                        # self.phi = np.concatenate((self.phi, phi[0]))
                        self.phi[k] = phi[0]
                        if phi > self.threshold:
                            x_hat, P_hat = self.adapt_covariances(x_hat, P_hat, y_k, phi.squeeze(), innovation, K)
                else:
                    x_hat = x_check
                    P_hat = P_check
                    if self.robust:
                        self.phi = np.concatenate((self.phi, np.array([0])))

                # Output
                self.x[[k]] = x_hat.T
                self.P[self.L*k:self.L*(k+1), :] = P_hat
                self.Q_diag[self.L*k:self.L*(k+1), :] = np.diag(self.Q)[None]
                self.R_diag[len(self.R)*k:len(self.R)*(k+1), :] = np.diag(self.R)[None]
        except Exception as e:
            print(repr(e))
        finally:
            return self.x, self.P

    def adapt_covariances(self, x_hat, P_hat, y_k, phi, innovation, K):
        """Adapt Q and R covariance matrices"""

        # Update Q
        lambda_ = np.max([self.lambda0, (phi - self.a * self.threshold) / phi])
        self.Q = (1 - lambda_) * self.Q + lambda_ * (K @ innovation.T @ innovation @ K.T)

        # Re-sample sigmapoints with new state estimate
        z = self.unscented_transform(P_hat, x_hat)
        x_cov_diff = z - x_hat.T

        y_check_z = self.plant.observe(z)
        mu_y = np.sum(self.alphas * y_check_z, axis=0, keepdims=True)
        yy_cov_diff = y_check_z - mu_y

        # Update R
        residual_y = y_k - self.plant.observe(x_hat.T)
        sigma_yy = (self.alphas * yy_cov_diff).T @ yy_cov_diff
        delta = np.max([self.delta0, (phi - self.b * self.threshold) / phi])
        self.R = (1 - delta) * self.R  + delta * (residual_y @ residual_y.T + sigma_yy)
        
        # Correct estimates
        P_ = (self.alphas * x_cov_diff).T @ x_cov_diff + self.Q
        sigma_xy = (self.alphas * x_cov_diff).T @ yy_cov_diff
        sigma_yy = sigma_yy + self.R                    
        K_ = sigma_xy @ np.linalg.inv(sigma_yy)
        innovation_ = y_k - mu_y

        # Corrected beliefs
        x_hat = x_hat + K_ @ innovation_.T
        P_hat = P_ - K_ @ sigma_yy @ K_.T
        P_hat = (P_hat + P_hat.T) / 2.0

        return x_hat, P_hat
