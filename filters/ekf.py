# Extended Kalman filter for neuronal state estimation
import numpy as np
from scipy.stats.distributions import chi2
from tqdm import tqdm

class Ekf:
    """Extended Kalman Filter (EKF) estimator"""
    
    def __init__(self, dynamics, measurements, inputs, slow_states, x0, P0, Q0, R0, sigma=0.5, robust=False):
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
            Q: process noise covariance matrix
            R: measurement noise covariance matrix
        """
        self.plant = dynamics
        self.theta = slow_states
        self.L = x0.shape[1]
        self.y = measurements
        self.u = inputs
        self.t_sim = len(measurements)
        self.max_iter = 1
        self.eps = 0.001
        self.robust = robust

        # Robust UKF parameters
        if self.robust:
            self.lambda0 = 0.1     # lambda0 \in (0, 1)
            self.a = 10.0
            self.delta0 = 0.1      # delta0 \in (0, 1)
            self.b = 2.0
            self.threshold = chi2.ppf(1 - sigma, df=self.y.shape[1])

        # I.C.
        self.initialize(x0, P0, Q0, R0)

    def initialize(self, x0, P0, Q0, R0):
        """(Re)Initialize simulation"""

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

    def run_correction(self, x_op, P_op, k):
        """Run Kalman filter correction iteration"""
        
        G_k = self.plant.linearized_observation(x_op)
        sigma_yy = G_k @ P_op @ G_k.T + self.R
        K = P_op @ G_k.T @ np.linalg.inv(sigma_yy)
        # K = P_op @ G_k.T @ np.linalg.inv(G_k @ P_op @ G_k.T + self.R)
        y_check = self.plant.observe(x_op)
        innovation = self.y[k] - y_check

        x_hat = x_op + (K @ innovation).T
        P_hat = (np.eye(self.L) - K @ G_k) @ P_op
        P_hat = (P_hat + P_hat.T) / 2.0

        if self.robust:
            phi = (innovation @ np.linalg.inv(sigma_yy) @ innovation.T)
            self.phi[k] = phi[0]
            if phi > self.threshold:
                x_hat, P_hat = self.adapt_covariances(x_op, P_op, self.y[k], self.u[k, 0], phi.squeeze(), innovation, K)

        return x_hat, P_hat

    def adapt_covariances(self, x_hat, P_hat, y_k, u_k, phi, innovation, K):
        """Adapt Q and R covariance matrices"""

        # Update Q
        lambda_ = np.max([self.lambda0, (phi - self.a * self.threshold) / phi])
        self.Q = (1 - lambda_) * self.Q + lambda_ * (K @ innovation.T @ innovation @ K.T)

        # Update R
        residual_y = y_k - self.plant.observe(x_hat)
        G_k = self.plant.linearized_observation(x_hat)
        sigma_yy = G_k @ P_hat @ G_k.T
        delta = np.max([self.delta0, (phi - self.b * self.threshold) / phi])
        self.R = (1 - delta) * self.R  + delta * (residual_y @ residual_y.T + sigma_yy)
        
        # Correct estimates
        F_k = self.plant.linearized_model(x_hat, u_k, self.p)
        P_ = F_k @ P_hat @ F_k + self.Q
        P_ = (P_ + P_.T) / 2.0
        sigma_yy += self.R
        K_ = P_hat @ G_k.T @ np.linalg.inv(sigma_yy)

        # Corrected beliefs
        x_hat = x_hat + (K_ @ residual_y).T
        P_hat = (np.eye(self.L) - K_ @ G_k) @ P_hat
        P_hat = (P_hat + P_hat.T) / 2.0

        return x_hat, P_hat
    
    def run_estimation(self, x_gt=None):
        """Recursive state estimation"""

        N_slow = len(self.theta)    # number of parameters to estimate
        N_fast = self.L - N_slow    # number of states to estimate
        for k in tqdm(range(1, self.t_sim)):
            ## PREDICTION STEP
            if x_gt is not None:
                F_k_1 = self.plant.linearized_model(x_gt[[k-1]], self.u[k, 0], self.p)
            else:
                F_k_1 = self.plant.linearized_model(self.x[[k-1]], self.u[k, 0], self.p)

            if self.theta:
                x_check = np.zeros_like(self.x[[k-1]])
                self.p[:, self.theta] = self.x[k-1, N_fast:]
                x_check_tmp = self.plant.forward(self.x[[k-1], :N_fast], self.u[k, 0], self.p)
                x_check[:, :N_fast] = x_check_tmp
                x_check[:, N_fast:] = self.x[[k-1], N_fast:]
            else:            
                x_check = self.plant.forward(self.x[[k-1]], self.u[k, 0], self.p)

            P_check = F_k_1 @ self.P[self.L*(k-1):self.L*k, :] @ F_k_1.T + self.Q
            P_check = (P_check + P_check.T) / 2.0

            ## CORRECTION STEP (iterative)
            x_op = x_check
            P_op = P_check
            for i in range(self.max_iter):
                x_hat, P_hat = self.run_correction(x_op, P_op, k)

                eps_norm = np.linalg.norm(x_hat - x_op)
                if eps_norm < self.eps:
                    break

                x_op = x_hat
                P_op = P_hat

            # Output
            self.x[[k]] = x_hat
            self.P[self.L*k:self.L*(k+1), :] = P_hat
            self.Q_diag[self.L*k:self.L*(k+1), :] = np.diag(self.Q)[None]
            self.R_diag[len(self.R)*k:len(self.R)*(k+1), :] = np.diag(self.R)[None]

        return self.x, self.P
