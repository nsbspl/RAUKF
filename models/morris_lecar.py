# Neuron classes for Morris-Lecar simulation
import numpy as np

class MorrisLecar2D:
    """
    Single compartment 2-dimensional Morris-Lecar model
    (https://doi.org/10.1371/journal.pcbi.1000198)
    """

    def __repr__(self):
        return repr("Single compartment 2D Morris-Lecar model")

    def __init__(self, p, obs_index=[0]):
        self.p = p # NeuronType parameters
        self.num_states = 2
        self.obs_index = obs_index
        self.initialize_state()

    def initialize_state(self):
        """Initialize state variables with ICs"""

        self.x = np.array([[-70, 0.25 * 1e-4]])
    
    def test(self, I_in_nA, int_factor=1):
        """Integrate neuron dynamics"""

        yy = np.zeros((int(self.p["t_stop"] / self.p["dt"]), self.num_states))
        yy[[0]] = self.x
        k = 1
        linspace = np.arange(self.p["dt"], self.p["t_stop"], self.p["dt"])
        params = np.array([list(self.p.values())])
        for t in linspace:
            yy[k] = self.forward(self.x, I_in_nA[k, 0], params, int_factor)
            self.x = yy[[k]]
            k= k + 1
        return yy

    def forward(self, x_k_1, I_k, p, int_factor=1):
        """RK4 integration"""
        dt = p[0, 1] / int_factor

        x_k = x_k_1
        for n in range(int_factor):
            # k1 = dt * self.step(x_k, I_k, p)
            # k2 = dt * self.step(x_k + k1 / 2.0, I_k, p)
            # k3 = dt * self.step(x_k + k2 / 2.0, I_k, p)
            # k4 = dt * self.step(x_k + k3, I_k, p)
            # x_k = x_k + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0

            """Modified Euler integration (Moye et al, 2018)"""
            k1 = dt * self.step(x_k, I_k, p)
            k2 = dt * self.step(x_k + k1, I_k, p)
            x_k = x_k_1 + k1 / 2.0 + k2 / 2.0
            x_k = x_k_1 + k1

        return x_k

    def step(self, x_k_1, I_k, p):
        """Advance process model by one timestep"""
        
        # Leak current
        I_L = p[:, 9] * (x_k_1[:, 0] - p[:, 4])

        # Fast current
        m_inf = 0.5 * (1 + np.tanh((x_k_1[:, 0] - p[:, 10]) / p[:, 11]))
        I_Ca = p[:, 7] * m_inf * (x_k_1[:, 0] - p[:, 6])

        # Slow current
        w_inf = 0.5 * (1 + np.tanh((x_k_1[:, 0] - p[:, 12]) / p[:, 13]))
        I_K = p[:, 8] * x_k_1[:, 1] * (x_k_1[:, 0] - p[:, 5])
        tau_w = np.cosh(0.5 * (x_k_1[:, 0] - p[:, 12]) / p[:, 13])

        dvdt = (I_k - I_L - I_K - I_Ca) / p[:, 2]
        dwdt = (w_inf - x_k_1[:, 1]) * p[:, 14] * tau_w
        return np.concatenate((dvdt[:, np.newaxis], dwdt[:, np.newaxis]), axis=1)

    def linearized_model(self, x_k_1, u_k, p):
        """Returns the Jacobian of the system at the operating point
            (Jacobian derived analytically offline includes conductances as states)
        """
        
        # Jacobian
        x_dim = x_k_1.shape[1]
        T = p[0, 1]
        D = np.eye(x_dim)
        D[0, 0] += T * (-(p[:, 9] - (p[:, 7]*(np.tanh((p[:, 10] - x_k_1[:, 0])/p[:, 11]) - 1))/2 + p[:, 8]*x_k_1[:, 1] + (p[:, 7]*(p[:, 6] - x_k_1[:, 0])*(np.tanh((p[:, 10] - x_k_1[:, 0])/p[:, 11])**2 - 1))/(2*p[:, 11]))/p[:, 2])
        D[0, 1] = T * ((p[:, 8]*(p[:, 5] - x_k_1[:, 0]))/p[:, 2])
        D[0, 2] = T * (p[:, 4] - x_k_1[:, 0]) # g_leak conductance
        D[0, 3] = T * x_k_1[:, 1] * (p[:, 5] - x_k_1[:, 0]) # g_slow conductance
        D[0, 4] = T * 0.5 * (1 + np.tanh((x_k_1[:, 0] - p[:, 10]) / p[:, 11])) * (p[:, 6] - x_k_1[:, 0]) # g_fast conductance
        D[1, 0] = T * ((p[:, 14]*np.sinh((p[:, 12] - x_k_1[:, 0])/(2*p[:, 13]))*(x_k_1[:, 1] + np.tanh((p[:, 12] - x_k_1[:, 0])/p[:, 13])/2 - 1/2))/(2*p[:, 13]) - (p[:, 14]*np.cosh((p[:, 12] - x_k_1[:, 0])/(2*p[:, 13]))*(np.tanh((p[:, 12] - x_k_1[:, 0])/p[:, 13])**2 - 1))/(2*p[:, 13]))
        D[1, 1] += T * (-p[:, 14]*np.cosh((p[:, 12] - x_k_1[:, 0])/(2*p[:, 13])))

        return D

    def observe(self, x):
        """Record measurement"""

        # NOTE: for now, direct observation of the membrane voltage: g(x) = x[:, 0]
        return x[:, [0]]

    def linearized_observation(self, x_k):
        """Linearized observation model (identity in the case of full-state observation"""
        
        G = np.zeros((len(self.obs_index), x_k.shape[1]))
        G[self.obs_index, self.obs_index] = 1
        return G

class MorrisLecar3D:
    """
    Single compartment 3-dimensional Morris-Lecar model
    (https://doi.org/10.1371/journal.pcbi.1000198)
    """

    def __repr__(self):
        return repr("Single compartment 3D Morris-Lecar model")

    def __init__(self, p, inward=False, obs_index=[0]):
        self.p = p # NeuronType parameters
        self.num_states = 3
        self.obs_index = obs_index
        self.inward = inward
        self.initialize_state()

    def initialize_state(self):
        """Initialize state variables with ICs"""

        self.x = np.array([[-70, 0.25 * 1e-4, 0.25 * 1e-4]])
    
    def test(self, I_in_nA, int_factor=1):
        """Integrate neuron dynamics"""

        yy = np.zeros((int(self.p["t_stop"] / self.p["dt"]), self.num_states))
        yy[[0]] = self.x
        k = 1
        linspace = np.arange(self.p["dt"], self.p["t_stop"], self.p["dt"])
        params = np.array([list(self.p.values())])
        for t in linspace:
            yy[k] = self.forward(self.x, I_in_nA[k, 0], params, int_factor)
            self.x = yy[[k]]
            k= k + 1
        return yy

    def forward(self, x_k_1, I_k, p, int_factor=1):
        """RK4 integration"""

        dt = p[0, 1] / int_factor

        x_k = x_k_1
        for n in range(int_factor):
            # k1 = dt * self.step(x_k, I_k, p)
            # k2 = dt * self.step(x_k + k1 / 2.0, I_k, p)
            # k3 = dt * self.step(x_k + k2 / 2.0, I_k, p)
            # k4 = dt * self.step(x_k + k3, I_k, p)
            # x_k = x_k + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0

            """Modified Euler integration (Moye et al, 2018)"""
            k1 = dt * self.step(x_k, I_k, p)
            k2 = dt * self.step(x_k + k1, I_k, p)
            x_k = x_k_1 + k1 / 2.0 + k2 / 2.0
            x_k = x_k_1 + k1

        return x_k

    def step(self, x_k_1, I_k, p):
        """Advance process model by one timestep"""
        
        # Leak current
        I_L = p[:, 10] * (x_k_1[:, 0] - p[:, 4])

        # Fast current
        m_inf = 0.5 * (1 + np.tanh((x_k_1[:, 0] - p[:, 12]) / p[:, 13]))
        I_Ca = p[:, 8] * m_inf * (x_k_1[:, 0] - p[:, 6])

        # Slow current
        y_inf = 0.5 * (1 + np.tanh((x_k_1[:, 0] - p[:, 14]) / p[:, 15]))
        I_del_rec = p[:, 9] * x_k_1[:, 1] * (x_k_1[:, 0] - p[:, 5])
        tau_y = p[:, 18] * np.cosh(0.5 * (x_k_1[:, 0] - p[:, 14]) / p[:, 15])
        
        z_inf = 0.5 * (1 + np.tanh((x_k_1[:, 0] - p[:, 16]) / p[:, 17]))
        I_sub = p[:, 11] * x_k_1[:, 2] * (x_k_1[:, 0] - p[:, 7])
        tau_z = p[:, 19] * np.cosh(0.5 * (x_k_1[:, 0] - p[:, 16]) / p[:, 17])

        dvdt = (I_k - I_L - I_del_rec - I_sub - I_Ca) / p[:, 2]
        dydt = tau_y * (y_inf - x_k_1[:, 1])
        dzdt = tau_z * (z_inf - x_k_1[:, 2])
        return np.concatenate((dvdt[:, np.newaxis], dydt[:, np.newaxis], dzdt[:, np.newaxis]), axis=1)

    def observe(self, x):
        """Record measurement"""

        # NOTE: for now, direct observation of the membrane voltage: g(x) = x[:, 0]
        return x[:, [0]]

    def linearized_observation(self, x_k):
        """Linearized observation model (identity in the case of full-state observation"""
        
        G = np.zeros((len(self.obs_index), x_k.shape[1]))
        G[self.obs_index, self.obs_index] = 1
        return G
