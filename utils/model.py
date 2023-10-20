import GPy
import numpy as np


class GPyModel:
    """GP model."""
    def __init__(self, input_dim, output_dim, noise_var, ker):
        super().__init__()

        self.ker = ker

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Training data containers.
        self.X_T = np.empty((0, self.input_dim))
        self.Y_T = np.empty((0, self.output_dim))

        self.noise_var = noise_var
        self.noise_std = np.sqrt(noise_var)
        
        self.model = None

    def add_sample(self, X_t, Y_t):
        """Add a new sample to the conditioning data of the GP."""
        self.X_T = np.concatenate([self.X_T, X_t], 0)
        self.Y_T = np.concatenate([self.Y_T, Y_t], 0)

        print("Sample count:", len(self.X_T))

    def update(self):
        """Update the GP."""
        train_X = self.X_T
        train_Y = self.Y_T

        self.model = GPy.models.GPRegression(train_X, train_Y, self.ker)
        self.model.Gaussian_noise.variance = self.noise_var

    def predict(self, test_X):
        """Get posterior mean and variance at test points."""
        m, v = self.model.predict(test_X, include_likelihood=False)
        
        return m, v

    def sample(self, test_X):
        """Get posterior samples at test points."""
        test_Y = self.model.posterior_samples_f(test_X, size=1)

        return test_Y
