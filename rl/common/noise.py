import numpy as np
from numpy.fft import rfftfreq, irfft

class ActionNoise:
    def __call__(self):
        raise NotImplementedError

    def reset(self):
        pass

# OU Noise taken from keras ddpg implementation
class OrnsteinUhlenbeckNoise(ActionNoise):
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class UniformNoise(ActionNoise):
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def __call__(self):
        return np.random.uniform(-self.scale, self.scale, self.shape)



class GaussianNoise(ActionNoise):
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def __call__(self):
        return np.random.normal(0, self.scale, self.shape)


class PinkNoise(ActionNoise):
    def __init__(self, shape, scale, seq_len=1024, rng=None):
        self.shape = shape
        self.scale = scale
        self.seq_len = seq_len
        self.rng = rng if rng is not None else np.random.default_rng()

        self._reset_buffer()

    def _reset_buffer(self):
        self.buffer = self._generate_pink_block()
        self.idx = 0

    def _generate_pink_block(self):
        n = self.seq_len
        dim = self.shape[0]

        # Frequencies for real FFT
        freqs = rfftfreq(n)

        # Avoid division by zero at DC
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0

        # Pink noise scaling: 1 / sqrt(f)
        scaling = 1.0 / np.sqrt(freqs)

        # Generate random complex coefficients
        real = self.rng.normal(size=(dim, len(freqs)))
        imag = self.rng.normal(size=(dim, len(freqs)))

        spectrum = (real + 1j * imag) * scaling

        # Ensure DC is real
        spectrum[:, 0] = spectrum[:, 0].real + 0j

        # Transform back to time domain
        noise = irfft(spectrum, n=n, axis=-1)

        # Normalize to unit variance
        noise /= np.std(noise, axis=-1, keepdims=True)

        return noise

    def __call__(self):
        if self.idx >= self.seq_len:
            self._reset_buffer()

        sample = self.buffer[:, self.idx]
        self.idx += 1

        return self.scale * sample
    

    def reset(self):
        self._reset_buffer()
