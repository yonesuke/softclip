from turtle import forward
import jax.numpy as jnp
from jax import nn

class SoftClip:
    def __init__(self, low=None, high=None, hinge_softness=1.0):
        self.low = low
        self.high = high
        assert hinge_softness != 0.0, "hinge_softness cannot be zero"
        self.hinge_softness = hinge_softness

        if self.low is not None and self.high is not None:
            assert self.low < self.high, f"low must be smaller than high, but your input is low({low}) > high({high}))"
            self.transform_fn = lambda x: self.low + (self.high - self.low) * nn.sigmoid(x / self.hinge_softness)
            self.inverse_fn = lambda y: self.hinge_softness * jnp.log(y - self.low) - self.hinge_softness * jnp.log(self.high - y)
        
        elif self.low is not None and self.high is None:
            self.transform_fn = lambda x: self.hinge_softness * nn.softplus(x / self.hinge_softness) + self.low
            self.inverse_fn = lambda y: self.hinge_softness * jnp.log(jnp.exp((y - self.low) / self.hinge_softness) - 1.0)
        
        elif self.low is None and self.high is not None:
            self.transform_fn = lambda x: -self.hinge_softness * nn.softplus(x / self.hinge_softness) + self.high
            self.inverse_fn = lambda y: self.hinge_softness * jnp.log(jnp.exp((self.high - y) / self.hinge_softness) - 1.0)
        
        else:
            self.transform_fn = lambda x: x
            self.inverse_fn = lambda y: y

    def forward(self, x):
        return self.transform_fn(x)

    def backward(self, y):
        return self.inverse_fn(y)

    def __repr__(self):
        return f"softclip.SoftClip(low={self.low}, high={self.high}, hinge_softness={self.hinge_softness})"