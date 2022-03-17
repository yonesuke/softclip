# softclip

Simple JAX implementation of softclip, inspired by tensorflow probability

- <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/SoftClip>

softclip is a differentiable bijector from the real number space to some interval.
This is useful when you want to optimize a parameter that is assumed to be inside the interval [low, high].

## Installation

softclip can be installed with pip directly from GitHub, with the following command:

```
pip install git+https://github.com/yonesuke/softclip.git
```

## QuickStart

The `forward` method is the function from the real number space to the interval [low, high].
The `inverse` method is the function from the interval [low, high] to the real number space, and is the inverse function of `forward`.

```python
from softclip import SoftClip

bij = SoftClip(low=1.0, high=3.0, hinge_softness=0.5)
y = bij.forward(2.0) # y = 2.9640274
bij.inverse(y) # 1.9999975 ≒ 2.0
```

Simply set `low=0.0` or `high=0.0` to create a bijector to a positive/negative number domain.

```python
bij_positive = SoftClip(low=0.0)
bij_negative = SoftClip(high=0.0)
```

By transforming softclip to distrax with `to_distrax`, you can create distrax bijectors:

```python
bij_distrax = bij.to_distrax()
```
