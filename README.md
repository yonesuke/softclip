# softclip
JAX implementation of softclip, inspired by tensorflow probability
- https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/SoftClip

## Installation
softclip can be installed with pip directly from GitHub, with the following command:
```
pip install git+git://github.com/yonesuke/softclip.git
```

## QuickStart
```python
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)
from softclip import SoftClip

bij = SoftClip(low=1.0, high=3.0, hinge_softness=0.5)
xs = jnp.arange(10) # DeviceArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64)
ys = bij.forward(xs) # DeviceArray([2.        , 2.76159416, 2.96402758, 2.99505475, 2.9993293 , 2.9999092 , 2.99998771, 2.99999834, 2.99999977, 2.99999997],            dtype=float64)
bij.backward(ys) # DeviceArray([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float64)
```
