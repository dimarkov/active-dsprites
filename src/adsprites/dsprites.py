import jax.numpy as jnp

from dataclasses import dataclass
from itertools import product
from typing import Dict, Optional
from jaxtyping import Array, Float, Int16, PRNGKeyArray
from jax import vmap, lax, nn
from jax import random as jr
from importlib.resources import files

Image = Float[Array, "channels height width"]
BatchImage = Float[Image, "batch"]
State = Int16[Array, "color shape scale orientation posx posy"]
BatchState = Int16[State, "batch"]

orientation_mask = nn.one_hot(3, 6, dtype=bool)

@dataclass
class DSprites(object):

    state_sizes: State
    state_bases: State
    metadata = Dict[str, Array]
    imgs: BatchImage
    states: Array
    batch_size: int
    key: PRNGKeyArray

    def __init__(self, batch_size: Optional[int] = 1, seed: Optional[int] = 0):
        self.key = jr.PRNGKey(seed)
        self.batch_size = batch_size

        filepath = str(files("dsprites-dataset").joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"))
        data = jnp.load(
            filepath, 
            allow_pickle=True, 
            encoding='bytes'
        )

        self.imgs = jnp.expand_dims(data['imgs'], -3).astype(jnp.float16)
        self.metadata = data['metadata'][()]
        self.state_sizes = self.metadata[b'latents_sizes']
        self.state_bases = jnp.concatenate((self.state_sizes[::-1].cumprod()[::-1][1:],
                                jnp.array([1,])))

    def state_to_index(self, states):
        return jnp.dot(states, self.state_bases).astype(int)

    def sample_states(self, keys, sizes, shape=()):
        dist = lambda key, size, shape: jr.randint(key, shape, 0, size)
        smpl = vmap(dist, in_axes=(0, 0, None), out_axes=-1)(keys, sizes, shape)
        return smpl

    def init_env(self) -> (BatchImage, BatchState):
        self.key, _key = jr.split(self.key)
        keys = jr.split(_key, len(self.state_sizes))
        states = self.sample_states(keys, self.state_sizes, (self.batch_size,))
        idxs = self.state_to_index(states)

        return self.imgs[idxs], states

    def _move(self, state, action):
        # increase latent state by value given in action
        state = state + jnp.pad(action, (2, 0))

        # allow unconstrained rotation
        corrected_state = state - jnp.sign(state) * self.state_sizes
        mask = orientation_mask * (corrected_state >= 0)
        state = jnp.where(mask, corrected_state, state)
        return jnp.clip(state, a_min=0, a_max=self.state_sizes - 1)

    def step(self, states: Int16[Array, "..."], actions: Int16[Array, "..."]) -> (BatchImage, BatchState):
        states = vmap(self._move)(states, actions)
        idxs = self.state_to_index(states)

        return self.imgs[idxs], states