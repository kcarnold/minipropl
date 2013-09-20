from collections import deque
import math
import random
import numpy as np


# Inspired by pymc
class ZeroProbability(ValueError):
    pass


class Unif(object):
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.sd = 10.  # FIXME.

    def logp(self, x):
        if x < self.lo or x > self.hi:
            raise ZeroProbability
        return -math.log(float(self.hi) - self.lo)

    def rand(self):
        return random.uniform(self.lo, self.hi)

    def propose(self, val):
        lo = self.lo
        hi = self.hi
        while True:
            x = val + random.normalvariate(0., self.sd)
            if lo < x < hi:
                return x


class Categorical(object):
    def __init__(self, probs):
        self.probs = probs
        self.logprobs = np.log(probs)

    def logp(self, choice):
        return self.logprobs[choice]

    def rand(self):
        return np.random.choice(len(self.probs), p=self.probs)

    def propose(self, val):
        while True:
            new_val = self.rand()
            if new_val != val:
                return new_val


class ModelState(object):
    def __init__(self, values=None):
        self.llk = 0.
        self.dists = {}
        self.values = values if values is not None else {}

    def draw_unif(self, name, lo, hi):
        return self._draw(name, Unif(lo, hi))

    def draw_categorical(self, name, probs):
        return self._draw(name, Categorical(probs))

    def _draw(self, name, dist):
        assert name not in self.dists
        self.dists[name] = dist
        if name in self.values:
            val = self.values[name]
        else:
            val = self.values[name] = dist.rand()
        self.llk += dist.logp(val)
        return val

    def exp_factor(self, factor):
        self.llk += factor

    def choose_value_to_change(self, choice_filter=None):
        # Change one of the values used this iteration.
        choices = self.dists.keys()
        if choice_filter is not None:
            choices = [choice for choice in choices if choice_filter(choice)]
        return random.choice(choices)

    def with_val_changed(self, to_change, new_value):
        new_values = dict(self.values)
        new_values[to_change] = new_value
        return ModelState(new_values)

    def make_proposal(self, to_change):
        return self.with_val_changed(
            to_change, self.dists[to_change].propose(self.values[to_change]))


class MHSampler(object):
    def __init__(self, program, memory_length=1000):
        self.program = program
        self.model_state = ModelState()
        self.results = deque(maxlen=memory_length)
        self.results.append(self.program(self.model_state))
        self.logps = deque(maxlen=memory_length)
        self.logps.append(self.model_state.llk)
        self.iter_count = 0
        self.accepts = 0
        self.rejects = 0

    def _step(self):
        # Propose a change
        to_change = self.model_state.choose_value_to_change()
        proposed_model_state = self.model_state.make_proposal(to_change)
        # Run it.
        result = self.program(proposed_model_state)

        # Stochastically accept.
        log_accept_prob = proposed_model_state.llk - self.model_state.llk
        if np.log(random.random()) < log_accept_prob:
            self.accepts += 1
            self.model_state = proposed_model_state
            self.logps.append(self.model_state.llk)
            self.results.append(result)
            return True
        else:
            self.rejects += 1
            return False

    def step(self):
        while not self._step():
            pass
        self.iter_count += 1
