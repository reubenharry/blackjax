"""Tests of the MCLMC adaptation options: Fisher mass matrix, larger L, initial step size search."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.adaptation.mclmc_adaptation import (
    fisher_inverse_mass_matrix,
    make_step_size_search,
)

SCALES = jnp.array([0.1, 1.0, 10.0, 3.0, 0.3, 2.0])


def logdensity_fn(x):
    return -0.5 * jnp.sum(jnp.square(x / SCALES))


def tune(position, **kwargs):
    key = jax.random.key(0)
    init_key, tune_key = jax.random.split(key)
    state = blackjax.mcmc.mclmc.init(position, logdensity_fn, init_key)
    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.mclmc.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix,
    )
    return blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel, num_steps=2000, state=state, rng_key=tune_key, **kwargs
    )


class MCLMCAdaptationTest(absltest.TestCase):
    def test_fisher_mass_matrix_limits(self):
        variances = jnp.array([4.0, 1.0, 0.25])
        grad_squared = 1.0 / jnp.array([2.0, 1.0, 0.5])
        np.testing.assert_allclose(
            fisher_inverse_mass_matrix(variances, grad_squared, 0.0), variances
        )
        np.testing.assert_allclose(
            fisher_inverse_mass_matrix(variances, grad_squared, 1.0),
            jnp.sqrt(variances / grad_squared),
        )

    def test_fisher_mass_matrix_falls_back_on_bad_moments(self):
        variances = jnp.array([4.0, 1.0])
        result = fisher_inverse_mass_matrix(variances, jnp.array([0.0, jnp.nan]), 1.0)
        np.testing.assert_allclose(result, variances)

    def test_defaults_recover_scales_and_larger_L(self):
        position = jax.random.normal(jax.random.key(1), (6,)) * SCALES
        _, params, steps = tune(position)
        np.testing.assert_allclose(
            jnp.sqrt(params.inverse_mass_matrix), SCALES, rtol=0.5
        )
        self.assertGreater(params.L, 2.0 * jnp.sqrt(6.0))  # original rule gives ~ sqrt(d) scale
        self.assertTrue(jnp.isfinite(params.step_size) and params.step_size > 0)
        self.assertGreater(steps, 0)

    def test_original_behaviour_with_options_off(self):
        position = jax.random.normal(jax.random.key(1), (6,)) * SCALES
        _, params, steps = tune(
            position, initial_step_size_search=False, fisher_weight=0.0, L_factor=None
        )
        self.assertEqual(int(steps), 2000)  # burn-in defaults to num_steps and is fully used
        self.assertTrue(jnp.all(params.inverse_mass_matrix > 0))

    def test_burn_in_budget(self):
        position = jax.random.normal(jax.random.key(1), (6,)) * SCALES
        for burn_in in (300, 1000):
            _, _, steps = tune(position, num_burn_in_steps=burn_in)
            self.assertLessEqual(int(steps), burn_in)
            self.assertGreater(int(steps), burn_in - 40)  # the search returns at most its reserve
        _, _, steps = tune(position)  # default: burn-in = num_steps = 2000
        self.assertLessEqual(int(steps), 2000)

    def test_search_stops_early_in_typical_set_and_grows_far_away(self):
        params = blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState(
            jnp.sqrt(6.0), 0.01, jnp.ones(6)
        )
        kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity_fn,
            integrator=blackjax.mcmc.mclmc.isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        search = make_step_size_search(kernel, max_steps=40)
        far = blackjax.mcmc.mclmc.init(30.0 * SCALES, logdensity_fn, jax.random.key(2))
        _, far_params, far_steps = search(far, params, jax.random.key(3))
        self.assertGreater(far_params.step_size, 10 * 0.01)  # grew while log-density increased
        near = blackjax.mcmc.mclmc.init(0.0 * SCALES, logdensity_fn, jax.random.key(4))
        _, _, near_steps = search(near, params, jax.random.key(5))
        self.assertLess(near_steps, 40)


if __name__ == "__main__":
    absltest.main()
