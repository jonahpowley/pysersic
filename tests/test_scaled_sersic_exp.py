from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.convolution import Gaussian2DKernel
from numpyro import distributions as dist, handlers, infer, optim
from pysersic import FitSingle, priors, rendering
from pysersic.multiband import FitMultiBandPoly, FitMultiBandBSpline


@pytest.fixture(scope='module')
def make_fitter():
    psf = Gaussian2DKernel(0.8, x_size=5, y_size=5).array
    renderer = rendering.MoGFourierRenderer((16, 16), psf)
    truth = dict(xc=7.5, yc=7.5, flux=100., f_1=0.4, r_eff_2=3.,
                 r_eff_1=1.35, n=2.5, ellip_1=0.15, ellip_2=0.3, theta=0.7)
    image = np.asarray(renderer.render_source(truth, 'sersic_exp'))
    rms = np.full(image.shape, 0.1)
    image = image + np.random.default_rng(42).normal(scale=rms)

    def make(low=0.5, high=6., suffix=''):
        prior = priors.ScaledSersicExpPrior(suffix=suffix)
        for name, scale in [('xc', 0.1), ('yc', 0.1), ('flux', 5.)]:
            prior.set_gaussian_prior(name, truth[name], scale)
        for name, bounds in dict(f_1=(0.25, 0.55), n=(1.5, 3.5), ellip_1=(0.05, 0.3),
                                 ellip_2=(0.15, 0.45), theta=(0.4, 1.), u_1=(0., 1.)).items():
            prior.set_uniform_prior(name, *bounds)
        prior.set_uniform_prior('r_eff_2', low, high)
        return FitSingle(image, rms, psf, prior, renderer=rendering.MoGFourierRenderer)
    return make


@pytest.mark.parametrize('kind', ['uniform', 'truncated', 'custom', 'decreasing', 'shifted_lognormal'])
@pytest.mark.parametrize('low', [0.2, 0.5, 1.7])
def test_prior_ordering(make_fitter, kind, low):
    prior = make_fitter().prior
    if kind == 'uniform':
        prior.set_uniform_prior('r_eff_2', low, low + 8.)
    elif kind == 'truncated':
        prior.set_truncated_gaussian_prior('r_eff_2', 5., 2., low=low)
    elif kind == 'custom':
        prior.set_custom_prior('r_eff_2', dist.TruncatedNormal(loc=5., scale=2., low=low))
    elif kind == 'decreasing':
        prior.set_custom_prior('r_eff_2', dist.TransformedDistribution(dist.Uniform(), dist.transforms.AffineTransform(low + 8., -8.)))
    else:
        prior.set_custom_prior('r_eff_2', dist.TransformedDistribution(dist.Normal(1., 0.5), [dist.transforms.ExpTransform(), dist.transforms.AffineTransform(low, 1.)]))
    actual_low, _ = priors.get_scaled_prior_bounds(prior.dist_dict['r_eff_2'])
    assert float(actual_low) == pytest.approx(low, abs=1e-6)
    assert prior.check_vars()
    for seed in (0, 12, 314):
        draws = infer.Predictive(prior, num_samples=1024)(jax.random.PRNGKey(seed))
        disc, bulge, u = [np.asarray(draws[name]) for name in ('r_eff_2', 'r_eff_1', 'u_1')]
        assert np.all(bulge >= actual_low)
        assert np.count_nonzero(bulge > disc) == 0
        np.testing.assert_allclose(bulge, actual_low + u * (disc - actual_low), atol=1e-6, rtol=1e-6)
        assert u.mean() == pytest.approx(0.5, abs=0.04)
        assert 's_1' not in draws


@pytest.mark.parametrize('radius', [0.5, 0.5000001, 3.])
@pytest.mark.parametrize('u', [0., 1e-7, 0.5, 1.])
def test_boundaries_and_gradients(make_fitter, radius, u):
    prior = make_fitter().prior
    def bulge(radius, u):
        return handlers.condition(handlers.seed(prior, 3), {'r_eff_2': radius, 'u_1': u})()['r_eff_1']
    result = jax.jit(bulge)(radius, u)
    assert 0.5 <= result <= radius
    assert np.isfinite(jax.grad(bulge, argnums=(0, 1))(radius, u)).all()


def test_custom_u_and_suffixes(make_fitter):
    prior = make_fitter(suffix='_1').prior
    prior.set_custom_prior('u_1', dist.Beta(2., 5.))
    for suffix in ('_F444W', '_F200W', ''):
        prior = priors.update_prior_suffix(prior, suffix)
        assert prior.check_vars()
        draws = infer.Predictive(prior, num_samples=1024)(jax.random.PRNGKey(20))
        assert np.asarray(draws['u_1' + suffix]).mean() == pytest.approx(2. / 7., abs=0.03)
        assert np.all(draws['r_eff_1' + suffix] >= 0.5)
        assert 'r_eff_1' + suffix not in prior.dist_dict


def test_invalid_configs(make_fitter):
    prior = make_fitter().prior
    for distribution in (dist.Normal(3., 1.), dist.Uniform(-1., 6.)):
        with pytest.raises(ValueError, match='lower support'):
            prior.set_custom_prior('r_eff_2', distribution)
    with pytest.raises(ValueError, match='inside'):
        prior.set_uniform_prior('u_1', -0.1, 1.)
    with pytest.raises(ValueError, match='independent'):
        prior.set_uniform_prior('r_eff_1', 0.5, 3.)
    with pytest.raises(ValueError, match='multisource'):
        priors.PySersicMultiPrior({'type': ['scaled_sersic_exp']})
    prior.dist_dict.pop('n')
    with pytest.raises(ValueError, match='Incomplete'):
        handlers.seed(prior, 0)()


def test_autoprior(make_fitter):
    fitter = make_fitter()
    prior = priors.SourceProperties(fitter.data).generate_prior('scaled_sersic_exp', suffix='_F444W')
    assert prior.check_vars()
    assert 'n_F444W' in prior.dist_dict
    sites = handlers.trace(handlers.seed(prior, 3)).get_trace()
    assert sites['r_eff_1_F444W']['type'] == 'deterministic'


@pytest.mark.parametrize('cls', [rendering.PixelRenderer, rendering.MoGFourierRenderer, rendering.EmulatorFourierRenderer, rendering.HybridRenderer])
@pytest.mark.parametrize('u', [0.1, 0.5, 0.9])
def test_render_equivalence(make_fitter, cls, u):
    fitter = make_fitter()
    params = handlers.condition(handlers.seed(fitter.prior, 4), {'u_1': u})()
    renderer = cls((16, 16), fitter.psf)
    ordinary = renderer.render_source(params, 'sersic_exp')
    scaled = renderer.render_source(params, 'scaled_sersic_exp')
    np.testing.assert_array_equal(ordinary, scaled)


@pytest.mark.parametrize('cls', [FitMultiBandPoly, FitMultiBandBSpline])
@pytest.mark.parametrize('disc_mode', ['unlinked', 'linked', 'constant'])
@pytest.mark.parametrize('u_mode', ['unlinked', 'linked', 'constant'])
def test_multiband_configurations(make_fitter, cls, disc_mode, u_mode):
    lows = [0.8, 0.3] if disc_mode == 'constant' else [0.3, 0.8]
    linked = []
    const = []
    for name, mode in [('r_eff_2', disc_mode), ('u_1', u_mode)]:
        if mode == 'linked':
            linked.append(name)
        elif mode == 'constant':
            const.append(name)
    mb = cls([make_fitter(low=low) for low in lows], [1., 2.], linked,
             const_params=const, band_names=['a', 'b'], wv_to_save=np.array([1., 1.5, 2.]))
    model = mb.build_model(return_model=True)
    draws = infer.Predictive(model, num_samples=64)(jax.random.PRNGKey(10))
    for band, low in zip(mb.band_names, lows):
        disc, bulge, u = [np.asarray(draws[f'{name}_{band}']) for name in ('r_eff_2', 'r_eff_1', 'u_1')]
        assert np.all(bulge >= low)
        assert np.all(bulge <= disc)
        np.testing.assert_allclose(bulge, low + u * (disc - low), atol=1e-6)
        if u_mode == 'constant':
            np.testing.assert_array_equal(u, draws['u_1'])
    assert np.isfinite(draws['model']).all()
    sites = handlers.trace(handlers.seed(model, 11)).get_trace()
    for i, (band, fitter) in enumerate(zip(mb.band_names, mb.fitter_list)):
        params = {}
        for name in priors.base_profile_params['scaled_sersic_exp']:
            params[name] = sites[f'{name}_{band}']['value'] if f'{name}_{band}' in sites else sites[name]['value']
        params['r_eff_1'] = sites[f'r_eff_1_{band}']['value']
        np.testing.assert_allclose(sites['model']['value'][i], fitter.renderer.render_source(params, 'sersic_exp'), atol=1e-6)


@pytest.mark.parametrize('cls', [FitMultiBandPoly, FitMultiBandBSpline])
def test_multiband_ranges_and_saved_curve(make_fitter, cls):
    fitters = [make_fitter(), make_fitter()]
    mb = cls(fitters, [1., 2.], ['r_eff_2', 'u_1'], band_names=['a', 'b'],
             linked_params_range={'u_1': [0.2, 0.7]}, wv_to_save=np.array([1., 1.5, 2.]))
    draws = infer.Predictive(mb.build_model(), num_samples=32)(jax.random.PRNGKey(12))
    bulge_at_wv = 0.5 + draws['u_1_at_wv'] * (draws['r_eff_2_at_wv'] - 0.5)
    assert np.all(bulge_at_wv >= 0.5)
    assert np.all(bulge_at_wv <= draws['r_eff_2_at_wv'])
    np.testing.assert_allclose(bulge_at_wv[:, 0], draws['r_eff_1_a'], atol=1e-6)
    np.testing.assert_allclose(bulge_at_wv[:, -1], draws['r_eff_1_b'], atol=1e-6)
    assert np.all(draws['u_1_at_wv'] >= 0.2 - 1e-6)
    assert np.all(draws['u_1_at_wv'] <= 0.7 + 1e-6)
    for fitter in fitters:
        fitter.prior.set_truncated_gaussian_prior('r_eff_2', 3., 1., low=0.5)
    with pytest.raises(ValueError, match='finite range'):
        cls(fitters, [1., 2.], ['r_eff_2'], wv_to_save=np.array([1., 2.]))
    cls(fitters, [1., 2.], ['r_eff_2'], linked_params_range={'r_eff_2': [0.5, 6.]}, wv_to_save=np.array([1., 2.]))
    with pytest.raises(ValueError, match='incompatible'):
        cls([make_fitter(low=0.3), make_fitter(low=0.8)], [1., 2.], [], const_params=['r_eff_2'], wv_to_save=np.array([1., 2.]))


@pytest.mark.parametrize('linked', [[], ['u_1'], ['u_1', 'r_eff_2']])
def test_cached_prior_rescaling(make_fitter, monkeypatch, linked):
    import arviz
    import pandas as pd
    fitters = [make_fitter(low=0.3), make_fitter(low=0.8)]
    summary = pd.DataFrame({'mean': [3., 0.5], 'sd': [0.2, 0.1]}, index=['r_eff_2', 'u_1'])
    for fitter in fitters:
        fitter.svi_results = SimpleNamespace(idata=None, prior=SimpleNamespace(suffix=fitter.prior.suffix))
    monkeypatch.setattr(arviz, 'summary', lambda *args, **kwargs: summary)
    mb = FitMultiBandPoly(fitters, [1., 2.], linked, band_names=['a', 'b'],
                         rescale_unlinked_priors=True, wv_to_save=np.array([1., 2.]))
    for fitter in mb.fitter_list:
        assert 'Uniform' in fitter.prior.repr_dict['u_1']
    draws = infer.Predictive(mb.build_model(), num_samples=32)(jax.random.PRNGKey(5))
    for band, low in [('a', 0.3), ('b', 0.8)]:
        assert np.all(draws[f'r_eff_1_{band}'] >= low - 1e-6)


@pytest.mark.parametrize('method', ['map', 'nuts', 'svi'])
def test_single_inference(make_fitter, method):
    fitter = make_fitter()
    model = fitter.build_model()
    if method == 'map':
        draws = fitter.find_MAP(jax.random.PRNGKey(9))
        assert np.sqrt(np.mean(((draws['model'] - fitter.data) / fitter.rms)**2)) < 2.
    elif method == 'nuts':
        mcmc = infer.MCMC(infer.NUTS(model, max_tree_depth=6), num_warmup=100, num_samples=64, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(10))
        draws = mcmc.get_samples()
        assert not np.any(mcmc.get_extra_fields()['diverging'])
    else:
        guide = infer.autoguide.AutoLowRankMultivariateNormal(model, init_scale=0.05)
        svi = infer.SVI(model, guide, optim.Adam(0.02), infer.TraceMeanField_ELBO(5))
        result = svi.run(jax.random.PRNGKey(11), 250, progress_bar=False)
        assert np.isfinite(result.losses).all()
        draws = guide.sample_posterior(jax.random.PRNGKey(12), result.params, sample_shape=(64,))
    assert np.all(draws['r_eff_1'] >= 0.5)
    assert np.all(draws['r_eff_1'] <= draws['r_eff_2'])
    assert np.isfinite(draws['model']).all()
    np.testing.assert_allclose(draws['r_eff_1'], 0.5 + draws['u_1'] * (draws['r_eff_2'] - 0.5), atol=2e-5)


@pytest.mark.parametrize('cls', [FitMultiBandPoly, FitMultiBandBSpline])
def test_multiband_nuts(make_fitter, cls):
    mb = cls([make_fitter(low=0.3), make_fitter(low=0.8)], [1., 2.], ['r_eff_2', 'u_1'],
             const_params=['xc', 'yc'], band_names=['a', 'b'], wv_to_save=np.array([1., 2.]))
    sampler = infer.MCMC(infer.NUTS(mb.build_model(), max_tree_depth=5), num_warmup=50, num_samples=32, num_chains=1, progress_bar=False)
    sampler.run(jax.random.PRNGKey(4))
    samples = sampler.get_samples()
    for band, low in [('a', 0.3), ('b', 0.8)]:
        assert np.all(samples[f'r_eff_1_{band}'] >= low)
        assert np.all(samples[f'r_eff_1_{band}'] < samples[f'r_eff_2_{band}'])
    assert np.isfinite(sampler.last_state.potential_energy)
