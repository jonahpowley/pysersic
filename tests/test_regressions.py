from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest
from astropy.convolution import Gaussian2DKernel
from numpyro import handlers

from pysersic import FitSingle, priors, rendering
from pysersic.multiband import FitMultiBandBSpline, FitMultiBandPoly


@pytest.fixture
def point_source_fitters():
    """Create two small single-band fits with a flat sky."""
    psf = Gaussian2DKernel(0.8, x_size=5, y_size=5).array
    renderer = rendering.MoGFourierRenderer((12, 12), psf)
    data = renderer.render_source(dict(xc=5.5, yc=5.5, flux=100.), 'pointsource')
    fitters = []
    for _ in range(2):
        prior = priors.PySersicSourcePrior('pointsource', sky_type='flat', sky_guess=0., sky_guess_err=0.1)
        prior.set_gaussian_prior('xc', 5.5, 0.2)
        prior.set_gaussian_prior('yc', 5.5, 0.2)
        prior.set_uniform_prior('flux', 20., 300.)
        fitters.append(FitSingle(data, np.full(data.shape, 0.1), psf, prior, renderer=rendering.MoGFourierRenderer))
    return fitters


@pytest.mark.parametrize('fitter_class', [FitMultiBandPoly, FitMultiBandBSpline])
@pytest.mark.parametrize('bounded', [False, True])
def test_optional_saved_wavelengths(point_source_fitters, fitter_class, bounded):
    ranges = {'flux': [20., 300.]} if bounded else {}
    kwargs = dict(fitter_list=point_source_fitters, wavelengths=[1., 2.], linked_params=['flux'],
                  const_params=['xc'], band_names=['a', 'b'], linked_params_range=ranges)
    without_grid = fitter_class(**kwargs)
    with_grid = fitter_class(**kwargs, wv_to_save=np.array([1., 1.5, 2.]))
    first = handlers.trace(handlers.seed(without_grid.build_model(return_model=True), 4)).get_trace()
    second = handlers.trace(handlers.seed(with_grid.build_model(return_model=True), 4)).get_trace()

    assert 'flux_at_wv' not in first
    assert second['flux_at_wv']['value'].shape == (3,)
    assert set(second) - set(first) == {'flux_at_wv'}
    for name, site in first.items():
        np.testing.assert_array_equal(site['value'], second[name]['value'])
    assert np.isfinite(first['model']['value']).all()


@pytest.mark.parametrize('fitter_class', [FitMultiBandPoly, FitMultiBandBSpline])
@pytest.mark.parametrize('case', ['cached', 'cached_suffix', 'fresh'])
def test_prior_rescaling(point_source_fitters, fitter_class, case, monkeypatch):
    import arviz

    band_names = ['back', 'x_sl']
    keys_used = []

    def result_for_band(band_index, suffix):
        names = [name + suffix for name in ('xc', 'yc', 'flux', 'sky_back')]
        summary = pd.DataFrame({'mean': [5.5, 5.5, 100. + 20. * band_index, 0.25],
                                'sd': [0.05, 0.05, 2., 0.05]}, index=names)
        return SimpleNamespace(idata=summary, prior=SimpleNamespace(suffix=suffix))

    def estimate_posterior(fitter, rkey):
        keys_used.append(np.asarray(jax.random.key_data(rkey)))
        band_index = band_names.index(fitter.prior.suffix.removeprefix('_'))
        return result_for_band(band_index, fitter.prior.suffix)

    monkeypatch.setattr(FitSingle, 'estimate_posterior', estimate_posterior)
    monkeypatch.setattr(arviz, 'summary', lambda idata, **kwargs: idata)
    if case != 'fresh':
        suffix = '_old' if case == 'cached_suffix' else ''
        for band_index, fitter in enumerate(point_source_fitters):
            fitter.svi_results = result_for_band(band_index, suffix)

    kwargs = dict(fitter_list=point_source_fitters, wavelengths=[1., 2.], linked_params=[],
                  band_names=band_names, rescale_unlinked_priors=True)
    multi = fitter_class(**kwargs)
    for band_index, fitter in enumerate(multi.fitter_list):
        suffix = fitter.prior.suffix
        flux_prior = fitter.prior.dist_dict['flux' + suffix]
        draws = np.asarray(flux_prior.sample(jax.random.PRNGKey(6), sample_shape=(4096,)))
        assert draws.mean() == pytest.approx(100. + 20. * band_index, abs=0.3)
        assert draws.std() == pytest.approx(4., abs=0.2)
        support = flux_prior.base_dist.support
        low, high = support.lower_bound, support.upper_bound
        for transform in flux_prior.transforms:
            low, high = transform(low), transform(high)
        assert float(low) == pytest.approx(20.)
        assert float(high) == pytest.approx(300.)
        assert set(fitter.prior.sky_prior.dist_dict) == {'sky_back' + suffix}
        sky_prior = fitter.prior.sky_prior.dist_dict['sky_back' + suffix]
        assert float(sky_prior.transforms[0](sky_prior.base_dist.mean)) == pytest.approx(0.25)

    sites = handlers.trace(handlers.seed(multi.build_model(return_model=True), 5)).get_trace()
    assert np.isfinite(sites['model']['value']).all()
    if case == 'fresh':
        fitter_class(**kwargs)
        assert len(keys_used) == 4
        assert not np.array_equal(keys_used[0], keys_used[1])
        np.testing.assert_array_equal(keys_used[:2], keys_used[2:])
    else:
        assert not keys_used
    original_prior = point_source_fitters[0].prior.dist_dict['flux']
    assert float(original_prior.transforms[0](original_prior.base_dist.mean)) == pytest.approx(160.)


@pytest.mark.parametrize('low,high,valid', [
    (0.65, 8., True), (0.6499999999999999, 8., True),
    (0.65 - 1e-7, 8. + 1e-7, True), (0.64, 8., False), (0.65, 8.01, False),
])
def test_renderer_index_bounds(point_source_fitters, low, high, valid):
    fitter = point_source_fitters[0]
    prior = priors.PySersicSourcePrior('sersic')
    prior.set_uniform_prior('n', low, high)
    if valid:
        FitSingle(fitter.data, fitter.rms, fitter.psf, prior, renderer=rendering.MoGFourierRenderer)
    else:
        with pytest.raises(AssertionError, match='outside the bounds'):
            FitSingle(fitter.data, fitter.rms, fitter.psf, prior, renderer=rendering.MoGFourierRenderer)


@pytest.mark.parametrize('profile', ['sersic_exp', 'doublesersic'])
def test_generated_bulge_index_bounds(point_source_fitters, profile):
    fitter = point_source_fitters[0]
    prior = priors.SourceProperties(fitter.data).generate_prior(profile)
    FitSingle(fitter.data, fitter.rms, fitter.psf, prior, renderer=rendering.MoGFourierRenderer)
