import numpy as np

from redback_csm.spline_mle import (
    SplineMLEProblem,
    default_spline_bounds,
    make_random_starts,
    spline_parameter_names,
)


def test_spline_mle_problem_with_fake_model():
    time = np.array([1.0, 2.0, 3.0])
    luminosity = np.array([2.0, 4.0, 6.0])
    error = np.ones_like(time)
    names = spline_parameter_names(
        n_nodes=3,
        profile="static",
        include_time_offset=False,
        include_nickel=False,
    )
    bounds = default_spline_bounds(
        n_nodes=3,
        profile="static",
        include_time_offset=False,
        include_nickel=False,
    )
    bounds["eff"] = (0.01, 5.0)

    def model_function(t, params):
        return params["eff"] * t

    start = {name: 0.5 * (bounds[name][0] + bounds[name][1]) for name in names}
    start.update(
        {
            "log_r_inner": 13.0,
            "log_r_outer": 15.0,
            "log_rho_0": -15.0,
            "log_rho_1": -15.0,
            "log_rho_2": -15.0,
            "delta_sn": 1.0,
            "nn_sn": 10.0,
            "mej_sn": 5.0,
            "esn": 1.0,
            "eff": 1.0,
        }
    )
    starts = make_random_starts(start, bounds, names, n_random=0)
    problem = SplineMLEProblem(
        time=time,
        luminosity=luminosity,
        error=error,
        parameter_names=names,
        bounds=bounds,
        model_function=model_function,
        profile="static",
    )
    result = problem.fit(starts, max_nfev=20)

    assert np.isclose(result.parameters["eff"], 2.0, rtol=1.0e-5)
    assert result.chi2 < 1.0e-10
