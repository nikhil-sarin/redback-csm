"""
prior_provider.py — Callable registered under the redback.model.priors entry point.

redback calls get_prior(model_name) when building priors for any model registered
by this plugin. Returns a bilby PriorDict or None if the model is not known here.
"""

import inspect
import os
import re

from bilby.core.prior import LogUniform, PriorDict, Uniform

PRIOR_DIR = os.path.join(os.path.dirname(__file__), "priors")


def _uniform(name, minimum, maximum, label):
    return Uniform(minimum=minimum, maximum=maximum, name=name, latex_label=label)


def _loguniform(name, minimum, maximum, label):
    return LogUniform(minimum=minimum, maximum=maximum, name=name, latex_label=label)


def _latex_label(name):
    labels = {
        "redshift": "$z$",
        "mdot": "$\\dot{M}~(M_\\odot/{\\mathrm{yr}})$",
        "mdot_0": "$\\dot{M}_0~(M_\\odot/{\\mathrm{yr}})$",
        "mdot_1": "$\\dot{M}_1~(M_\\odot/{\\mathrm{yr}})$",
        "mdot_2": "$\\dot{M}_2~(M_\\odot/{\\mathrm{yr}})$",
        "mdot_baseline": "$\\dot{M}_{\\mathrm{base}}~(M_\\odot/{\\mathrm{yr}})$",
        "mdot_peak": "$\\dot{M}_{\\mathrm{peak}}~(M_\\odot/{\\mathrm{yr}})$",
        "vwind": "$v_{\\mathrm{wind}}~({\\mathrm{km/s}})$",
        "mexp": "$M_{\\mathrm{exp}}~(M_\\odot)$",
        "mexp_out": "$M_{\\mathrm{exp,out}}~(M_\\odot)$",
        "mej_sn": "$M_{\\mathrm{ej,SN}}~(M_\\odot)$",
        "eexp": "$E_{\\mathrm{exp}}~({\\mathrm{foe}})$",
        "eexp_out": "$E_{\\mathrm{exp,out}}~({\\mathrm{foe}})$",
        "esn": "$E_{\\mathrm{SN}}~({\\mathrm{foe}})$",
        "eff": "$\\epsilon$",
        "f_nickel": "$f_{\\mathrm{Ni}}$",
        "kappa": "$\\kappa~({\\mathrm{cm}}^2/{\\mathrm{g}})$",
        "kappa_gamma": "$\\kappa_\\gamma$",
        "temperature_floor": "$T_{\\mathrm{floor}}~({\\mathrm{K}})$",
        "logepsb": "$\\log_{10}\\epsilon_B$",
        "logepse": "$\\log_{10}\\epsilon_e$",
        "logepsx": "$\\log_{10}\\epsilon_X$",
        "p": "$p$",
        "base_density": "$\\rho_{\\mathrm{base}}$",
        "base_index": "$s$",
        "eta": "$\\eta$",
        "m_csm": "$M_{\\mathrm{CSM}}~(M_\\odot)$",
        "r_inner": "$r_{\\mathrm{in}}~({\\mathrm{cm}})$",
        "r_outer": "$r_{\\mathrm{out}}~({\\mathrm{cm}})$",
        "log_r_inner": "$\\log_{10} r_{\\mathrm{in}}$",
        "log_r_outer": "$\\log_{10} r_{\\mathrm{out}}$",
        "interval": "$\\Delta t~({\\mathrm{days}})$",
        "interval_sn": "$\\Delta t_{\\mathrm{SN}}~({\\mathrm{days}})$",
        "t1": "$t_1~({\\mathrm{yr}})$",
        "t2": "$t_2~({\\mathrm{yr}})$",
        "t_peak": "$t_{\\mathrm{peak}}~({\\mathrm{yr}})$",
        "t_width": "$t_{\\mathrm{width}}~({\\mathrm{yr}})$",
        "t_break1": "$t_{\\mathrm{break},1}~({\\mathrm{yr}})$",
        "t_break2": "$t_{\\mathrm{break},2}~({\\mathrm{yr}})$",
        "alpha1": "$\\alpha_1$",
        "alpha2": "$\\alpha_2$",
        "alpha3": "$\\alpha_3$",
        "vej_max_ratio": "$v_{\\mathrm{max}}/v_t$",
        "delta": "$\\delta$",
        "delta_sn": "$\\delta_{\\mathrm{SN}}$",
        "delta_out": "$\\delta_{\\mathrm{out}}$",
        "nn": "$n$",
        "nn_sn": "$n_{\\mathrm{SN}}$",
        "nn_out": "$n_{\\mathrm{out}}$",
    }
    if name in labels:
        return labels[name]
    match = re.match(r"log_rho_(\d+)$", name)
    if match:
        idx = match.group(1)
        return f"$\\log_{{10}}\\rho_{{{idx}}}$"
    match = re.match(r"shell(\d+)_(radius|width|density)$", name)
    if match:
        idx, quantity = match.groups()
        if quantity == "radius":
            return f"$r_{{{idx}}}~({{\\mathrm{{AU}}}})$"
        if quantity == "width":
            return f"$w_{{{idx}}}~({{\\mathrm{{AU}}}})$"
        return f"$\\rho_{{{idx}}}$"
    return f"${name}$"


def _prior_for_parameter(name):
    label = _latex_label(name)
    if name == "redshift":
        return _uniform(name, 1e-3, 3.0, label)
    if name in {"mdot", "mdot_0", "mdot_1", "mdot_2", "mdot_baseline", "mdot_peak"}:
        return _loguniform(name, 1e-5, 10.0, label)
    if name == "vwind":
        return _loguniform(name, 10.0, 1000.0, label)
    if name in {"mexp", "mexp_out", "mej_sn"}:
        return _loguniform(name, 0.1, 50.0, label)
    if name in {"eexp", "eexp_out", "esn"}:
        return _loguniform(name, 1e-3, 100.0, label)
    if name in {"delta", "delta_sn", "delta_out"}:
        return _uniform(name, 0.0, 3.0, label)
    if name in {"nn", "nn_sn", "nn_out"}:
        return _uniform(name, 6.0, 14.0, label)
    if name == "eff":
        return _uniform(name, 0.01, 1.0, label)
    if name == "f_nickel":
        return _loguniform(name, 1e-3, 1.0, label)
    if name == "kappa":
        return _uniform(name, 0.05, 2.0, label)
    if name == "kappa_gamma":
        return _loguniform(name, 1e-4, 1e4, label)
    if name == "temperature_floor":
        return _loguniform(name, 100.0, 1e4, label)
    if name in {"logepsb", "logepse", "logepsx"}:
        return _uniform(name, -4.0, 0.0, label)
    if name == "p":
        return _uniform(name, 2.0, 4.0, label)
    if name in {"base_density"} or re.match(r"shell\d+_density$", name):
        return _loguniform(name, 1e-20, 1e-12, label)
    if name in {"base_index", "eta"}:
        return _uniform(name, -4.0, 0.0, label)
    if name == "m_csm":
        return _loguniform(name, 1e-3, 100.0, label)
    if name in {"r_inner", "r_outer"}:
        return _loguniform(name, 1e12, 1e18, label)
    if name == "log_r_inner":
        return _uniform(name, 12.0, 15.5, label)
    if name == "log_r_outer":
        return _uniform(name, 14.5, 18.0, label)
    if re.match(r"log_rho_\d+$", name):
        return _uniform(name, -22.0, -10.0, label)
    if name in {"interval", "interval_sn"}:
        return _loguniform(name, 1.0, 10000.0, label)
    if name in {"t1", "t2", "t_peak", "t_width", "t_break1", "t_break2"}:
        return _loguniform(name, 1e-2, 10000.0, label)
    if name in {"alpha1", "alpha2", "alpha3"}:
        return _uniform(name, -8.0, 8.0, label)
    if re.match(r"shell\d+_(radius|width)$", name):
        return _loguniform(name, 0.1, 10000.0, label)
    if name == "vej_max_ratio":
        return _uniform(name, 3.0, 10.0, label)
    raise KeyError(f"No generated prior template for parameter '{name}'")


def _model_parameters(model_name):
    import redback_csm.models as models

    if not hasattr(models, model_name):
        return None
    sig = inspect.signature(getattr(models, model_name))
    names = []
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.name in {"time"}:
            continue
        names.append(param.name)
    return names


def _generated_prior(model_name):
    names = _model_parameters(model_name)
    if names is None:
        return None
    if model_name == "csm_xray":
        return None

    priors = PriorDict()
    for name in names:
        priors[name] = _prior_for_parameter(name)

    if "nickel" in model_name:
        for extra in ("kappa", "kappa_gamma"):
            if extra not in priors:
                priors[extra] = _prior_for_parameter(extra)

    is_multiband = not (
        model_name.endswith("_bolometric")
        or model_name.endswith("_radio")
        or model_name.endswith("_xray")
    )
    if is_multiband and "redshift" in priors and "temperature_floor" not in priors:
        priors["temperature_floor"] = _prior_for_parameter("temperature_floor")

    has_bpl_ejecta = any(name in priors for name in ("delta", "delta_sn", "delta_out"))
    if has_bpl_ejecta and "vej_max_ratio" not in priors:
        priors["vej_max_ratio"] = _prior_for_parameter("vej_max_ratio")

    return priors


def get_prior(model_name: str):
    """
    Return a bilby PriorDict for the given model name, or None if not found.

    Used by redback.priors.get_priors() via the redback.model.priors entry point.
    """
    generated = _generated_prior(model_name)
    if generated is not None:
        return generated

    path = os.path.join(PRIOR_DIR, f"{model_name}.prior")
    if not os.path.exists(path):
        return None
    priors = PriorDict()
    priors.from_file(path)
    return priors
