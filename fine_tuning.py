#!/usr/bin/env python
# -------------------------------
#  Translates
# -------------------------------
import pandas as pd
import numpy as np
from sympy import S
from sympy.solvers import solve
import amplitudes

# -------------------------------
import generate_plots as gp

# -------------------------------
import logging

# -------------------------------
logger = logging.getLogger("fine_tuning")
logger.setLevel(logging.WARNING)
_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(_format)

logger.addHandler(_ch)

# -------------------------------

nucleons = ["deut", "h3", "he3"]

pars = gp.parameters(normalized=True)

# -------------------------------


def get_coefficient(ops, r):
    "Compute the coefficient such that amp_lo = r (amp_nlo - amp_lo)"
    op_lo, op_nlo = get_amplitudes(ops, S("g"))

    val_lo = op_lo ** 2
    val_nlo = op_nlo ** 2
    val = val_lo - r * (val_nlo - val_lo)

    sols = np.array(solve(val, S("g")), dtype=np.float128)

    max_val = -10
    for new_sol in sols:
        new_max = val_lo.subs({"g": new_sol})
        if new_max > max_val:
            sol = new_sol

    return np.float128(sol)


# -------------------------------


def get_amplitudes(ops, coefficient):
    "Compute the lo and nlo amplitudes with OP_q + OP_g * coefficient"
    quark_op_lo = ops["scalar"] + ops["tau3"]
    gluon_op_lo = ops["scalar-g"]

    quark_op_nlo = quark_op_lo + ops["scalar_q2"] + ops["2-pion-q"]
    gluon_op_nlo = gluon_op_lo + ops["2-pion-g"]

    gluon_op_lo *= coefficient
    gluon_op_nlo *= coefficient

    amp_lo = quark_op_lo + gluon_op_lo
    amp_nlo = quark_op_nlo + gluon_op_nlo

    return amp_lo, amp_nlo


# -------------------------------
chiral_amplitudes = {}
amps_lo = {}
amps_nlo = {}
for nucleon in nucleons:
    ops = {}
    all_ops = gp.get_operators(nucleon, parameters=pars)
    for name, op in all_ops.items():
        logger.info("Wave used for ratio bench-mark:" + op.amp_frame.loc[0, "wave"])
        # just take the part essential for benching
        ops[name] = op.frame.loc[0, "mat"]

    t_amplitudes = pd.DataFrame()
    for r in np.linspace(1, 10, 10).astype(int):
        coefficient = get_coefficient(ops, r)
        amp_lo, amp_nlo = get_amplitudes(all_ops, coefficient)

        amps_lo[(nucleon, r)] = amp_lo
        amps_nlo[(nucleon, r)] = amp_nlo

        chiral_amp = amplitudes.ChiralErrorEstimate(amp_lo.get_chiral_frame(False))
        chiral_amp = chiral_amp.frame.groupby(["q", "Order"])[["amp", "d_amp"]].mean()

        nlo_c_amp = amplitudes.ChiralErrorEstimate(amp_nlo.get_chiral_frame(False))
        nlo_c_amp = nlo_c_amp.frame.groupby(["q", "Order"])[["amp", "d_amp"]].mean()

        chiral_amp.loc[:, "amp_nlo"] = nlo_c_amp.loc[:, "amp"]
        chiral_amp.loc[:, "d_amp_nlo"] = nlo_c_amp.loc[:, "d_amp"]

        chiral_amp.reset_index(inplace=True)

        # interpolate
        new_q = list(np.linspace(0, 200, 40)) * 5
        new_order = (
            ["n0lo"] * 40
            + ["n1lo"] * 40
            + ["n2lo"] * 40
            + ["n3lo"] * 40
            + ["n4lo"] * 40
        )

        intp_frame = pd.DataFrame(zip(new_q, new_order), columns=["q", "Order"])

        chiral_amp = pd.concat([chiral_amp, intp_frame], ignore_index=True)
        chiral_amp = chiral_amp.drop_duplicates(subset=["q", "Order"], keep="first")
        chiral_amp.sort_values(["Order", "q"], inplace=True)
        chiral_amp = chiral_amp.reset_index().drop("index", axis=1)
        chiral_amp = chiral_amp.interpolate()

        chiral_amp.loc[:, "amp_diff"] = np.abs(
            chiral_amp.loc[:, "amp"] - chiral_amp.loc[:, "amp_nlo"]
        )
        chiral_amp.loc[:, "err_max"] = np.abs(
            chiral_amp.loc[:, "d_amp"] + chiral_amp.loc[:, "d_amp_nlo"]
        )
        chiral_amp.loc[:, "not_overlap"] = (
            chiral_amp.loc[:, "err_max"] < chiral_amp.loc[:, "amp_diff"]
        )

        chiral_amp.loc[:, "r"] = r

        t_amplitudes = t_amplitudes.append(chiral_amp, ignore_index=True)

    chiral_amplitudes[nucleon] = t_amplitudes
