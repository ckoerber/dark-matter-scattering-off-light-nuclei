#!/usr/bin/env python
# -------------------------------
#  Generates the plots for the paper
# -------------------------------
import os
import copy
import logging
import argparse

# -------------------------------
import amplitudes as amp
from sympy import S
import pandas as pd
import numpy as np

# -------------------------------
import seaborn as sns
import matplotlib.pylab as plt

sns.set(font_scale=1.3, style="ticks", font="Source Sans Pro")
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
# -------------------------------

logger = logging.getLogger("generate_plots")
logger.setLevel(logging.INFO)
_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(_format)

_sh = logging.FileHandler("generate_plots.log", mode="a")
_sh.setLevel(logging.DEBUG)
_sh.setFormatter(_format)

logger.addHandler(_ch)
logger.addHandler(_sh)
# -------------------------------


parser = argparse.ArgumentParser(description="Plotting tool for DM amplitudes.")
parser.add_argument(
    "--sonly",
    help="Specifies if plot shall only contain pure S-waves.",
    action="store_true",
)
parser.add_argument(
    "--non-normalized",
    help="Specifies if plot non-normalized results.",
    action="store_true",
)
parser.add_argument(
    "--out-folder",
    type=str,
    help='Specify the output folder. If not specified: "results"',
)
parser.add_argument(
    "--estimate-type",
    type=int,
    help="""Specify the chiral truncation error estimate type:
    0 -> Evgeny et. al. without postdictions\n
    1 -> Evgeny et. al. with postdictions\n
    2 -> All orders with postdictions\n
  """,
)
_estimate_type_dict = {
    0: "Evgeny et. al. without postdictions",
    1: "Evgeny et. al. with postdictions",
    2: "All orders with postdictions",
}
# -------------------------------

HOME = os.path.dirname(__file__)
DM_FOLDER = os.path.join(HOME, "data")


# -------------------------------
class parameters(object):
    def __init__(self, normalized=True, self_normalized=False, isoscalar_only=False):
        self.normalized = normalized
        self.self_normalized = self_normalized
        self.nuc_sysetms = ["deut", "h3", "he3"]
        self.files = [
            ("scalar_QMAX1.00_NQ11_NX30.dat", "scalar", 0),
            ("scalar_QMAX1.00_NQ11_NX30.dat", "scalar-g", 0),
            ("tau3_QMAX1.00_NQ11_NX30.dat", "tau3", 0),
            ("scalar_q2_QMAX1.00_NQ11_NX30.dat", "scalar_q2", 0),
            # two-pion deut
            ("2-pion-QMAX1.00_NQ11_NP48.dat", "2-pion-q", 0),
            # two-pion 3B
            ("2-pion_QMAX1.00_NQ11_NP48.dat", "2-pion-q", 0),
            # two-pion deut gluon full and quark kpart)
            ("2-pion-QMAX1.00_NQ11_NP48.dat", "2-pion-g", 1),
            ("2-pion-QMAX1.00_NQ11_NP48.dat", "2-pion-gq", 0),
            # two-pion 3B
            ("2-pion-g_QMAX1.00_NQ11_NP48.dat", "2-pion-g", 0),
            ("2-pion_QMAX1.00_NQ11_NP48.dat", "2-pion-gq", 0),
        ]
        self.facts = {
            "scalar": amp.operator_factors["1-scalar"],
            "tau3": amp.operator_factors["1-tau3"],
            "scalar_q2": amp.operator_factors["1-scalar-q2"],
            "2-pion-q": amp.operator_factors["2-pion-q"],
            "scalar-g": amp.operator_factors["1-scalar-g"],
            "2-pion-g": amp.operator_factors["2-pion-g"],
            "2-pion-gq": 16
            * S("pi")
            / 9
            * 2
            * S("mpi") ** 2
            * S("c_g")
            / (2 * S("pi")) ** 3
            / 2,
        }
        if isoscalar_only:
            self.facts["tau3"] = S("0")
        if self_normalized:
            # scalar quark
            normalizer = self.facts["scalar"]
            self.facts["scalar"] /= normalizer
            self.facts["scalar_q2"] /= normalizer
            self.facts["2-pion-q"] /= normalizer
            # isovector quark
            normalizer = self.facts["tau3"]
            if not (isoscalar_only):
                self.facts["tau3"] /= normalizer
            # gluon
            normalizer = self.facts["scalar-g"]
            self.facts["scalar-g"] /= normalizer
            self.facts["2-pion-g"] /= normalizer
            self.facts["2-pion-gq"] /= normalizer
        elif normalized:
            for key, val in self.facts.items():
                self.facts[key] = val / (S("sigma_pi_N"))


par = parameters(True)

inch_max_w = 8.0 * 2.0


def get_operators(nucleon, s_only=False, ci=(1, 1, 1), parameters=par):
    """
    Gets all the operators for a given file
    """
    operators = {}
    folder = os.path.join(DM_FOLDER, nucleon + "_out")
    logger.info("Looking up folder: " + folder)

    if parameters.normalized or parameters.self_normalized:
        if nucleon == "deut":
            A = 2
        elif nucleon == "he3":
            A = 3
        elif nucleon == "h3":
            A = 3
        else:
            raise ValueError("nucleon: {n} not implemented".format(n=nucleon))
    else:
        A = 1

    facts = {}

    # Quark operators
    for (file_name, kind, ind) in par.files:
        if s_only:
            if kind == "2-pion-q":
                file_name = file_name.replace(".dat", "_Sonly.dat")
            if "g" in kind:
                continue
        try:
            file = os.path.join(folder, file_name)
            ops = amp.get_op_from(file)
            op = ops[ind]
            facts[kind] = (
                parameters.facts[kind]
                .subs({"c_u": ci[0], "c_d": ci[1], "c_g": ci[2]})
                .subs({"c_u": ci[0], "c_d": ci[1], "c_g": ci[2]})
                .subs(amp.op_consts)
                / A
            )
            op *= np.float128(facts[kind])
            operators[kind] = op
        except IOError:  # Inconsistent naming scheme bug fix
            pass

    # Print operators
    for kind, op in operators.items():
        logger.info(
            "Found matrix element: {nuc:4s} - {kind} ({name})".format(
                nuc=nucleon, kind=kind, name=op.name
            )
        )
        logger.info(
            "obtains the factor: {fact} = {nfact}\n".format(
                fact=par.facts[kind], nfact=np.float128(facts[kind])
            )
        )

    return operators


# -------------------------------
def get_amplitudes(operators, s_only=False, parameters=par, adjust_g_q_piece=False):
    """
    Computes all possible amplitudes and differences
    """
    logger.info("Computing amplitudes.")
    if not (s_only):
        amp_lo_g = operators["scalar-g"]

        amp_nlo_g_q = amp_lo_g + operators["2-pion-gq"]
        amp_nlo_g = amp_lo_g + operators["2-pion-g"]
        if adjust_g_q_piece:
            temp = copy.deepcopy(operators["2-pion-gq"])
            temp *= -(1.0 / 2)
            amp_nlo_g = amp_nlo_g + temp
        amp_nlo_g_g = amp_nlo_g + operators["2-pion-gq"]

        del_lo_g = {
            "q": amp_lo_g.get_amp_difference(amp_nlo_g_q),
            "g": amp_lo_g.get_amp_difference(amp_nlo_g_g),
        }
        del_nlo_g = {"NLO": amp_nlo_g.get_amp_difference(amp_lo_g)}
    else:
        amp_lo_g = None
        del_lo_g = {}
        amp_nlo_g = None
        del_nlo_g = {}

    amps = [
        {"sm": "g", "chi": "lo", "type": "amp", "val": amp_lo_g},
        {"sm": "g", "chi": "lo", "type": "q", "val": del_lo_g.get("q")},
        {"sm": "g", "chi": "lo", "type": "g", "val": del_lo_g.get("g")},
        {"sm": "g", "chi": "nlo", "type": "amp", "val": amp_nlo_g},
        {"sm": "g", "chi": "nlo", "type": "NLO", "val": del_nlo_g.get("NLO")},
    ]

    amp_lo_q = operators["scalar"] + operators["tau3"]
    del_lo_q = {"iv": amp_lo_q.get_amp_difference(operators["scalar"])}

    amp_nlo_q_1B = amp_lo_q + operators["scalar_q2"]
    amp_nlo_q_2B = amp_lo_q + operators["2-pion-q"]
    amp_nlo_q = amp_nlo_q_1B + operators["2-pion-q"]

    del_nlo_q = {
        "NLO": amp_nlo_q.get_amp_difference(amp_lo_q),
        "1B": amp_nlo_q.get_amp_difference(amp_nlo_q_2B),
        "2B": amp_nlo_q.get_amp_difference(amp_nlo_q_1B),
    }

    amps += [
        {"sm": "q", "chi": "lo", "type": "amp", "val": amp_lo_q},
        {"sm": "q", "chi": "lo", "type": "iv", "val": del_lo_q.get("iv")},
        {"sm": "q", "chi": "nlo", "type": "amp", "val": amp_nlo_q},
        {"sm": "q", "chi": "nlo", "type": "NLO", "val": del_nlo_q.get("NLO")},
        {"sm": "q", "chi": "nlo", "type": "1B", "val": del_nlo_q.get("1B")},
        {"sm": "q", "chi": "nlo", "type": "2B", "val": del_nlo_q.get("2B")},
    ]

    amplitudes = pd.DataFrame(amps, columns=["sm", "chi", "type", "val", "label"])

    if parameters.normalized:
        amp_type = r"\mathcal{{ {F} }} "
    else:
        amp_type = r"\overline{{ \mathcal{{ {M} }} }}"

    for ind in amplitudes.index:
        sm, chi, tp, val, label = amplitudes.loc[ind]
        div_A2 = ""
        if parameters.normalized:
            sm = ""
            if tp == "amp":
                div_A2 = r"/ A^2"
        #       if sm == "q":
        #         if sm=="iv":
        #           sm="2"
        #         else:
        #           sm="1"
        #       else:
        #         sm = "3"
        label = (
            r" \left| "
            r"{amp_type}  _{{{sm}}}^{{(\mathrm{{{chi}}})}}"
            r" \right|^2 {div_A2}"
        )
        label = label.format(sm=sm, chi=chi.upper(), amp_type=amp_type, div_A2=div_A2)
        if tp == "amp":
            amplitudes.loc[ind, "label"] = "$" + label + "$"
        else:
            amplitudes.loc[ind, "label"] = (
                r"$ \Delta^{{({tp})}}".format(tp=tp) + label + "$ in %"
            )

    logger.info(
        "Collected the following amplitudes:\n" + str(amplitudes[["sm", "chi", "type"]])
    )

    return amplitudes


# -------------------------------
def make_plots(
    amplitudes,
    name,
    query="sm == 'q'",
    estimate_type=1,
    results_folder="results",
    **kwargs
):
    """ """

    RESULTS_FOLDER = os.path.join(DM_FOLDER, "analysis", results_folder)

    logger.info("Start plotting. Query: " + query)
    amps = amplitudes.query(query)
    plot_ops = list(amps["val"])
    container = amp.OperatorPlotContainer(plot_ops, estimate_type=estimate_type)
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    logger.info("Putting results in: " + RESULTS_FOLDER)

    plot_labels = list(amps["label"])
    N_plots = len(plot_labels)

    dpi = kwargs.get("dpi", 500)
    legend = kwargs.get("legend", False)

    # q_dependence
    fig = plt.figure(figsize=(inch_max_w, inch_max_w * N_plots * 1.0 / 5), dpi=dpi)
    container.plot_q_dep(fig, legend=legend, ylabels=plot_labels)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(RESULTS_FOLDER + "/amp_" + name + ".pdf", bbox_inches="tight")

    # cutoff
    fig = plt.figure(figsize=(inch_max_w, inch_max_w * N_plots * 1.0 / 4), dpi=dpi)
    container.plot_cut_dep(fig, legend=legend, ylabels=plot_labels)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(RESULTS_FOLDER + "/err_" + name + ".pdf", bbox_inches="tight")

    # conclusion
    fig = plt.figure(figsize=(inch_max_w, inch_max_w * 1.0 / (N_plots + 0.5)), dpi=dpi)
    container.plot_amplitudes(fig, legend=legend, ylabels=plot_labels)
    plt.subplots_adjust(wspace=0.25, hspace=0.02)
    fig.savefig(RESULTS_FOLDER + "/con_" + name + ".pdf", bbox_inches="tight")


# -------------------------------
def do_plots(s_only, estimate_type, results_folder, parameters=par):
    """ """
    if s_only:
        s_str = "_S"
    else:
        s_str = ""

    for nucleon in par.nuc_sysetms:
        operators = get_operators(nucleon, s_only=s_only, parameters=parameters)
        amplitudes = get_amplitudes(operators, s_only=s_only, parameters=parameters)
        make_plots(
            amplitudes,
            nucleon + "_q" + s_str,
            query="sm == 'q'",
            estimate_type=estimate_type,
            results_folder=results_folder,
            legend=True,
        )
        if not (s_only):
            make_plots(
                amplitudes,
                nucleon + "_g" + s_str,
                query="sm == 'g'",
                estimate_type=estimate_type,
                results_folder=results_folder,
                legend=True,
            )


# ===============================================================================
#     Exe
# ===============================================================================
if __name__ == "__main__":

    args = parser.parse_args()
    s_only = args.sonly
    out_folder = args.out_folder
    estimate_type = args.estimate_type
    normalized = not (args.non_normalized)

    logger.info("Start generating plots")
    logger.info("DM folder: " + DM_FOLDER)

    if s_only:
        logger.info("Using S only results.")
    if out_folder:
        logger.info("Out folder:" + out_folder)
    if not (estimate_type):
        estimate_type = 2
    if not (estimate_type in [0, 1, 2]):
        raise ValueError("Estimate type needs to be 0,1 or 2.")
    logger.info("Estiamte type: " + _estimate_type_dict[estimate_type])
    if normalized:
        logger.info("Plotting normalized results.")
    else:
        logger.info("Plotting non-normalized results.")
    par = parameters(normalized)

    do_plots(s_only, estimate_type, out_folder, parameters=par)
