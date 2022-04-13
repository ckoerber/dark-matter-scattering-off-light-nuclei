#!/usr/bin/env python
# -------------------------------
#  Reades output files for cross sections
# -------------------------------
import pandas as pd
import numpy as np
import scipy.special as spec

# -------------------------------
from sympy import S
import phys_consts
import scipy.interpolate as itp

# -------------------------------
import re
import phys_consts as consts
import copy
from collections import OrderedDict

# -------------------------------

# ---------------Definitions----------------
cutoff_dict_fm = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2}

cutoff_dict_MeV = {}
for key, val in cutoff_dict_fm.items():
    cutoff_dict_MeV[key] = 2.0 / val * consts.hbarc.MeV


def order_to_tex(order):
    order = re.findall("(n*)([0-9]*)lo", order)
    order = int(order[0][1])
    if order == 0:
        out = "LO"
    elif order == 1:
        out = "NLO"
    else:
        out = "N$^{" + str(order) + "}$LO"
    return out


Lambda_dict_MeV = {
    1: 600,
    2: 600,
    3: 600,
    4: 500,
    5: 400,
}

op_consts = {
    "epsilon": 0.37,
    "delta_epsilon": 0.03,
    "sigma_pi_N": 59.1 / phys_consts.hbarc.MeV,
    "delta_sigma_pi_N": 3.5 / phys_consts.hbarc.MeV,
    "delta_mN": 2.49 / phys_consts.hbarc.MeV,
    "delta_delta_mN": 0.17 / phys_consts.hbarc.MeV,
    "sigma_s": 40 / phys_consts.hbarc.MeV,
    "mpi": phys_consts.mpi.fm,
    "pi": np.pi,
    "gA": phys_consts.gA.MeV,
    "f_pi": phys_consts.fpi.fm,
    "mN": phys_consts.mN.fm,
}

_fact1 = (1 - S("epsilon")) * S("c_u") + (1 + S("epsilon")) * S("c_d")
_fact2 = (1 - 1 / S("epsilon")) * S("c_u") + (1 + 1 / S("epsilon")) * S("c_d")

_add_fact = +1

operator_factors = {
    "2-pion-q": _add_fact * S("mpi") ** 2 / 4 * _fact1 / (2 * S("pi")) ** 3,
    "1-scalar": S("sigma_pi_N") / 2 * _fact1,
    "1-scalar-q2": -5
    * S("gA") ** 2
    * S("mpi")
    / 32
    / (4 * S("pi") * S("f_pi")) ** 2
    * _fact1,
    "1-tau3": -S("delta_mN") / 4 * _fact2,
    "2-pion-g": _add_fact * S("c_g") / (2 * S("pi")) ** 3 / 2,
    "1-scalar-g": -S("c_g")
    * S("pi")
    / 9
    * 8
    * (S("mN") - S("sigma_pi_N") - S("sigma_s")),
}

normalized_factors = {}
for key, val in operator_factors.items():
    normalized_factors[key] = val / (S("sigma_pi_N"))


# ---------------Definitions----------------


def spline(x_old, y_old, x_new):
    return itp.interp1d(x_old, y_old)(x_new)


# ---------------get_op_from----------------
def get_op_from(file_name):
    """
    Reads the file and returns the operators as an array of 'Operator' classes
    """
    frame = pd.read_csv(file_name, sep=r"\s+")
    cols = list(frame.columns)
    ops = cols[:]
    non_op_cols = ["wave", "xi", "mxi", "q"]
    for col in non_op_cols:
        ops.remove(col)

    operators = []

    for op_string in ops:
        op_name = file_name.split("/")[-1] + "_" + op_string
        op_frame = frame[non_op_cols + [op_string]]
        op_frame.columns = non_op_cols + ["mat"]
        operators.append(Operator(op_name, op_frame))

    return operators
    # ---------------get_op_from----------------


# ---------------Operator----------------
class Operator(object):
    # ---------------__init__----------------
    def __init__(self, op_name, op_frame):
        """
        Stores the operator in a pandas Frame containing the amplitude.
        The computation parameters can be found in mesh.
        """
        self.name = op_name
        self.frame = op_frame.round({"q": 3})

        # get additional data
        variables = ["QMAX", "NQ", "NX", "NPHI", "XI", "L"]
        self.mesh = {}
        for var in variables:
            val = re.findall("_" + var + "([0-9.]+)", self.name)
            try:
                self.mesh[var] = int(val[0]) if len(val) > 0 else None
            except Exception:
                self.mesh[var] = float(val[0]) if len(val) > 0 else None
        if self.mesh["XI"] is None:
            self.mesh["XI"] = self.frame.xi.max()
        # compute amplitude according to amp = Sum_{xi, mxi} A_{xi, mxi}^2 / hat(xi)
        amp = self.frame.copy()
        amp.loc[:, "mat"] *= amp.loc[:, "mat"]
        amp.loc[:, "mat"] /= 2 * amp.loc[:, "xi"] + 1
        self.amp = amp.groupby(["wave", "q"]).agg({"mat": "sum"}).unstack(0)
        self.amp.columns = self.amp.columns.droplevel()
        self.amp_frame = self.amp.unstack().reset_index()
        self.amp_frame.columns = list(self.amp_frame.columns)[:-1] + ["amp"]
        # ---------------__init__----------------

    # ---------------get_chiral_frame----------------
    def get_chiral_frame(self, tex=True, frame=None):
        """ """
        if frame is None:
            frame = self.amp_frame
        frame = frame[frame.wave.str.contains("^ich")].copy()

        cutoff_patterns = {}
        for n in range(1, 5 + 1):
            if tex:
                cutoff_patterns["=" + str(n)] = r"${0:3.0f} \ MeV$".format(
                    cutoff_dict_MeV[n]
                )
                order_pattern = {
                    "-lo-": r"$LO   $",
                    "-nlo-": r"$N LO$",
                    "-n2lo-": r"$N^2LO$",
                    "-n3lo-": r"$N^3LO$",
                    "-n4lo-": r"$N^4LO$",
                }
            else:
                cutoff_patterns["=" + str(n)] = "c" + str(n)
                order_pattern = {
                    "-lo-": "lo",
                    "-nlo-": "nlo",
                    "-n2lo-": "n2lo",
                    "-n3lo-": "n3lo",
                    "-n4lo-": "n4lo",
                }

        filters = {
            "Order": order_pattern,
            "Cutoff": cutoff_patterns,
        }

        for key, rules in filters.items():
            for pattern, val in rules.items():
                inds = frame[frame.wave.str.contains(pattern)].index
                frame.loc[inds, key] = val

        frame.sort_values(["q", "Order", "Cutoff"], inplace=True)
        frame.index = np.arange(len(frame.index))
        frame.loc[:, "q"] *= consts.hbarc.MeV

        return frame.drop("wave", axis=1)
        # ---------------get_chiral_frame----------------

    # ---------------__imul__----------------
    def __imul__(self, other):
        """ """
        self.frame.loc[:, "mat"] *= other
        for col in self.amp.columns:
            self.amp.loc[:, col] *= abs(other) ** 2
        self.amp_frame.loc[:, "amp"] *= abs(other) ** 2
        return self
        # ---------------__imul__----------------

    # ---------------__add__----------------
    def __add__(self, other):
        """ """
        if not (isinstance(other, Operator)):
            raise TypeError(str(other) + " needs to be an 'Operator'")

        # Works since pandas rounds q to third digit
        q1x100 = np.array(self.frame.q.unique() * 100, dtype=np.int)
        q2x100 = np.array(other.frame.q.unique() * 100, dtype=np.int)
        q = np.array(list(set(q1x100).intersection(q2x100)), dtype=np.float) / 100

        op_frame = self.frame.copy()[self.frame["q"].isin(q)]
        op_frame.set_index(["wave", "xi", "mxi", "q"], inplace=True)

        other_frame = other.frame.copy()[other.frame["q"].isin(q)]
        other_frame.set_index(["wave", "xi", "mxi", "q"], inplace=True)

        other_frame = other_frame.loc[op_frame.index].fillna(0)
        op_frame = op_frame.loc[other_frame.index].fillna(0)

        op_frame += other_frame

        op_frame.reset_index(inplace=True)

        return Operator(self.name + other.name, op_frame)
        # ---------------__add__----------------

    # ---------------__sub__----------------
    def __sub__(self, other):
        """ """
        o = copy.deepcopy(other)
        s = copy.deepcopy(self)
        o *= -1
        return s + o
        # ---------------__sub__----------------

    # ---------------subtract_amp----------------
    def get_amp_difference(self, other, scale=None, relative=True):
        """
        Subtract the amp of other operator from this operator and returns a new
        operator frame.  The ampliuted is the relative difference in percent.
        """
        diff_op = copy.deepcopy(self)
        other_frame = other.amp_frame.copy()

        diff_op.amp_frame.set_index(["wave", "q"], inplace=True)
        other_frame.set_index(["wave", "q"], inplace=True)
        other_frame = other_frame.loc[diff_op.amp_frame.index].fillna(0)

        divide_by = abs(diff_op.amp_frame + other_frame)
        if scale:
            divide_by += 1.0e-3 * scale

        fact = np.float128(1)
        if relative:
            fact = np.float128(200) / divide_by
        diff_op.amp_frame = (diff_op.amp_frame - other_frame) * fact
        diff_op.amp_frame.reset_index(inplace=True)
        diff_op.amp_frame = diff_op.amp_frame.round({"amp": 7})

        return diff_op

        # ---------------subtract_amp----------------


# ---------------Operator----------------


# ---------------HelmFormFactor----------------
class HelmFormFactor(object):
    """
    Allocates the Helm form factors according to
    arXiv:1403.5134v2
    """

    # ---------------init-----------------------
    def __init__(self, A, a=0.52, s=0.9):
        """
        rN^2 = c^2 + 7/3 pi^2 a^2 - 5 s^2
        c    = 1.23 A^{1/3} - 0.60fm
        a in fm
        s in fm
        """
        self.c = 1.23 * A ** (1.0 / 3) - 0.6
        self.a = a
        self.s = s
        self.A = A
        self.rN = np.sqrt(
            self.c ** 2 + 7.0 / 3 * np.pi ** 2 * self.a ** 2 - 5 * self.s ** 2
        )

    # ---------------init-----------------------

    # ---------------amp-----------------------
    def amp(self, q):
        """
        -> q [fm^{-1}]

        Returns the Helm Form Factor defined by
        F_A^{Helm}(q) = 3* j_1( x ) / x * exp( -(qs)**2 /2 )

        with x = q*r_N and j_1 being the first order spherical Bessel function
        """
        x = self.rN * q
        out = 3 * spec.spherical_jn(1, x) * np.exp(-((q * self.s) ** 2) / 2)

        # make sure to not divide by zero
        return np.array([o / x0 if abs(x0) > 1.0e-7 else 1.0 for o, x0 in zip(out, x)])
        # ---------------amp-----------------------


# ---------------HelmFormFactor----------------


# ---------------ChiralErrorEstimate----------------
class ChiralErrorEstimate(object):
    """
    Establishes a chiral error estimate for a chiral frame.
    """

    # ---------------__init__-----------------------
    def __init__(self, chiral_frame, estimate_type=2, modify_Q=1.0):
        r"""
        Takes a pandas frame with the columns
        'q', 'amp', 'Cutoff', 'Order',
        where order \in {'lo', 'nlo', 'n2lo', ...}.
        """
        self.estimate_type = estimate_type
        self._organize_frame(chiral_frame)
        self.del_x_frame = pd.DataFrame()

        self.modify_Q = modify_Q

        for cutoff in self.cutoffs:
            cut_frame = self.frame.query("Cutoff == @cutoff")
            for q in self.q:
                val_frame = cut_frame.query("q == @q")
                del_amps = self._estimate_chiral_errors(val_frame)
                self.frame.loc[val_frame.index, "d_amp"] = del_amps
        # ---------------__init__-----------------------

    # ---------------_organize_frame-----------------------
    def _organize_frame(self, chiral_frame):
        """
        Sorts and organizes the internal frame
        """
        # Check columns
        cols = set(["q", "amp", "Cutoff", "Order"])
        if cols == set(chiral_frame.columns):
            self.frame = chiral_frame[["Cutoff", "Order", "q", "amp"]]
        else:
            raise ValueError("Frame does not have the right columns")

        # store values
        self.q = self.frame.q.unique()
        self.cutoffs = list(self.frame.Cutoff.unique())
        orders = list(self.frame.Order.unique())

        # check if orders are strange and sort
        self.orders = []
        self._order_dict = {}
        for order in orders:
            match = re.findall("(n*)([0-9]*)lo", order)
            if match:
                if match[0][0] == "":
                    n_order = "0"
                elif match[0][0] == "n":
                    if match[0][1] == "":
                        n_order = "1"
                    else:
                        n_order = match[0][1]
                else:
                    raise ValueError("Order not specified: " + order)
            else:
                raise ValueError("Order not specified: " + order)
            new_order = "n" + n_order + "lo"
            self._order_dict[new_order] = order
            self.orders.append(new_order)
            inds = self.frame.query("Order == @order").index
            self.frame.loc[inds, "Order"] = new_order

        self.orders.sort()
        self.frame.sort_values(["Cutoff", "Order", "q", "amp"], inplace=True)
        # ---------------_organize_frame-----------------------

    # ---------------_estimate_chiral_errors-----------------------
    def _estimate_chiral_errors(self, val_frame):
        """
        Executes the error for a given data point at different orders.  Assumes
        that the orders are already sorted and that the first entry is LO.
        Also assumes q is in MeV.
        """
        cutoff = val_frame.Cutoff.unique()[0]
        cutoff = int(re.findall("c([0-9]+)", cutoff)[0])
        Lambda = Lambda_dict_MeV[cutoff]

        q = val_frame.q.max()

        Q = max(val_frame.q.max(), phys_consts.mpi.MeV) / Lambda
        Q *= self.modify_Q
        # Q    = phys_consts.mpi.MeV / Lambda

        # get values

        X = np.array(val_frame.amp)
        if self.estimate_type < 2:
            # Note: Delta_X starts at order Q^2
            Delta_X = [abs(X[0])] + list(abs(X[1:] - X[:-1]))
            del_X = [abs(X[0]) * Q ** 2]

            for ni in range(1, len(X)):
                i = ni + 1
                order_estimates = [Q ** (i + 1) * Delta_X[0]]
                for nj in range(1, i):
                    j = nj + 1
                    order_estimates.append(Q ** (i + 1 - j) * Delta_X[nj])

                # cannot be smaller than previous order estimate
                del_x = max(order_estimates + [Q * del_X[ni - 1]])

                #
                if self.estimate_type == 1:
                    for xk in X[ni:]:
                        for xj in X[ni:]:
                            diff = abs(xk - xj)
                            if diff > del_x:
                                del_x = diff
                del_X.append(del_x)

        elif self.estimate_type == 2:
            # get chiral orders
            Xo = [(0, X[0])]
            for n_order, x in enumerate(X[1:]):
                Xo += [(n_order + 2, x)]

            order_estimates = []
            del_x_dat = [
                {
                    "nu1": 0,
                    "nu2": 0,
                    "delta_X": abs(X[-1]),
                    "cutoff": cutoff,
                    "q": q,
                }
            ]

            for n_item1, (order1, x1) in enumerate(Xo[1:]):
                for order2, x2 in Xo[: n_item1 + 1]:
                    # q = ( val_frame.q.max() +  phys_consts.mpi.MeV ) / Lambda / 2
                    delta_X = abs(x1 - x2) * 1.0 / Q ** (order2 + 1)
                    order_estimates += [delta_X]
                    del_x_dat += [
                        {
                            "nu1": order1,
                            "nu2": order2,
                            "delta_X": delta_X,
                            "cutoff": cutoff,
                            "q": q,
                        }
                    ]

            del_max = max(order_estimates) * 1.0 / (1 - Q)

            del_X = [Q ** 2 * del_max]

            for nu, x in Xo[1:]:
                del_X += [Q ** (nu + 1) * del_max]

            self.del_x_frame = self.del_x_frame.append(
                pd.DataFrame(data=del_x_dat),
                ignore_index=True,
            )

        else:
            raise ValueError("Error estimate type = 0,1,2")

        return del_X
        # ---------------_estimate_chiral_errors-----------------------


# ---------------ChiralErrorEstimate----------------


# ---------------OperatorPlotContainer----------------
class OperatorPlotContainer(object):
    """ """

    # ---------------__init__-----------------------
    def __init__(self, operator_list, estimate_type=2, modify_Q=1.0):
        """ """
        # get frames
        self.alpha = 0.5
        self.hatch = {
            "n1lo": "\\",
            "n2lo": r"\\",
            "n3lo": "\\\\\\",
            "n4lo": r"/",
            "n5lo": r"//",
            "n6lo": r"///",
            "n7lo": r"+",
        }
        self.amplitudes = []
        self.chiral_estimates = []
        self.delta_X = []
        self.cutoffs = ["c" + str(n) for n in range(1, 6)]
        self.NCutoffs = 5
        self.orders = ["n" + str(n) + "lo" for n in range(0, 5)]
        self.NOrders = 5
        self.phen_colors = [
            u"#2b2b2b",
            u"#5c5c5c",
            u"#828282",
            u"#adadad",
            u"#d1d1d1",
            u"#ededed",
        ]
        self.line_styles = [
            "-",
            "--",
            "-.",
            ":",
            " ",
        ]
        for n_op, op in enumerate(operator_list):
            if not (isinstance(op, Operator)):
                raise ValueError(str(op) + " needs to be an 'Operator' instance.")

            not_chiral_amps = [
                not (el) for el in op.amp_frame.wave.str.contains("^ich")
            ]
            self.amplitudes.append(op.amp_frame[not_chiral_amps].copy())
            self.amplitudes[n_op].loc[:, "q"] *= phys_consts.hbarc.MeV
            chiral_estimate = ChiralErrorEstimate(
                op.get_chiral_frame(tex=False),
                estimate_type=estimate_type,
                modify_Q=modify_Q,
            )
            self.chiral_estimates.append(chiral_estimate.frame)
            self.delta_X.append(chiral_estimate.del_x_frame)

            cutoffs = list(self.chiral_estimates[n_op].Cutoff.unique())
            if cutoffs != self.cutoffs:
                raise IndexError("Cutoffs need to be ['c1', ..., 'c5' ]")

            orders = list(self.chiral_estimates[n_op].Order.unique())
            if orders != self.orders:
                raise IndexError("Orders need to be ['n0lo', ..., 'n4lo' ]")

        # Base plot defs
        self.colors = {
            "n1lo": "#FFEB8F",
            "n2lo": "#92D258",
            "n3lo": "#2A83FD",
            "n4lo": "#FB0007",
        }
        self.markers = {
            "n1lo": "s",
            "n2lo": "v",
            "n3lo": "d",
            "n4lo": "o",
        }
        self.NAmps = len(self.amplitudes)
        # ---------------__init__-----------------------

    # ---------------plot_q_dep-----------------------
    def plot_q_dep(
        self,
        fig,
        amp_range=None,
        ylabels=None,
        legend=False,
        interpolate=False,
    ):
        """ """
        if isinstance(amp_range, list):
            amps = amp_range
        else:
            amps = range(self.NAmps)

        if isinstance(ylabels, list):
            pass
        else:
            ylabels = [r"$ \left| \overline{\mathcal{M}} \right|^2$"] * self.NAmps

        axs = []
        counter = 0

        for nrow, namp in enumerate(amps):
            cf = self.chiral_estimates[namp]
            axrow = []
            for ncol, cut in enumerate(self.cutoffs):
                counter += 1
                sharex = None
                sharey = None
                if ncol > 0:
                    sharey = axrow[0]
                if nrow > 0:
                    sharex = axs[0][0]
                ax = fig.add_subplot(
                    len(amps), self.NCutoffs, counter, sharex=sharex, sharey=sharey
                )
                axrow.append(ax)

                f = cf.query("Cutoff == @cut")

                for order in self.orders:
                    if order == "n0lo":
                        continue
                    tf = f.query("Order == @order")
                    q = tf.q
                    amp = tf.amp
                    d_amp = tf.d_amp
                    if interpolate:
                        q_old = q
                        q = np.linspace(q.min(), q.max(), 100)
                        amp = spline(q_old, amp, q)
                        d_amp = spline(q_old, d_amp, q)
                    ax.fill_between(
                        q,
                        amp + d_amp,
                        amp - d_amp,
                        color=self.colors[order],
                        alpha=self.alpha,
                        hatch=self.hatch[order],
                        edgecolor=self.colors[order],
                        label=order_to_tex(order),
                        interpolate=True,
                        zorder=-1,
                    )
                    ax.fill_between(
                        q,
                        amp + d_amp,
                        amp - d_amp,
                        color="None",
                        hatch=self.hatch[order],
                        edgecolor=self.colors[order],
                        interpolate=True,
                        zorder=-1,
                    )
                    # ax.plot(q, amp+d_amp, "--", color=self.colors[order])
                    # ax.plot(q, amp-d_amp, "--", color=self.colors[order])

                for n_wave, wave in enumerate(self.amplitudes[namp].wave.unique()):
                    tf = self.amplitudes[namp].query("wave == @wave")
                    q = tf.q
                    amp = tf.amp
                    if interpolate:
                        q_old = q
                        q = np.linspace(q.min(), q.max(), 100)
                        amp = spline(q_old, amp, q)
                    ax.plot(
                        q,
                        amp,
                        self.line_styles[n_wave],
                        color=self.phen_colors[n_wave],
                        label=self._non_chiral_wave_label(wave),
                    )

                ax.set_xlim([0, 197])

            axs.append(axrow)

        # Axes labeling
        for nrow, axrow in enumerate(axs):
            for ncol, ax in enumerate(axrow):
                if ncol == 0 and nrow < len(axs) - 1:
                    ax.get_yticklabels()[0].set_visible(False)
                if nrow == len(amps) - 1:
                    ax.set_xlabel(r"$ q \ [MeV \,]$")
                else:
                    for xlabel in ax.get_xticklabels():
                        xlabel.set_visible(False)
                if ncol == len(self.cutoffs) - 1:
                    if legend and nrow == len(axs) / 2:
                        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
                if ncol == 0:
                    ax.set_ylabel(ylabels[nrow])
                else:
                    for ylabel in ax.get_yticklabels():
                        ylabel.set_visible(False)
                if nrow == 0:
                    ax.set_title(
                        r"$\Lambda = {cutoff:3.0f} \ MeV$".format(
                            cutoff=cutoff_dict_MeV[ncol + 1]
                        ),
                        y=1.05,
                    )

        # ---------------plot_q_dep-----------------------

    # ---------------_get_position-----------------------
    def _get_position(self, cut, order):
        """ """
        x0 = int(re.findall("c([0-9]+)", cut)[0])
        delx0 = int(re.findall("n([0-9]+)lo", order)[0]) * 1.0
        delx0 = (delx0 - self.NOrders * 1.0 / 2) / 4 / 2
        return x0 + delx0
        # ---------------_get_position-----------------------

    # ---------------plot_cut_dep-----------------------
    def plot_cut_dep(
        self,
        fig,
        amp_range=None,
        ylabels=None,
        NQ=4,
        legend=False,
        band=["n3lo", "n4lo"],
        band_type="average",
    ):
        """ """
        if isinstance(amp_range, list):
            amps = amp_range
        else:
            amps = range(self.NAmps)

        if isinstance(ylabels, list):
            pass
        else:
            ylabels = [r"$ \left| \overline{\mathcal{M}} \right|^2$"] * self.NAmps

        axs = []
        counter = 0
        for nrow, namp in enumerate(amps):
            axrow = []
            cf = self.chiral_estimates[namp]
            qrange = cf.q.unique()
            qrange = qrange[np.linspace(0, len(qrange) - 1, NQ, dtype=np.int)]
            for ncol, q in enumerate(qrange):
                counter += 1
                sharex = None
                sharey = None
                if ncol > 0:
                    sharey = axrow[0]
                if nrow > 0:
                    sharex = axs[0][0]
                ax = fig.add_subplot(
                    len(amps), NQ, counter, sharex=sharex, sharey=sharey
                )
                axrow.append(ax)
                tf = cf.query("q == @q").copy().drop("q", axis=1)
                avg_amp = {}
                avg_d_amp = {}
                for ind in tf.index:
                    cut, order, amp_, d_amp = tf.loc[ind]
                    if order == "n0lo":
                        continue
                    (_, caps, _) = ax.errorbar(
                        self._get_position(cut, order),
                        amp_,
                        yerr=d_amp,
                        linestyle="None",
                        label=order_to_tex(order),
                        color=self.colors[order],
                        marker=self.markers[order],
                        markersize=6.0,
                    )
                    for cap in caps:
                        cap.set_markeredgewidth(2)

                    # averaging
                    try:
                        if band_type == "average":
                            avg_amp[order] += amp_ / self.NCutoffs
                            avg_d_amp[order] += d_amp / self.NCutoffs
                        elif band_type == "best":
                            if abs(avg_d_amp[order]) > abs(d_amp):
                                avg_amp[order] = amp_
                                avg_d_amp[order] = d_amp
                        else:
                            raise ValueError("Unknown band_type")
                    except Exception:
                        if band_type == "average":
                            avg_amp[order] = amp_ / self.NCutoffs
                            avg_d_amp[order] = d_amp / self.NCutoffs
                        elif band_type == "best":
                            avg_amp[order] = amp_
                            avg_d_amp[order] = d_amp
                        else:
                            raise ValueError("Unknown band_type")

                for order in tf.Order.unique():
                    if not (order in band):
                        continue
                    ax.fill_between(
                        range(self.NCutoffs + 2),
                        avg_amp[order] - avg_d_amp[order],
                        avg_amp[order] + avg_d_amp[order],
                        color=self.colors[order],
                        alpha=self.alpha,
                        hatch=self.hatch[order],
                        edgecolor=self.colors[order],
                    )
                    ax.fill_between(
                        range(self.NCutoffs + 2),
                        avg_amp[order] - avg_d_amp[order],
                        avg_amp[order] + avg_d_amp[order],
                        color="None",
                        hatch=self.hatch[order],
                        edgecolor=self.colors[order],
                    )

            axs.append(axrow)

        # Axes labeling
        for nrow, axrow in enumerate(axs):
            for ncol, ax in enumerate(axrow):
                if nrow > 0 and ncol == 0:
                    pass
                    # ax.get_yticklabels()[-1].set_visible(False)
                    # ax.get_yticklabels()[ 0].set_visible(False)
                if nrow == len(amps) - 1:
                    ax.set_xticklabels(
                        [""]
                        + [
                            r"$\Lambda_{cut}$".format(cut=cut + 1)
                            for cut in range(self.NCutoffs)
                        ]
                        + [""]
                    )
                else:
                    for xlabel in ax.get_xticklabels():
                        xlabel.set_visible(False)
                if ncol == NQ - 1:
                    if legend and nrow == len(amps) / 2:
                        # ax.legend()
                        handles, labels = ax.get_legend_handles_labels()
                        by_label = OrderedDict(zip(labels, handles))
                        ax.legend(
                            by_label.values(),
                            by_label.keys(),
                            numpoints=1,
                            loc="upper left",
                            bbox_to_anchor=(1.05, 1),
                        )
                if ncol == 0:
                    ax.set_ylabel(ylabels[nrow])
                else:
                    for ylabel in ax.get_yticklabels():
                        ylabel.set_visible(False)
                if nrow == 0:
                    ax.set_title(
                        r"$q = {q:3.0f} \ \mathrm{{MeV}}$".format(q=qrange[ncol]),
                        y=1.05,
                    )

        # ---------------plot_cut_dep-----------------------

    # ---------------plot_amplitudes-----------------------
    def plot_amplitudes(
        self,
        fig,
        amp_range=None,
        ylabels=None,
        legend=False,
        band=["n3lo", "n4lo"],
        interpolate=True,
        one_frame=False,
        phenemenological=True,
        upper_bound_only=False,
        row=(1, 1),
        band_type="average",
        legend_cols=1,
        skip_waves=[],
        cut_at_zero=False,
    ):
        """ """
        if isinstance(amp_range, list):
            amps = amp_range
        else:
            amps = range(self.NAmps)

        if isinstance(ylabels, list):
            pass
        else:
            ylabels = [r"$ \left| \overline{\mathcal{M}} \right|^2$"] * self.NAmps

        axs = []
        query = ["Order == '{order}'".format(order=order) for order in band]
        query = " or ".join(query)
        for ncol, namp in enumerate(amps):
            cf = self.chiral_estimates[namp].query(query)
            sharex = None
            if ncol > 0:
                sharex = axs[0]
            if one_frame:
                if ncol == 0:
                    if len(row) == 3:
                        ax = fig.add_subplot(row[0], row[1], row[2])
                    else:
                        ax = fig.add_subplot(1, row[1], row[0])
                    axs.append(ax)
            else:
                ax = fig.add_subplot(
                    row[1],
                    len(amps),
                    ncol + 1 + len(amps) * (row[0] - 1),
                    sharex=sharex,
                )
                axs.append(ax)

            pf = pd.DataFrame()

            # find best cutoff in terms of minimal error bars
            total_del_prev = 1.0e20
            for cutoff in self.cutoffs:
                tf = cf.query("Cutoff == @cutoff").drop(["Cutoff"], axis=1)
                tf.index = range(tf.shape[0])
                try:
                    if band_type == "average":
                        pf.loc[:, ["amp", "d_amp"]] += tf.loc[:, ["amp", "d_amp"]]
                    elif band_type == "best":
                        total_del = np.sum(np.abs(tf.loc[:, "d_amp"]))
                        if total_del < total_del_prev:
                            pf = tf
                            total_del_prev = total_del
                    elif isinstance(band_type, int):
                        if str(band_type) in cutoff:
                            pf = tf
                    else:
                        raise ValueError("Band type unknown")
                except ValueError and KeyError:
                    pf = tf

            if band_type == "average":
                pf.loc[:, ["amp", "d_amp"]] *= 1.0 / len(self.cutoffs)

            for order in pf.Order.unique():
                tf = pf.query("Order == @order")
                q = tf.q
                amp = tf.amp
                d_amp = tf.d_amp
                if interpolate:
                    q_old = q
                    q = np.linspace(q.min(), q.max(), 100)
                    amp = spline(q_old, amp, q)
                    d_amp = spline(q_old, d_amp, q)
                if one_frame and not (phenemenological):
                    color = self.phen_colors[ncol]
                    hatch = list(self.hatch.values())[ncol]
                    label = ylabels[ncol]
                else:
                    color = self.colors[order]
                    hatch = self.hatch[order]
                    label = order_to_tex(order)
                if upper_bound_only:
                    ax.plot(
                        q, amp, color=color, alpha=self.alpha, label=label, zorder=-1
                    )
                    if np.logical_and(amp - d_amp < 1.0e-6, amp > 1.0e-5).any():
                        ax.errorbar(
                            q,
                            amp + d_amp,
                            (amp + d_amp) * 0.8,
                            color=color,
                            alpha=self.alpha,
                            uplims=True,
                            zorder=-2,
                        )
                    else:
                        amp_min = amp - d_amp
                        if cut_at_zero:
                            amp_min = np.min([amp_min, np.zeros(len(amp_min))], axis=1)
                        ax.fill_between(
                            q,
                            amp + d_amp,
                            amp_min,
                            color=color,
                            alpha=self.alpha,
                            hatch=hatch,
                            edgecolor=color,
                            interpolate=True,
                            zorder=-1,
                        )
                else:
                    amp_min = np.array(amp - d_amp)
                    if cut_at_zero:
                        amp_min = np.max([amp_min, np.zeros(len(amp_min))], axis=0)
                    ax.fill_between(
                        q,
                        amp + d_amp,
                        amp_min,
                        color=color,
                        alpha=self.alpha,
                        hatch=hatch,
                        edgecolor=color,
                        label=label,
                        interpolate=True,
                        zorder=-1,
                    )
                    ax.fill_between(
                        q,
                        amp + d_amp,
                        amp_min,
                        color="None",
                        alpha=self.alpha,
                        hatch=hatch,
                        edgecolor=color,
                        interpolate=True,
                        zorder=-1,
                    )
                    ax.plot(q, amp, "-", color=color, zorder=-1)

            if phenemenological and not (one_frame):
                for n_wave, wave in enumerate(self.amplitudes[namp].wave.unique()):
                    if wave in skip_waves:
                        continue
                    tf = self.amplitudes[namp].query("wave == @wave")
                    q = tf.q
                    amp = tf.amp
                    if interpolate:
                        q_old = q
                        q = np.linspace(q.min(), q.max(), 100)
                        amp = spline(q_old, amp, q)
                    ax.plot(
                        q,
                        amp,
                        self.line_styles[n_wave],
                        color=self.phen_colors[n_wave],
                        label=self._non_chiral_wave_label(wave),
                    )

                ax.set_xlim([0, 197])

        # Axes labeling
        for ncol, ax in enumerate(axs):
            if row[0] == row[1] or one_frame:
                ax.set_xlabel(r"$ q \ [\mathrm{MeV} \,]$")
            else:
                for xlabel in ax.get_xticklabels():
                    xlabel.set_visible(False)
            if ncol == len(axs) - 1:
                if legend:
                    ax.legend(
                        loc="upper left", bbox_to_anchor=(1.05, 1), ncol=legend_cols
                    )
            if not (one_frame) and row[0] == 1:
                ax.set_title(ylabels[ncol], y=1.05)

        # ---------------plot_amplitues-----------------------

    # ---------------_non_chiral_wave_label-----------------------
    def _non_chiral_wave_label(self, wave):
        """ """
        if "av18" in wave:
            label = "AV18"
        elif "cdb" in wave:
            label = "CDB"
        elif "nijm93" in wave:
            label = "NIJM93"

        three_body = re.findall(r"-empot-([\w0-9]+)-", wave)
        if len(three_body) > 0:
            three_body = three_body[0]
            if three_body == "no3nf":
                label += " no TNF"
            else:
                label += " + TNF"

        return label
        # ---------------_non_chiral_wave_label-----------------------


# ---------------OperatorPlotContainer----------------
