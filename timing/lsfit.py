import ROOT
import sys

from registry import register_routine
import numpy as np


def load_segments_txt(filename):
    """
    Format expected:

    xc x0 x1 a b c d
    """

    segs = []

    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue

            vals = line.split()

            segs.append({
                "xc": float(vals[0]),
                "x0": float(vals[1]),
                "x1": float(vals[2]),
                "a":  float(vals[3]),
                "b":  float(vals[4]),
                "c":  float(vals[5]),
                "d":  float(vals[6]),
            })

    return segs


class PiecewiseCubicSplineNP:

    def __init__(self, segments, Ts):
        self.x0 = np.array([s["x0"] for s in segments])
        self.x1 = np.array([s["x1"] for s in segments])
        self.xc = np.array([s["xc"] for s in segments])

        self.a = np.array([s["a"] for s in segments])
        self.b = np.array([s["b"] for s in segments])
        self.c = np.array([s["c"] for s in segments])
        self.d = np.array([s["d"] for s in segments])
        self.Ts = Ts

    def _select(self, x):

        x = np.asarray(x)[None, ...]  # (1, E, C, N)

        x0 = self.x0[:, None, None]
        x1 = self.x1[:, None, None]

        inside = (x >= x0) & (x <= x1)

        idx = np.argmax(inside, axis=0)  # (E, C, N)

        xc = self.xc[idx]
        dx = x[0] - xc
        mask = np.abs(dx) <= (self.Ts / 2.0)

        return idx, dx, mask


    def eval(self, x):

        idx, dx, mask = self._select(x)

        a = self.a[idx]
        b = self.b[idx]
        c = self.c[idx]
        d = self.d[idx]

        f = a + b*dx + c*dx*dx + d*dx*dx*dx

        return np.where(mask, f, 0.0)



    def derivative(self, x):

        idx, dx, mask = self._select(x)

        b = self.b[idx]
        c = self.c[idx]
        d = self.d[idx]

        deriv = b + 2*c*dx + 3*d*dx*dx

        return np.where(mask, deriv, 0.0)



def fit_pulse_iterative(waveforms, pulse, t, t_data_peak, t_template_peak, n_iter=4):

    EC, N = waveforms.shape

    # initial alignment from DATA only
    dt = np.full((EC), t_data_peak - t_template_peak)   # (EC)
    A  = np.ones((EC))


    for _ in range(n_iter):

        # -------------------------
        # build shifted time grid
        # -------------------------
        # (EC, N)
        t_shift = t[None, :] - dt[:, None]

        # -------------------------
        # evaluate pulse
        # -------------------------
        P  = pulse.eval(t_shift)        # (EC, N)
        dP = pulse.derivative(t_shift) # (EC, N)
        # -------------------------
        # projections (sum over samples)
        # -------------------------
        Ap = np.sum(waveforms * P, axis=1)   # (EC)
        Ad = np.sum(waveforms * dP, axis=1)


        PP   = np.sum(P * P, axis=1)
        PdP  = np.sum(P * dP, axis=1)
        dPdP = np.sum(dP * dP, axis=1)

        # -------------------------
        # solve 2x2 system
        # -------------------------
        denom = PP * dPdP - PdP * PdP
        denom = np.clip(denom, 1e-12, None)

        A_new = (Ap * dPdP - Ad * PdP) / denom
        Ccorr = (Ad * PP - Ap * PdP) / denom

        # -------------------------
        # update
        # -------------------------
        dt += (-Ccorr / np.clip(A_new, 1e-12, None))
        A = A_new

    return A, dt


@register_routine("lsfit")
def lsfit(signal_window, valid, max_idx, values_max, **kwargs):

    globals().update(kwargs)
    Ts = 1/sampling_rate

    idx_valid = np.where(valid)

    signal_window_valid = signal_window[idx_valid]

    max_idx_valid = max_idx[idx_valid]

    segs = load_segments_txt(spline_file)
    pulse = PiecewiseCubicSplineNP(segs, Ts)

    t_grid = np.arange(signal_window_valid.shape[1]) * Ts

    pulse_values = pulse.eval(t_grid)
    t_pulse_peak = t_grid[np.argmax(pulse_values)]


    amp_valid, dt_valid = fit_pulse_iterative(
        signal_window_valid,
        pulse,
        t_grid,
        signal_samples_pre_peak,  # data peak (scalar)
        t_pulse_peak     # model peak (scalar)
    )

    fit_time_valid = dt_valid + np.ones(dt_valid.shape)*signal_samples_pre_peak*Ts + max_idx_valid*Ts

    fit_t = np.zeros(valid.shape, dtype=np.float32)

    fit_t[idx_valid] = fit_time_valid

    amp_fit = np.zeros(valid.shape, dtype=np.float32)

    amp_fit[idx_valid] = amp_valid

    return {"amp": amp_fit, "time": fit_t}
