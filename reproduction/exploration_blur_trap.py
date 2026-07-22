# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14.0",
#   "matplotlib>=3.9.0",
#   "numpy>=1.26.0",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Exploration matters for escaping the 3DGS blur trap

    This notebook is a self-contained companion to the reproduction of
    *Exploration Matters for Escaping the Blur Trap in 3D Gaussian
    Splatting* (arXiv:2607.17965). It summarizes terminal measurements from
    **46 successful Kubernetes runs**. The real-scene implementation uses
    official Graphdeco 3DGS with independently implemented paper-described
    exploration operators because the authors' implementation was not
    publicly available.

    **Verdict: partially reproduced.** The controlled mechanisms reproduce
    strongly; the real-scene direction repeats across independent seed
    blocks, but its magnitude is small and scene-dependent.
    """)
    return


@app.cell
def _():
    evidence = {
        "headline": {
            "scenes": ["Dr Johnson", "Playroom", "Train", "Truck"],
            "baseline": [29.299235, 30.336821, 22.040118, 25.408424],
            "split20": [29.333204, 30.354950, 22.142424, 25.384945],
            "block_gains": [0.032731, 0.014546],
        },
        "timing": {
            "labels": ["Early\n49", "Middle\n50", "Late\n50", "Distributed\n48", "Full\n144"],
            "delta": [0.001077, 0.000979, -0.001862, 0.015060, 0.032731],
        },
        "seeds": {
            "count": [1, 5, 20, 100],
            "delta": [0.000522, -0.013855, 0.027287, 0.026132],
        },
        "efficiency": {
            "labels": ["seed20", "split20", "distributed48", "seed100", "split200", "half threshold"],
            "time": [0.212, 0.023, 0.497, 0.548, 1.340, 64.793],
            "delta": [0.027287, 0.032731, 0.015060, 0.026132, 0.022404, -0.237653],
            "gaussians": [0.075, 0.132, -0.008, 0.580, 1.161, 129.814],
        },
        "diagnostic": {
            "labels": ["Naive\nrandom", "Largest-scale\nearly", "Opacity\npreserving", "Low-gradient\nlarge-scale", "Rear\nlarge-scale"],
            "psnr": [39.44, 39.80, 41.89, 39.84, 46.91],
            "baseline": 44.67,
        },
    }
    return (evidence,)


@app.cell
def _(mo):
    view = mo.ui.dropdown(
        options={
            "Headline — native scenes": "headline",
            "Dynamics — timing coverage": "timing",
            "Robustness — seed dose": "seeds",
            "Efficiency — runtime frontier": "efficiency",
            "Diagnostic — what to split": "diagnostic",
        },
        value="Headline — native scenes",
        label="Evidence view",
    )
    view
    return (view,)


@app.cell
def _(evidence, mo, np, plt, view):
    _colors = {"baseline": "#7F8894", "split": "#D95F59", "accent": "#6F4E7C"}
    _fig, _ax = plt.subplots(figsize=(8.8, 4.8))

    if view.value == "headline":
        _d = evidence["headline"]
        _x = np.arange(len(_d["scenes"]))
        _width = 0.36
        _ax.bar(_x - _width / 2, _d["baseline"], _width, color=_colors["baseline"], label="Official 3DGS")
        _ax.bar(_x + _width / 2, _d["split20"], _width, color=_colors["split"], label="Random split20")
        for _i, (_base, _split) in enumerate(zip(_d["baseline"], _d["split20"])):
            _ax.text(_i + _width / 2, _split + 0.16, f"{_split - _base:+.3f}", ha="center", color=_colors["split"])
        _ax.set_xticks(_x, _d["scenes"])
        _ax.set_ylim(20, 31.5)
        _ax.set_ylabel("Test PSNR (dB)")
        _ax.set_title("Native 30k split20; independent block gains +0.033 / +0.015 dB")
        _ax.legend(frameon=False)

    elif view.value == "timing":
        _d = evidence["timing"]
        _x = np.arange(len(_d["labels"]))
        _bars = _ax.bar(_x, _d["delta"], color=["#7FB3D5", "#2878B5", "#1B4F72", _colors["accent"], _colors["split"]])
        _ax.axhline(0, color="#222222", linewidth=1)
        for _bar, _value in zip(_bars, _d["delta"]):
            _ax.text(_bar.get_x() + _bar.get_width() / 2, _value + (0.0015 if _value >= 0 else -0.0015), f"{_value:+.3f}", ha="center", va="bottom" if _value >= 0 else "top")
        _ax.set_xticks(_x, _d["labels"])
        _ax.set_ylabel("PSNR change (dB)")
        _ax.set_title("Matched event count: distributed coverage beats contiguous windows")

    elif view.value == "seeds":
        _d = evidence["seeds"]
        _ax.plot(_d["count"], _d["delta"], marker="o", linewidth=2.2, color="#2C7FB8")
        _ax.axhline(0, color="#222222", linewidth=1)
        for _count, _value in zip(_d["count"], _d["delta"]):
            _ax.text(_count, _value + 0.002, f"{_value:+.3f}", ha="center")
        _ax.set_xscale("log")
        _ax.set_xticks(_d["count"], [str(_v) for _v in _d["count"]])
        _ax.set_xlabel("Random seeds per event")
        _ax.set_ylabel("PSNR change (dB)")
        _ax.set_title("Native seed dose is non-monotonic; 20/event is the efficient peak")

    elif view.value == "efficiency":
        _d = evidence["efficiency"]
        _sizes = 55 + 5 * np.sqrt(np.maximum(_d["gaussians"], 0))
        _ax.scatter(_d["time"][:-1], _d["delta"][:-1], s=_sizes[:-1], color="#2C7FB8")
        for _label, _xv, _yv in zip(_d["labels"][:-1], _d["time"][:-1], _d["delta"][:-1]):
            _ax.annotate(_label, (_xv, _yv), xytext=(6, 5), textcoords="offset points")
        _ax.axhline(0, color="#222222", linewidth=1)
        _ax.set_xlabel("Training-time overhead (%)")
        _ax.set_ylabel("PSNR change (dB)")
        _ax.set_title("Sparse exploration stays near the quality/runtime frontier")
        _ax.text(0.98, 0.08, "Brute force: −0.238 dB\n+64.8% time, +129.8% Gaussians", transform=_ax.transAxes, ha="right", bbox={"facecolor": "#F4F4F4", "edgecolor": "#BBBBBB"})

    else:
        _d = evidence["diagnostic"]
        _x = np.arange(len(_d["labels"]))
        _bar_colors = ["#C5C9CE"] * 4 + [_colors["split"]]
        _ax.bar(_x, _d["psnr"], color=_bar_colors)
        _ax.axhline(_d["baseline"], color=_colors["baseline"], linestyle="--", linewidth=2, label=f"No-split baseline: {_d['baseline']:.2f} dB")
        _ax.set_xticks(_x, _d["labels"])
        _ax.set_ylim(34, 49)
        _ax.set_ylabel("Near-region test PSNR (dB)")
        _ax.set_title("Controlled diagnostic: where and what to split matters")
        _ax.legend(frameon=False)

    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(axis="y", alpha=0.2)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("Source: terminal Kubernetes run summaries; see the linked detailed report for run IDs, uncertainty, and limitations.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Evidence boundary

    - 46 successful runs contained terminal metric evidence.
    - 10 cancelled and 7 failed attempts are excluded from all measurements.
    - The Kubernetes leader exposed only eight individual rank records for
      real-scene jobs; 16-trial claims use the terminal aggregate summaries,
      not an unsupported 16-pair significance test.
    - The real-scene code is an independent approximation of unreleased
      ExploreGS code, so the result is **partially reproduced**, not a
      bit-exact confirmation.
    """)
    return


if __name__ == "__main__":
    app.run()
