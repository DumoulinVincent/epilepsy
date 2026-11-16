"""Image stimulus for the image-recovery task (conditions.IMAGE_GRID).

Grid convention (seeded_network.e_cell_xy): E cell i sits at x = i % 10,
y = i // 10, i.e. i = 10 y + x. An image is a binary mask p (100 cells, exactly
k_white white). During an image, cell i receives

    lambda_i = f + a (p_i - q),   q = k_white / 100

so white cells get f + 0.9 a, black cells f - 0.1 a and the summed rate is
exactly 100 f; during blanks every cell gets f. The image source REPLACES the
ordinary drive (one independent Poisson train per E cell, one-to-one, same
synaptic weight as the restored drive).

Everything is on a sched_ms grid (10 ms): image and blank durations, onsets,
spike bins and readout windows are whole numbers of schedule bins, so there
is no boundary arithmetic in floating point.
"""
import numpy as np

N_CELLS = 100
GRID = 10


def cell_xy():
    i = np.arange(N_CELLS)
    return i % GRID, i // GRID


# ------------------------------------------------------------------ libraries
def random_masks(n_blocks, k_white=10, seed=0):
    """n_blocks x (100 / k_white) masks: each block is a random permutation of
    the 100 cells cut into disjoint k_white-cell masks, so every cell is white
    in exactly n_blocks masks (balanced exposure)."""
    if N_CELLS % k_white:
        raise ValueError("k_white must divide 100")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_blocks):
        perm = rng.permutation(N_CELLS)
        for m in range(N_CELLS // k_white):
            mask = np.zeros(N_CELLS, np.uint8)
            mask[perm[m * k_white:(m + 1) * k_white]] = 1
            out.append(mask)
    return np.array(out)


def line_masks():
    """The 10 horizontal (y = r) then the 10 vertical (x = c) full lines;
    every cell is white in exactly 2 of the 20."""
    x, y = cell_xy()
    return np.array([(y == r).astype(np.uint8) for r in range(GRID)]
                    + [(x == c).astype(np.uint8) for c in range(GRID)])


def make_library(g):
    if g["library"] == "random":
        masks = random_masks(g["n_library_blocks"], g["k_white"], g["stim_seed"])
        names = [f"random{m}" for m in range(len(masks))]
    elif g["library"] == "lines":
        if g["k_white"] != GRID:
            raise ValueError("the line library has 10 white pixels per image")
        masks = line_masks()
        names = [f"row{r}" for r in range(GRID)] + [f"col{c}" for c in range(GRID)]
    else:
        raise ValueError(g["library"])
    assert np.all(masks.sum(1) == g["k_white"]), "every mask must have k_white white pixels"
    return masks, names


# ------------------------------------------------------------------- schedule
def make_schedule(n_masks, run_s, g):
    """Blank, image, blank, image, ... until run_s. Presentation order: shuffled
    permutations of the mask IDs, concatenated (every mask once per block).
    Returns, in schedule bins (sched_ms): onset, offset (image is
    [onset, offset)), blank (the blank before each image), mask_id, and
    img_series (per schedule bin: 0 = blank, m + 1 = mask m shown)."""
    sb = g["sched_ms"]
    for v in (g["image_ms"], *g["blank_ms"], g["response_start_ms"], *g["readout_ms"]):
        if v % sb:
            raise ValueError(f"{v} ms is not a multiple of sched_ms = {sb}")
    if g["response_start_ms"] + max(g["readout_ms"]) > g["image_ms"]:
        raise ValueError("every readout window must end while the image is shown")
    n_total = int(round(run_s * 1000)) // sb
    img_n = g["image_ms"] // sb
    blanks = np.arange(g["blank_ms"][0] // sb, g["blank_ms"][1] // sb + 1)
    rng = np.random.default_rng(g["order_seed"])
    onset, offset, blank, mask_id = [], [], [], []
    order = []
    t = 0
    while True:
        if not order:
            order = list(rng.permutation(n_masks))
        b = int(rng.choice(blanks))
        if t + b + img_n > n_total:
            break
        onset.append(t + b)
        offset.append(t + b + img_n)
        blank.append(b)
        mask_id.append(int(order.pop()))
        t += b + img_n
    onset, offset = np.array(onset), np.array(offset)
    img_series = np.zeros(n_total, np.int32)
    for a, z, m in zip(onset, offset, mask_id):
        img_series[a:z] = m + 1
    return dict(onset=onset, offset=offset, blank=np.array(blank), mask_id=np.array(mask_id),
                img_series=img_series, n_bins=n_total)


def rate_table(masks, f_hz, a_hz, k_white):
    """Row 0: blank (every cell f). Row m + 1: mask m, f + a (p - q). Asserts
    non-negative rates and an exact summed rate of 100 f (no clipping)."""
    q = k_white / N_CELLS
    table = np.vstack([np.full(N_CELLS, float(f_hz)),
                       f_hz + a_hz * (masks.astype(float) - q)])
    if np.any(table < 0):
        raise ValueError("negative input rate: f must be >= a * q")
    if not np.allclose(table.sum(1), N_CELLS * f_hz):
        raise ValueError("summed input rate must be 100 f in every row")
    return table


def add_image_input(net, img_series, table, sched_ms, weight, name="img"):
    """The image drive as one Brian2 PoissonGroup (one train per E cell,
    one-to-one onto s_AMPA_ext with `weight`). Rates are looked up, per
    schedule bin counted from the current net.t, in `table`: row
    img_series[k] during bin k. Call after any restore, before the run."""
    from brian2 import TimedArray, PoissonGroup, Synapses, Hz, ms
    img_ta = TimedArray(np.asarray(img_series, float), dt=sched_ms * ms)
    # rows of the table are addressed as "times" 0, 1, 2, ... ms; +0.5 ms
    # keeps the floor() of the lookup inside the intended row
    rate_ta = TimedArray(np.asarray(table, float) * Hz, dt=1 * ms)
    pg = PoissonGroup(table.shape[1], rates="rate_ta((img_ta(t - t_img0) + 0.5) * ms, i)",
                      name=name, namespace=dict(img_ta=img_ta, rate_ta=rate_ta, t_img0=net.t))
    syn = Synapses(pg, net["P_E"], model="w:1", on_pre="s_AMPA_ext += w", method="euler",
                   name=f"{name.upper()}_E")
    syn.connect(j="i")
    syn.w[:] = float(weight)
    net.add(pg, syn)
    return pg, syn


# ------------------------------------------------------------- tick binning
def tick_bins(t_s, t0_s, dt_s, ticks_per_bin):
    """Bin index of event times that lie on integration ticks: integer ticks
    since t0 (rounded, so floating-point noise cannot cross a boundary), then
    integer division. Half-open bins [k, k + 1)."""
    ticks = np.round((np.asarray(t_s, float) - t0_s) / dt_s).astype(np.int64)
    return ticks // ticks_per_bin
