import os
import multiprocessing as mp
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


_seed_env = os.environ.get("SEED")
SEED = int(_seed_env) if _seed_env is not None else None
N_TRIALS = int(os.environ.get("N_TRIALS", 10))
N_PAIRS_NOINH = int(os.environ.get("N_PAIRS_NOINH", 13))
N_PAIRS_INH = int(os.environ.get("N_PAIRS_INH", 5))
OUTDIR = os.environ.get("OUTDIR", "barnes_results")


R_ARENA = 54.0
N_HOLES = 18
HOLE_D = 10.0
CREVICE = 1.5
HOLE_RING_R = R_ARENA - HOLE_D / 2 - CREVICE
_goal_env = os.environ.get("GOAL_INDEX")
GOAL_INDEX = int(_goal_env) if _goal_env is not None else int(np.random.default_rng(SEED).integers(0, N_HOLES))
_ga = 2 * np.pi * GOAL_INDEX / N_HOLES
GOAL = np.array([HOLE_RING_R * np.cos(_ga), HOLE_RING_R * np.sin(_ga)], dtype=np.float32)
GOAL_R = 5.0
WALL_R = 51.0
DEMO_START = np.array([-10.0, 0.0], dtype=np.float32)


DT = 0.2
N_PLACE = 1000
N_DIR = 36
PLACE_FIELD_R = np.sqrt(1775 / np.pi)
SIGMA = 7.0
I_PLACE = 1.5

PL_REST, PL_THR, PL_R, PL_C = -65.8, -42.3, 186.0, 0.2688
DR_REST, DR_THR, DR_R, DR_C = -76.0, -44.0, 180.0, 0.1
REFRAC_STEPS = int(1.0 / DT)
STEPS_PER_TICK = 25
WARMUP_STEPS = 100

G_SYN = 0.10
W_EI, W_IE = 0.30, 0.15
TAU_SYN = 4.0
SYN_DECAY = np.exp(-DT / TAU_SYN)


I_SPONT = float(os.environ.get("I_SPONT", 0.16))
DIR_ANGLES = (-np.pi + np.arange(N_DIR) * 2 * np.pi / N_DIR).astype(np.float32)


W0 = 0.008
C_REF = 60.0
GATE_DEG = 30.0

A_PLUS = 0.03
A_MINUS = -0.03
TAU_PLUS = 50.0
TAU_MINUS = 50.0
STDP_W_MAX = 0.1
STDP_W_MIN = 0.0001
STDP_WINDOW_DEG = 15.0
STDP_GAIN = float(os.environ.get("STDP_GAIN", 3.0))
EXPLORED_RULE = os.environ.get("EXPLORED_RULE", "stdp").lower()
I_TAG = 1.2
EXPLORED_SCALE = float(os.environ.get("EXPLORED_SCALE", 27.0))
TAG_RATE = float(os.environ.get("TAG_RATE", 0.01))
W_CAP = STDP_W_MAX


DEMO_SPEED = 12.0
OBS_SPEED = 4.5
TICK_CM = 8.0 / 3.0
LEARN_SUBSTEPS = int(os.environ.get("LEARN_SUBSTEPS", 10))
DRIVE_EVERY = int(os.environ.get("DRIVE_EVERY", 5))
N_WORKERS = int(os.environ.get("N_WORKERS", min(os.cpu_count() or 1, 8)))
EXP2_MAX_TICKS = int(os.environ.get("EXP2_MAX_TICKS", 300))
EXP2_SUBSTEPS = int(os.environ.get("EXP2_SUBSTEPS", 3))
TIME_SCALE = float(os.environ.get("TIME_SCALE", (TICK_CM / DEMO_SPEED * 1000.0) / (STEPS_PER_TICK * LEARN_SUBSTEPS * DT)))
STEP_CM, N_SUB = 8.0, 3
WANDER_SD = 0.35
FOLLOW_SD = float(os.environ.get("FOLLOW_SD", 0.55))
EXPLORE_PROB = float(os.environ.get("EXPLORE_PROB", 0.15))
MAX_TIME = 300.0
SPEED_EXP2 = 12.0

OBSERVATION_TIMES = [20, 60, 100, 140]


AVOID_MARGIN = 3.0
DEMO_DETECT_R = float(os.environ.get("DEMO_DETECT_R", GOAL_R + AVOID_MARGIN))
NO_GIVE_UP_TIME = float(os.environ.get("NO_GIVE_UP_TIME", 3000.0))


V_REF = 2.5
SELF_MARK = True
SELF_RATE = float(os.environ.get("SELF_RATE", 1.0))
KERNEL_SD = 30.0
NOV_EPS = float(os.environ.get("NOV_EPS", 0.05))
KERNEL_EXP = float(os.environ.get("KERNEL_EXP", 1.0))
KAPPA_PERSIST = 8.0
BETA_NOVEL = 40.0


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


N_SOCIAL = N_PLACE
SOC_SIGMA = SIGMA
SOC_FIELD_R = PLACE_FIELD_R
I_SOCIAL = I_PLACE
SOC_REST, SOC_THR, SOC_R, SOC_C = PL_REST, PL_THR, PL_R, PL_C
SOC2PLACE_W = 1.0
SOC2PLACE = np.arange(N_SOCIAL)
SOC2PLACE_IS_IDENTITY = bool(np.array_equal(SOC2PLACE, np.arange(N_PLACE)))


def social_centres(centres):
    return centres.copy()


def social_current(pos, centres_s):
    d2 = ((centres_s[:, None, :, :] - pos[:, :, None, :]) ** 2).sum(-1)
    act = np.exp(-d2 / (2 * SOC_SIGMA ** 2)) * (d2 <= SOC_FIELD_R ** 2)
    return (I_SOCIAL * act).astype(np.float32)


def _relay(x):
    out = np.zeros_like(x)
    out[..., SOC2PLACE] = SOC2PLACE_W * x
    return out


def social_window(v, ref, I, nsteps):
    counts = np.zeros(v.shape, np.float32)
    for _ in range(nsteps):
        counts += lif_step(v, ref, I, SOC_REST, SOC_THR, SOC_R, SOC_C)
    return SOC2PLACE_W * counts if SOC2PLACE_IS_IDENTITY else _relay(counts)


def social_raster(v, ref, I, nsteps):
    out = np.zeros((v.shape[0], nsteps, v.shape[-1]), np.float32)
    for k in range(nsteps):
        sp = lif_step(v, ref, I, SOC_REST, SOC_THR, SOC_R, SOC_C)[:, 0]
        if SOC2PLACE_IS_IDENTITY:
            out[:, k] = SOC2PLACE_W * sp
        else:
            out[:, k, SOC2PLACE] = SOC2PLACE_W * sp
    return out


def lif_step(v, ref, I, rest, thr, r, c):
    v += (DT / (r * c)) * (-(v - rest) + r * I)
    inref = ref > 0
    v[inref] = rest
    ref[inref] -= 1
    s = v >= thr
    v[s] = rest
    ref[s] = REFRAC_STEPS
    return s


def place_current(pos, centres):
    d2 = ((centres[:, None, :, :] - pos[:, :, None, :]) ** 2).sum(-1)
    act = np.exp(-d2 / (2 * SIGMA ** 2)) * (d2 <= PLACE_FIELD_R ** 2)
    return (I_PLACE * act).astype(np.float32)


def place_window(v, ref, I, nsteps):
    counts = np.zeros(v.shape, np.float32)
    for _ in range(nsteps):
        counts += lif_step(v, ref, I, PL_REST, PL_THR, PL_R, PL_C)
    return counts


def direction_window(st, drive, nsteps, rng):
    cE = np.zeros(st["vE"].shape, np.float32)
    for _ in range(nsteps):
        noise = rng.triangular(0.01, 0.03, 0.08, st["vE"].shape).astype(np.float32)
        Iexc = drive + noise - W_IE * st["tr_i"][..., None]
        sE = lif_step(st["vE"], st["rE"], Iexc, DR_REST, DR_THR, DR_R, DR_C)
        st["tr_e"] = st["tr_e"] * SYN_DECAY + W_EI * sE.sum(-1)
        Iinh = st["tr_e"][..., None] * np.ones(1, np.float32)
        sI = lif_step(st["vI"], st["rI"], Iinh, DR_REST, DR_THR, DR_R, DR_C)
        st["tr_i"] = st["tr_i"] * SYN_DECAY + sI.sum(-1)
        cE += sE
    return cE


def new_dir_state(shape):
    G, K = shape
    return {
        "vE": np.full((G, K, N_DIR), DR_REST, np.float32), "rE": np.zeros((G, K, N_DIR), np.int8),
        "vI": np.full((G, K, N_DIR), DR_REST, np.float32), "rI": np.zeros((G, K, N_DIR), np.int8),
        "tr_e": np.zeros((G, K), np.float32), "tr_i": np.zeros((G, K), np.float32),
    }


def make_centres(G, rng):
    ang = rng.uniform(0, 2 * np.pi, (G, N_PLACE))
    rad = np.sqrt(rng.uniform(0, 1, (G, N_PLACE))) * R_ARENA
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1).astype(np.float32)


def natural_demonstrator_path(rng):
    n_ticks = int(max(OBSERVATION_TIMES) * DEMO_SPEED / TICK_CM)
    pos = DEMO_START.astype(np.float64).copy()
    heading = rng.uniform(-np.pi, np.pi)
    pts = [pos.copy()]
    for _ in range(n_ticks):
        heading += rng.normal(0, 0.45)
        new = pos + TICK_CM * np.array([np.cos(heading), np.sin(heading)])
        if np.hypot(*new) > WALL_R:
            heading = np.arctan2(-pos[1], -pos[0]) + rng.normal(0, 0.4)
            new = pos
        pos = new
        pts.append(pos.copy())
        if np.hypot(*(pos - GOAL)) <= DEMO_DETECT_R:
            while np.hypot(*(pos - GOAL)) > GOAL_R:
                d = GOAL - pos
                pos = pos + d / np.hypot(*d) * min(TICK_CM, np.hypot(*d))
                pts.append(pos.copy())
            return np.array(pts, np.float32), (len(pts) - 1) * TICK_CM / DEMO_SPEED
    return np.array(pts, np.float32), np.inf


def pad_paths(paths):
    lengths = np.array([len(p) for p in paths])
    T = lengths.max()
    out = np.zeros((len(paths), T, 2), np.float32)
    for i, p in enumerate(paths):
        out[i, :len(p)] = p
        out[i, len(p):] = p[-1]
    return out, lengths


def place_raster(v, ref, I, nsteps):
    out = np.zeros((v.shape[0], nsteps, v.shape[-1]), np.float32)
    for k in range(nsteps):
        out[:, k] = lif_step(v, ref, I, PL_REST, PL_THR, PL_R, PL_C)[:, 0]
    return out


def stdp_learn_from_paths(paths, lengths, centres, eta, W_prev=None, learn_mask=None, seed_off=2, substeps=None, social=False):
    G, T, _ = paths.shape
    substeps = substeps or LEARN_SUBSTEPS
    S = STEPS_PER_TICK * substeps
    tscale = TIME_SCALE * LEARN_SUBSTEPS / substeps
    vP = np.full((G, 1, N_PLACE), PL_REST, np.float32)
    rP = np.zeros((G, 1, N_PLACE), np.int8)
    st = new_dir_state((G, 1))
    syn = np.zeros((G, N_PLACE), np.float32)
    x_pre = np.zeros((G, N_PLACE), np.float32)
    x_post = np.zeros((G, N_DIR), np.float32)
    dW = np.zeros((G, N_DIR, N_PLACE), np.float32)
    if learn_mask is None:
        learn_mask = np.ones(G, bool)
    base = W_prev if W_prev is not None else np.full((G, N_DIR, N_PLACE), W0, np.float32)
    if not learn_mask.any():
        return np.clip(base, STDP_W_MIN, STDP_W_MAX).astype(np.float32)
    Wt = np.ascontiguousarray(weight_readout(base).transpose(0, 2, 1))
    T = min(T, int(lengths[learn_mask].max()) + 1)
    lrng = np.random.default_rng(None if SEED is None else SEED + seed_off)
    dec_p, dec_m = np.float32(np.exp(-DT / TAU_PLUS)), np.float32(np.exp(-DT / TAU_MINUS))
    syn_dec = np.float32(SYN_DECAY)
    one = np.ones(1, np.float32)
    half = np.deg2rad(STDP_WINDOW_DEG)
    gate_rad = np.deg2rad(GATE_DEG)

    for t in range(1, T):
        pos = paths[:, t]
        step = pos - paths[:, t - 1]
        heading = np.arctan2(step[:, 1], step[:, 0])
        goal_dir = np.arctan2(GOAL[1] - pos[:, 1], GOAL[0] - pos[:, 0])
        theta_D = wrap(heading - goal_dir)
        valid = (t < lengths) & learn_mask & (np.hypot(*(pos - GOAL).T) > GOAL_R) \
            & (np.hypot(step[:, 0], step[:, 1]) > 1e-6)

        if social:
            pre = social_raster(vP, rP, social_current(pos[:, None, :], social_centres(centres)), S)
        else:
            pre = place_raster(vP, rP, place_current(pos[:, None, :], centres), S)
        post = np.zeros((G, S, N_DIR), np.float32)
        base_I = lrng.triangular(0.01, 0.03, 0.08, (S, G, 1, N_DIR)).astype(np.float32) + np.float32(I_SPONT)
        for k in range(S):
            syn = syn * syn_dec + pre[:, k]
            if k % DRIVE_EVERY == 0:
                drive = np.float32(G_SYN) * np.matmul(syn[:, None, :], Wt)
            Iexc = base_I[k] + drive - np.float32(W_IE) * st["tr_i"][..., None]
            sE = lif_step(st["vE"], st["rE"], Iexc, DR_REST, DR_THR, DR_R, DR_C)
            st["tr_e"] = st["tr_e"] * syn_dec + np.float32(W_EI) * sE.sum(-1)
            Iinh = st["tr_e"][..., None] * one
            sI = lif_step(st["vI"], st["rI"], Iinh, DR_REST, DR_THR, DR_R, DR_C)
            st["tr_i"] = st["tr_i"] * syn_dec + sI.sum(-1)
            post[:, k] = sE[:, 0]

        x_pre_h = np.empty((G, S, N_PLACE), np.float32)
        x_post_h = np.empty((G, S, N_DIR), np.float32)
        for k in range(S):
            x_pre *= dec_p
            x_post *= dec_m
            x_pre_h[:, k] = x_pre
            x_post_h[:, k] = x_post
            x_pre += pre[:, k]
            x_post += post[:, k]

        if not valid.any():
            continue
        diff = wrap(DIR_ANGLES[None] - goal_dir[:, None])
        in_win = (diff >= -half) & (diff < half)
        out_win = ~((diff > -half) & (diff <= half))
        f_ltp = (in_win * (np.abs(theta_D) <= gate_rad)[:, None]
                 * (1.1 - np.abs(np.rad2deg(theta_D)) / 30.0)[:, None] * valid[:, None]).astype(np.float32)
        f_ltd = (out_win * (np.abs(theta_D) > gate_rad)[:, None]
                 * (np.abs(np.rad2deg(theta_D)) / 180.0)[:, None] * valid[:, None]).astype(np.float32)
        ltp = np.matmul(post.transpose(0, 2, 1), x_pre_h)
        ltd = np.matmul(x_post_h.transpose(0, 2, 1), pre)
        dW += tscale * eta * (A_PLUS * f_ltp[:, :, None] * ltp + A_MINUS * f_ltd[:, :, None] * ltd)

    return np.clip(base + dW, STDP_W_MIN, STDP_W_MAX).astype(np.float32)


def _stdp_chunk(job):
    return stdp_learn_from_paths(*job)


def _trim_paths(paths, lengths, learn_mask, max_ticks):
    G = paths.shape[0]
    segs = []
    for g in range(G):
        if learn_mask[g]:
            L = int(lengths[g])
            segs.append(paths[g, max(0, L - max_ticks - 1):L])
        else:
            segs.append(paths[g, :1])
    return pad_paths(segs)


def learn_from_paths(paths, lengths, centres, eta, W_prev=None, learn_mask=None, max_ticks=None, substeps=None, social=False):
    G = paths.shape[0]
    if learn_mask is None:
        learn_mask = np.ones(G, bool)
    if max_ticks is not None:
        paths, lengths = _trim_paths(paths, lengths, learn_mask, max_ticks)
    nw = min(N_WORKERS, int(learn_mask.sum()))
    if nw <= 1:
        return stdp_learn_from_paths(paths, lengths, centres, eta, W_prev, learn_mask, substeps=substeps, social=social)
    order = np.argsort(lengths)
    parts = [i for i in np.array_split(order, nw) if len(i)]
    jobs = [(paths[i][:, :int(lengths[i].max())], lengths[i], centres[i], eta,
             None if W_prev is None else W_prev[i], learn_mask[i], 2 + k, substeps, social)
            for k, i in enumerate(parts)]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        out = pool.map(_stdp_chunk, jobs)
    W = np.empty((G, N_DIR, N_PLACE), np.float32)
    for i, o in zip(parts, out):
        W[i] = o
    return W


def stdp_explored_from_paths(paths, lengths, centres, mask):
    G, T, _ = paths.shape
    S = STEPS_PER_TICK
    vP = np.full((G, 1, N_PLACE), PL_REST, np.float32)
    rP = np.zeros((G, 1, N_PLACE), np.int8)
    vT = np.full((G, 1, 1), DR_REST, np.float32)
    rT = np.zeros((G, 1, 1), np.int8)
    x_pre = np.zeros((G, N_PLACE), np.float32)
    w = np.full((G, N_PLACE), W0, np.float32)
    dec_p = np.exp(-DT / TAU_PLUS)
    for t in range(1, T):
        pos = paths[:, t]
        moved = (t < lengths) & mask & (np.hypot(*(pos - paths[:, t - 1]).T) > 1e-6)
        pre = social_raster(vP, rP, social_current(pos[:, None, :], social_centres(centres)), S)
        It = (I_TAG * moved)[:, None, None].astype(np.float32)
        post = np.zeros((G, S), np.float32)
        x_pre_h = np.empty((G, S, N_PLACE), np.float32)
        for k in range(S):
            x_pre *= dec_p
            x_pre_h[:, k] = x_pre
            post[:, k] = lif_step(vT, rT, It, DR_REST, DR_THR, DR_R, DR_C)[:, 0, 0]
            x_pre += pre[:, k]
        if moved.any():
            w += TAG_RATE * A_PLUS * np.matmul(post[:, None, :], x_pre_h)[:, 0] * moved[:, None]
            w = np.clip(w, STDP_W_MIN, STDP_W_MAX)
    return (EXPLORED_SCALE * np.clip(w - W0, 0.0, None) / (STDP_W_MAX - W0)).astype(np.float32)


def learn_explored_from_paths(paths, lengths, centres, mask):
    if EXPLORED_RULE == "stdp":
        return stdp_explored_from_paths(paths, lengths, centres, mask)
    G, T, _ = paths.shape
    vP = np.full((G, 1, N_PLACE), PL_REST, np.float32)
    rP = np.zeros((G, 1, N_PLACE), np.int8)
    V = np.zeros((G, N_PLACE), np.float32)
    for t in range(1, T):
        pos = paths[:, t]
        moved = (t < lengths) & mask & (np.hypot(*(pos - paths[:, t - 1]).T) > 1e-6)
        sP = social_window(vP, rP, social_current(pos[:, None, :], social_centres(centres)), STEPS_PER_TICK)[:, 0]
        V += sP * moved[:, None]
    return V


def weight_readout(W):
    return np.clip(W0 + STDP_GAIN * (W - W0), 0.0, None).astype(np.float32)


def navigate(W, centres, starts, rng, speed, max_time, V_demo=None, give_up=None, use_novelty=None):
    G, K, _ = starts.shape
    Wt = np.ascontiguousarray(weight_readout(W).transpose(0, 2, 1))
    pos = starts.astype(np.float32).copy()
    heading = rng.uniform(-np.pi, np.pi, (G, K)).astype(np.float32)
    vP = np.full((G, K, N_PLACE), PL_REST, np.float32)
    rP = np.zeros((G, K, N_PLACE), np.int8)
    st = new_dir_state((G, K))
    done = np.zeros((G, K), bool)
    latency = np.full((G, K), np.nan)
    steps = np.zeros((G, K), int)
    sub = STEP_CM / N_SUB
    paths = [pos.copy()]
    n_dec = int(max_time * speed / STEP_CM)
    explore = V_demo is not None
    gu_n = None if give_up is None else np.floor(np.asarray(give_up) * speed / STEP_CM).astype(np.int64)
    frng = np.random.default_rng(None if SEED is None else SEED + 1)
    if explore:
        V = np.repeat(V_demo[:, None, :], K, axis=1).astype(np.float32)
        half_bin = np.pi / N_DIR

    for k in range(n_dec):
        if gu_n is not None:
            newly = (k >= gu_n)[:, None] & ~done
            steps[newly] = k * N_SUB
            done |= newly
        if done.all():
            break
        cP = place_window(vP, rP, place_current(pos, centres), WARMUP_STEPS if k == 0 else STEPS_PER_TICK)
        drive = (G_SYN * np.matmul(cP, Wt)).astype(np.float32)
        cE = direction_window(st, drive, STEPS_PER_TICK, rng)
        fired = cE.sum(-1) > 0
        j = np.argmax(cE + 0.01 * rng.random(cE.shape), -1)
        if explore:
            if SELF_MARK:
                V += SELF_RATE * cP * (~done)[..., None]
            wander = _novelty_heading(pos, heading, centres, V, rng, half_bin)
            if use_novelty is not None and not use_novelty.all():
                wander = np.where(use_novelty[:, None], wander, heading + rng.normal(0, WANDER_SD, (G, K)))
        else:
            wander = heading + rng.normal(0, WANDER_SD, (G, K))
        follow = DIR_ANGLES[j] + frng.normal(0, FOLLOW_SD, (G, K))
        detour = frng.random((G, K)) < EXPLORE_PROB
        heading = np.where(fired & ~detour, follow, wander).astype(np.float32)

        for s in range(N_SUB):
            alive = ~done
            pos[..., 0] += sub * np.cos(heading) * alive
            pos[..., 1] += sub * np.sin(heading) * alive
            r = np.hypot(pos[..., 0], pos[..., 1])
            out = r > WALL_R
            if out.any():
                pos[out] *= (WALL_R / r[out])[:, None]
                heading[out] = np.arctan2(-pos[..., 1][out], -pos[..., 0][out]) + rng.normal(0, 0.4, out.sum())
            reached = alive & (np.hypot(pos[..., 0] - GOAL[0], pos[..., 1] - GOAL[1]) <= GOAL_R)
            n_sub_done = k * N_SUB + s + 1
            latency[reached] = n_sub_done * sub / speed
            steps[reached] = n_sub_done
            done |= reached
            paths.append(pos.copy())
    steps[~done] = len(paths) - 1
    return latency, np.array(paths), steps


def _novelty_heading(pos, heading, centres, V, rng, half_bin):
    G, K, _ = pos.shape
    d = centres[:, None, :, :] - pos[:, :, None, :]
    dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
    if KERNEL_EXP > 0:
        ker = np.exp(-dist / KERNEL_SD)
    else:
        ker = np.exp(-dist ** 2 / (2 * KERNEL_SD ** 2))
    w = ker * np.exp(-V / V_REF)
    sx = (w * d[..., 0] / dist).sum(-1)
    sy = (w * d[..., 1] / dist).sum(-1)
    if NOV_EPS > 0:
        pull = np.hypot(sx, sy) / (w.sum(-1) + NOV_EPS * ker.sum(-1))
    else:
        pull = np.hypot(sx, sy) / ker.sum(-1)
    ang = np.arctan2(sy, sx)
    logit = KAPPA_PERSIST * np.cos(DIR_ANGLES[None, None] - heading[..., None]) \
        + BETA_NOVEL * pull[..., None] * np.cos(DIR_ANGLES[None, None] - ang[..., None])
    pr = np.exp(logit - logit.max(-1, keepdims=True))
    pr /= pr.sum(-1, keepdims=True)
    idx = np.minimum((pr.cumsum(-1) < rng.random((G, K, 1))).sum(-1), N_DIR - 1)
    return DIR_ANGLES[idx] + rng.uniform(-half_bin, half_bin, (G, K))


def start_positions():
    g = np.arange(-20, 21, 10)
    pts = np.array([(x, y) for x in g for y in g], np.float32)
    order = np.argsort(np.hypot(*(pts - GOAL).T))
    return pts[order]


def experiment1(rng):
    starts = start_positions()
    centres = make_centres(N_TRIALS, rng)
    full = [natural_demonstrator_path(rng) for _ in range(N_TRIALS)]
    t_find = np.array([f[1] for f in full])
    print("  demonstrator find times [s]: "
          + str([int(round(t)) if np.isfinite(t) else f">{max(OBSERVATION_TIMES)}" for t in np.sort(t_find)]))
    print("  trials in which the demonstrator has found the goal by "
          + ", ".join(f"{T} s: {int((t_find <= T).sum())}/{N_TRIALS}" for T in OBSERVATION_TIMES))
    res = {}
    for T in OBSERVATION_TIMES:
        t0 = time.time()
        n = int(T * DEMO_SPEED / TICK_CM)
        demo = [f[0][:n + 1] for f in full]
        found = np.array([len(d) - 1 <= n and np.hypot(*(d[-1] - GOAL)) <= GOAL_R for d in demo])
        paths, lengths = pad_paths(demo)

        W = learn_from_paths(paths, lengths, centres, eta=1.0, learn_mask=found, social=True)
        V = None if found.all() else learn_explored_from_paths(paths, lengths, centres, ~found)
        if found.all():
            give_up, max_time = None, MAX_TIME
        elif not found.any():
            give_up, max_time = None, NO_GIVE_UP_TIME
        else:
            give_up, max_time = np.where(found, MAX_TIME, NO_GIVE_UP_TIME), NO_GIVE_UP_TIME
        lat, tr, steps = navigate(W, centres, np.tile(starts, (N_TRIALS, 1, 1)), rng, OBS_SPEED, max_time,
                                  V_demo=V, give_up=give_up, use_novelty=~found)
        cap = np.repeat(np.where(found, MAX_TIME, NO_GIVE_UP_TIME)[:, None], starts.shape[0], axis=1)
        res[T] = dict(lat=lat, W=W, V=V, centres=centres, demo=demo, tr=tr, steps=steps,
                      found=found, cap=cap)
        print(f"  {T:>3d} s observation (goal seen in {int(found.sum())}/{N_TRIALS} trials): "
              f"success {np.mean(~np.isnan(lat)) * 100:5.1f} %, "
              f"mean latency {np.where(np.isnan(lat), cap, lat).mean():6.1f} s  ({time.time() - t0:.0f} s compute)")
    return starts, res


def draw_arena(ax):
    ax.add_patch(plt.Circle((0, 0), R_ARENA, fill=False, lw=2))
    for i in range(N_HOLES):
        a = 2 * np.pi * i / N_HOLES
        ax.add_patch(plt.Circle((HOLE_RING_R * np.cos(a), HOLE_RING_R * np.sin(a)), HOLE_D / 2,
                                color="red" if i == GOAL_INDEX else "gray", alpha=0.7))
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect("equal")
    ax.set_xlabel("X [cm]"); ax.set_ylabel("Y [cm]"); ax.grid(alpha=0.3)


def typical_run(res):
    lat = {T: np.where(np.isnan(res[T]["lat"]), res[T]["cap"], res[T]["lat"]) for T in OBSERVATION_TIMES}
    score = sum(np.abs(np.log((lat[T] + 1) / (np.median(lat[T]) + 1))) for T in OBSERVATION_TIMES)
    g, k = np.unravel_index(np.argmin(score), score.shape)
    return int(g), int(k)


def plot_experiment1(starts, res, tag):
    colors = {20: "tab:blue", 60: "tab:orange", 100: "tab:green", 140: "tab:red"}
    idx = np.arange(1, len(starts) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
    for T in OBSERVATION_TIMES:
        lat, cap = res[T]["lat"], res[T]["cap"]
        ok = ~np.isnan(lat)
        mean_lat = np.where(ok, lat, cap).mean(0)
        axs[0].plot(idx, mean_lat, "o-", ms=4, color=colors[T], label=f"{T} s")
        axs[1].plot(idx, ok.mean(0) * 100, "s-", ms=4, color=colors[T], label=f"{T} s")
    axs[0].set_xlabel("Initial position of observer (sorted by distance to goal)")
    axs[0].set_ylabel("Latency [s]"); axs[0].set_title(f"Latency to goal (n={N_TRIALS})")
    axs[1].set_xlabel("Initial position of observer (sorted by distance to goal)")
    axs[1].set_ylabel("Success rate [%]"); axs[1].set_ylim(0, 105)
    axs[1].set_title("Goal reached (no give-up if goal not seen)")
    for a in axs[:2]:
        a.grid(alpha=0.3); a.legend()
    means = [np.where(np.isnan(res[T]["lat"]), res[T]["cap"], res[T]["lat"]).mean() for T in OBSERVATION_TIMES]
    axs[2].bar([f"{T} s" for T in OBSERVATION_TIMES], means, color=[colors[T] for T in OBSERVATION_TIMES])
    for i, m in enumerate(means):
        axs[2].text(i, m, f"{m:.0f}", ha="center", va="bottom")
    axs[2].set_ylabel("Mean latency [s]"); axs[2].set_title("Mean latency by observation window")
    fig.suptitle(f"Scenario: {tag}")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/exp1_{tag}_latency_success.png", dpi=150)


    g0, s0 = typical_run(res)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    for a, T in zip(axs, OBSERVATION_TIMES):
        r = res[T]
        lat = np.where(np.isnan(r["lat"][:, s0]), r["cap"][:, s0], r["lat"][:, s0])
        g = g0
        tr, n = r["tr"][:, g, s0], r["steps"][g, s0]
        draw_arena(a)
        a.plot(tr[:n + 1, 0], tr[:n + 1, 1], "b-", lw=0.8)
        a.scatter(*tr[0], c="black", s=40, zorder=4)
        a.set_title(f"Observer after {T} s observation\n"
                    + ("goal found in %.0f s" % lat[g] if lat[g] < r["cap"][g, s0] else "no goal found (%.0f s)" % r["cap"][g, s0]))
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/exp1_{tag}_observer_paths.png", dpi=150)


    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    for col, T in enumerate(OBSERVATION_TIMES):
        r = res[T]
        cen = r["centres"][g0]
        ax = axs[0, col]; draw_arena(ax)
        if r["V"] is not None and not r["found"][g0]:
            ex = 1 - np.exp(-r["V"][g0] / V_REF)
            sc = ax.scatter(cen[:, 0], cen[:, 1], c=ex, cmap="viridis", vmin=0, vmax=1, s=14)
            p = r["demo"][g0]; ax.plot(p[:, 0], p[:, 1], "r-", lw=0.5, alpha=0.6)
            ax.set_title(f"{T} s: goal NOT seen\nplace cells tagged 'explored' (yellow)")
        else:
            ax.set_title(f"{T} s: goal seen - no 'nothing here' memory")
        ax2 = axs[1, col]; draw_arena(ax2)
        W = r["W"][g0]
        learned = W.max(0) > 0.5 * W_CAP
        pref = DIR_ANGLES[W.argmax(0)]
        if learned.any():
            ax2.quiver(cen[learned, 0], cen[learned, 1], np.cos(pref[learned]), np.sin(pref[learned]),
                       angles="xy", scale=25, width=0.004)
        ax2.set_title(f"{T} s: {learned.sum()} cells learned a goal direction")
    fig.suptitle(f"Scenario: {tag}")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/exp1_{tag}_learned_memory.png", dpi=150)


    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    for a, T in zip(axs, OBSERVATION_TIMES):
        draw_arena(a)
        p = res[T]["demo"][g0]
        a.plot(p[:, 0], p[:, 1], "r-", lw=0.8)
        a.scatter(*p[0], c="darkred", s=30)
        a.set_title(f"Demonstrator, first {T} s ({'reaches goal' if res[T]['found'][g0] else 'goal not reached'})")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/exp1_{tag}_demonstrator_paths.png", dpi=150)

    s_pick = np.array([s0])
    cols = plt.cm.tab10(np.arange(len(s_pick)))
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.5))
    for c, T in enumerate(OBSERVATION_TIMES):
        r = res[T]
        a = axs[0, c]; draw_arena(a)
        p = r["demo"][g0]
        a.plot(p[:, 0], p[:, 1], "r-", lw=0.8)
        a.scatter(*p[0], c="darkred", s=40, zorder=4, label="Demonstrator start")
        if r["found"][g0]:
            a.scatter(*p[-1], marker="*", c="gold", edgecolor="k", s=180, zorder=5, label="Goal reached")
        a.set_title(f"Demonstrator, first {T} s\n" + ("goal reached" if r["found"][g0] else "goal NOT reached"))
        a.legend(loc="lower right", fontsize=7)
        a = axs[1, c]; draw_arena(a)
        for col, s_i in zip(cols, s_pick):
            n = r["steps"][g0, s_i]
            tr = r["tr"][:n + 1, g0, s_i]
            a.plot(tr[:, 0], tr[:, 1], "-", lw=1.2, color="tab:blue")
            a.scatter(*tr[0], color="black", s=45, zorder=4, label="Observer start")
            if not np.isnan(r["lat"][g0, s_i]):
                a.scatter(*tr[-1], marker="*", color="gold", edgecolor="k", s=180, zorder=5, label="Goal found")
        ok = ~np.isnan(r["lat"][g0, s_pick])
        a.set_title(f"Observer after {T} s observation\n"
                    + (f"goal found after {r['lat'][g0, s_pick[0]]:.0f} s" if ok[0] else "goal not found"), fontsize=10)
        a.legend(loc="lower right", fontsize=7)
    fig.suptitle(f"Scenario: {tag} - trial {g0}, the most typical observer run (closest to the median latency of every window)")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/exp1_{tag}_demonstrator_and_observer.png", dpi=150)
    if matplotlib.get_backend().lower() == "agg":
        plt.close("all")


def paths_of(tr, steps, g):
    n = int(steps[g, 0]) + 1
    return tr[:n, g, 0]


def experiment2(eta, n_pairs, rng, n_trials=5, max_time=600.0):
    centres = make_centres(n_pairs, rng)
    Wd = np.full((n_pairs, N_DIR, N_PLACE), W0, np.float32)
    Wo = Wd.copy()
    d_start = np.tile(DEMO_START, (n_pairs, 1, 1))
    o_start = d_start.copy()
    lat_d = np.zeros((n_trials, n_pairs)); lat_o = np.zeros((n_trials, n_pairs))

    for trial in range(n_trials):
        ld, trd, std = navigate(Wd, centres, d_start, rng, SPEED_EXP2, max_time)
        dpaths, dlen = pad_paths([paths_of(trd, std, g) for g in range(n_pairs)])
        found = ~np.isnan(ld[:, 0])
        Wo = learn_from_paths(dpaths, dlen, centres, eta, W_prev=Wo, learn_mask=found, max_ticks=EXP2_MAX_TICKS, substeps=EXP2_SUBSTEPS, social=True)
        Wd = learn_from_paths(dpaths, dlen, centres, eta, W_prev=Wd, learn_mask=found, max_ticks=EXP2_MAX_TICKS, substeps=EXP2_SUBSTEPS)
        lo, tro, sto = navigate(Wo, centres, o_start, rng, SPEED_EXP2, max_time)
        opaths, olen = pad_paths([paths_of(tro, sto, g) for g in range(n_pairs)])
        Wo = learn_from_paths(opaths, olen, centres, eta, W_prev=Wo, learn_mask=~np.isnan(lo[:, 0]), max_ticks=EXP2_MAX_TICKS, substeps=EXP2_SUBSTEPS)
        print(f"    eta={eta}, trial {trial + 1}/{n_trials} done ({time.strftime('%H:%M:%S')})", flush=True)
        lat_d[trial] = np.where(found, ld[:, 0], max_time)
        lat_o[trial] = np.where(~np.isnan(lo[:, 0]), lo[:, 0], max_time)
    return lat_d, lat_o


def plot_experiment2(noinh, inh):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 2]})

    d, o = noinh[0][0], noinh[1][0]
    bp = axs[0].boxplot([d, o], showmeans=True, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["tab:blue", "tab:orange"]):
        patch.set_facecolor(c)
    p = mannwhitneyu(d, o).pvalue
    axs[0].set_xticks([1, 2]); axs[0].set_xticklabels(["Demonstrator", "Observer"])
    axs[0].set_title(f"No inhibition, trial 1 (n={len(d)})  MWU p={p:.3g}")
    axs[0].set_ylabel("Latency [s]"); axs[0].set_ylim(0, 620)

    nt = inh[0].shape[0]
    for t in range(nt):
        for k, (data, c) in enumerate([(inh[0][t], "tab:blue"), (inh[1][t], "tab:orange")]):
            b = axs[1].boxplot(data, positions=[t + 1 + (k - 0.5) * 0.35], widths=0.3, showmeans=True,
                               patch_artist=True)
            b["boxes"][0].set_facecolor(c)
        pv = mannwhitneyu(inh[0][t], inh[1][t]).pvalue
        axs[1].text(t + 1, 600, "*" if pv < 0.05 else "n.s.", ha="center")
    axs[1].set_xticks(range(1, nt + 1)); axs[1].set_xlabel("Trial")
    axs[1].set_title(f"Learning rate x0.1 (n={inh[0].shape[1]})  blue: demonstrator, orange: observer")
    axs[1].set_ylim(0, 620)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/exp2_inhibition.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"Goal hole {GOAL_INDEX} at {GOAL.round(1)} cm; trials per condition: {N_TRIALS}")

    if os.environ.get("RUN_EXP1", "1") != "0":
        tag = "random_find_time"
        print(f"Experiment 1 [{tag}]: the demonstrator explores at random and finds the goal whenever it happens to "
              f"(it notices the hole within {DEMO_DETECT_R:.0f} cm)")
        starts, res = experiment1(rng)
        plot_experiment1(starts, res, tag)
        lat = {T: np.where(np.isnan(res[T]["lat"]), res[T]["cap"], res[T]["lat"]).mean() for T in OBSERVATION_TIMES}
        print("  mean latency per window: " + ", ".join(f"{T}s={lat[T]:.0f}s" for T in OBSERVATION_TIMES))

    if os.environ.get("RUN_EXP2", "1") == "0":
        print(f"Figures saved in ./{OUTDIR}/")
        raise SystemExit(0)
    print("Experiment 2: no inhibition (eta=1) and inhibition (eta=0.1)")
    noinh = experiment2(1.0, N_PAIRS_NOINH, rng, n_trials=1)
    inh = experiment2(0.1, N_PAIRS_INH, rng, n_trials=5)
    print("  no inhibition, trial 1: demonstrator median %.0f s, observer median %.0f s"
          % (np.median(noinh[0][0]), np.median(noinh[1][0])))
    for t in range(5):
        print("  inhibition, trial %d: demonstrator median %.0f s, observer median %.0f s"
              % (t + 1, np.median(inh[0][t]), np.median(inh[1][t])))
    plot_experiment2(noinh, inh)

    print(f"Figures saved in ./{OUTDIR}/")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
