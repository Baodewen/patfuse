import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

PEMS_START = {
    "PEMS03": "2018-09-01T00:00",
    "PEMS04": "2018-01-01T00:00",
    "PEMS07": "2017-05-01T00:00",
    "PEMS08": "2016-07-01T00:00",
}
CAL_RANGE = np.array([60, 24, 7, 31, 366], dtype=np.float32)


def calendar_features(start, T):
    t = np.datetime64(start) + np.arange(T) * np.timedelta64(5, "m")
    h = t.astype("datetime64[h]")
    D = t.astype("datetime64[D]")
    minute = (t.astype("datetime64[m]") - h.astype("datetime64[m]")).astype(int)
    hour = (h - D.astype("datetime64[h]")).astype(int)
    dow = (D.astype(int) + 3) % 7
    dom = (D - D.astype("datetime64[M]").astype("datetime64[D]")).astype(int)
    doy = (D - D.astype("datetime64[Y]").astype("datetime64[D]")).astype(int)
    cal = np.stack([minute, hour, dow, dom, doy], 1).astype(np.float32)
    return cal / CAL_RANGE


def make_synthetic(T=3000, N=12, F=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    ph = rng.uniform(0, 2 * np.pi, (1, N))
    flow = 200 + 100 * np.sin(2 * np.pi * t / 288 + ph) + 30 * np.sin(2 * np.pi * t / 36 + 2 * ph)
    flow = flow + rng.normal(0, 8, (T, N))
    occ = flow / 2000 + rng.normal(0, 0.002, (T, N))
    spd = 65 - flow / 20 + rng.normal(0, 1, (T, N))
    return np.stack([flow, occ, spd][:F], -1).astype(np.float32)


class TrafficData:
    def __init__(self, raw, start, Th, Tf, device):
        T = raw.shape[0]
        self.Th, self.Tf, self.device = Th, Tf, device
        n_tr, n_va = int(0.6 * T), int(0.2 * T)
        self.bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, T)}
        tr = raw[:n_tr]
        self.mean = tr.reshape(-1, raw.shape[-1]).mean(0)
        self.std = tr.reshape(-1, raw.shape[-1]).std(0) + 1e-6
        self.X = torch.tensor((raw - self.mean) / self.std, dtype=torch.float32, device=device)
        self.M = torch.tensor(calendar_features(start, T), device=device)
        self.mu0, self.sd0 = float(self.mean[0]), float(self.std[0])
        self.N, self.F, self.K = raw.shape[1], raw.shape[2], self.M.shape[1]
        self.starts = {}
        for k, (lo, hi) in self.bounds.items():
            self.starts[k] = torch.arange(lo, hi - Th - Tf + 1, device=device)
        self.ar_h = torch.arange(Th, device=device)
        self.ar_f = torch.arange(Tf, device=device)

    def batch(self, split, idx):
        s = self.starts[split][idx]
        x = self.X[s[:, None] + self.ar_h]
        m = self.M[s[:, None] + self.ar_h]
        y = self.X[s[:, None] + self.Th + self.ar_f][..., 0]
        return x, m, y

    def n(self, split):
        return len(self.starts[split])


class MHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, q, k, v):
        B, Lq, d = q.shape
        Lk = k.shape[1]
        q = self.q(q).view(B, Lq, self.h, self.dk).transpose(1, 2)
        k = self.k(k).view(B, Lk, self.h, self.dk).transpose(1, 2)
        v = self.v(v).view(B, Lk, self.h, self.dk).transpose(1, 2)
        o = F_.scaled_dot_product_attention(q, k, v)
        return self.o(o.transpose(1, 2).reshape(B, Lq, d))


class PTIMLayer(nn.Module):

    def __init__(self, d, d_ff, heads, drop, use_pta=True, use_cgca=True):
        super().__init__()
        self.use_pta, self.use_cgca = use_pta, use_cgca
        self.sa, self.ca = MHA(d, heads), MHA(d, heads)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, H, V):
        B, N, L, d = H.shape
        if self.use_pta:
            h = H.reshape(B * N, L, d)
            h = self.ln1(h + self.drop(self.sa(h, h, h)))
            H = h.view(B, N, L, d)
        if self.use_cgca:
            G = H[:, :, -1]
            G = self.ln2(G + self.drop(self.ca(G, V, V)))
            H = torch.cat([H[:, :, :-1], G.unsqueeze(2)], 2)
        return self.ln3(H + self.drop(self.ffn(H)))


class APEFM(nn.Module):

    def __init__(self, Th, P, d, df, tau=1.0, variant="full", eps=1e-6):
        super().__init__()
        self.Th, self.P, self.S, self.Kf, self.df, self.tau, self.eps = Th, P, Th // P, Th // 2 + 1, df, tau, eps
        self.variant = variant
        self.phi_d = nn.Linear(1, df)
        self.selector = nn.Sequential(nn.Linear(df, df), nn.GELU(), nn.Linear(df, df))
        self.w_a = nn.Parameter(torch.tensor(1.0))
        self.b_a = nn.Parameter(torch.tensor(0.0))
        self.w_f = nn.Parameter(torch.randn(self.Kf - 1, df) * 0.1)
        self.beta_raw = nn.Parameter(torch.zeros(df))
        self.phi_f = nn.Linear(P, d)
        if variant == "static_correction":
            self.static_a = nn.Parameter(torch.tensor(0.0))
            self.static_logits = nn.Parameter(torch.zeros(self.Kf - 1))
        if variant == "wo_period":
            self.w_a = self.b_a = self.w_f = None

    def forward(self, xtar):
        B, Th, N = xtar.shape
        x = xtar.permute(0, 2, 1).reshape(B * N, Th, 1)
        Xm = self.phi_d(x)
        pi = torch.softmax(self.selector(Xm.mean(1)), -1)
        with torch.autocast(device_type=xtar.device.type, enabled=False):
            Phi = torch.fft.rfft((Xm * pi[:, None, :]).float(), dim=1)
            rho = Phi.abs().pow(2).sum(-1)
            rho_p = rho[:, 1:]
            k_star = rho_p.argmax(-1) + 1
            c = rho_p.gather(1, (k_star - 1)[:, None]).squeeze(1) / (rho_p.sum(1) + self.eps)
            ks = torch.arange(1, self.Kf, device=xtar.device)[None].float()
            if self.variant == "wo_period":
                Phi_e = Phi
            else:
                if self.variant == "static_correction":
                    alpha = torch.sigmoid(self.static_a).expand(B * N)
                    omega = torch.softmax(self.static_logits, -1)[None].expand(B * N, -1)
                else:
                    alpha = torch.sigmoid(self.w_a * c + self.b_a)
                    omega = torch.softmax(-(ks - k_star[:, None].float()).abs() / self.tau, -1)
                gain = 1 + alpha[:, None, None] * omega[:, :, None] * torch.tanh(self.w_f)[None]
                Phi_e = torch.cat([Phi[:, :1], Phi[:, 1:] * gain], 1)
            zq = torch.fft.irfft(Phi_e, n=Th, dim=1)
        beta = torch.sigmoid(self.beta_raw)
        beta = beta / (beta.sum() + self.eps)
        z = (zq * beta).sum(-1)
        Pf = self.phi_f(z.view(B * N, self.S, self.P))
        XF = torch.cat([Pf, Pf.mean(1, keepdim=True)], 1)
        return XF.view(B, N, self.S + 1, -1)


class DSFFN(nn.Module):

    def __init__(self, d, d_ff, drop):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(2 * d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.ln, self.drop = nn.LayerNorm(d), nn.Dropout(drop)

    def forward(self, X):
        g = X.mean(2, keepdim=True).expand_as(X)
        return self.ln(X + self.drop(self.mlp(torch.cat([self.local(X), g], -1))))


class PaTFuse(nn.Module):
    VARIANTS = ["full", "wo_apefm", "wo_ptim", "wo_pta", "wo_cgca", "concat_fusion", "wo_period",
                "wo_dsffn", "ptim_backbone", "shared_gate", "static_correction"]

    def __init__(self, F, K, Th=12, Tf=12, P=4, d=64, d_ff=64, df=None, R=2, heads=8, drop=0.1,
                 tau=1.0, variant="full"):
        super().__init__()
        assert Th % P == 0 and variant in self.VARIANTS
        self.Th, self.Tf, self.P, self.S, self.d, self.variant = Th, Tf, P, Th // P, d, variant
        self.L = self.S + 1
        df = df or d
        self.phi_p = nn.Linear(P, d)
        self.g0 = nn.Parameter(torch.randn(1, 1, 1, d) * 0.02)
        self.phi_tr = nn.Linear(Th, d)
        self.phi_cal = nn.Linear(Th, d)
        use_pta, use_cgca = variant != "wo_pta", variant != "wo_cgca"
        self.layers = nn.ModuleList([PTIMLayer(d, d_ff, heads, drop, use_pta, use_cgca) for _ in range(R)])
        self.apefm = APEFM(Th, P, d, df, tau, variant)
        self.g1, self.g2 = nn.Linear(d, d), nn.Linear(d, d)
        if variant == "concat_fusion":
            self.concat = nn.Linear(2 * d, d)
        if variant == "wo_dsffn":
            self.ffn_head = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
            self.ffn_ln, self.ffn_drop = nn.LayerNorm(d), nn.Dropout(drop)
        else:
            self.dsffn = DSFFN(d, d_ff, drop)
        self.proj = nn.Linear(self.L * d, Tf)

    def forward(self, X, M):
        B, Th, N, Fv = X.shape
        xtar = X[..., 0]
        patches = xtar.permute(0, 2, 1).reshape(B, N, self.S, self.P)
        H = torch.cat([self.phi_p(patches), self.g0.expand(B, N, 1, self.d)], 2)
        V = torch.cat([self.phi_tr(X.permute(0, 3, 2, 1).reshape(B, Fv * N, Th)),
                       self.phi_cal(M.permute(0, 2, 1))], 1)
        XT = H
        if self.variant != "wo_ptim":
            for layer in self.layers:
                XT = layer(XT, V)
        if self.variant == "ptim_backbone":
            out = XT
        else:
            if self.variant == "wo_apefm":
                Xg = XT
            else:
                XF = self.apefm(xtar)
                if self.variant == "concat_fusion":
                    Xg = self.concat(torch.cat([XT, XF], -1))
                else:
                    cand = torch.tanh(self.g1(XT))
                    gate = torch.sigmoid(self.g2(XF))
                    if self.variant == "shared_gate":
                        loc = gate[:, :, :self.S].mean(2, keepdim=True).expand(-1, -1, self.S, -1)
                        gate = torch.cat([loc, gate[:, :, self.S:]], 2)
                    Xg = cand * gate
            if self.variant == "wo_dsffn":
                out = self.ffn_ln(Xg + self.ffn_drop(self.ffn_head(Xg)))
            else:
                out = self.dsffn(Xg)
        y = self.proj(out.reshape(B, N, self.L * self.d))
        return y.permute(0, 2, 1)


@torch.no_grad()
def evaluate(model, data, split, bs=64):
    model.eval()
    Tf = data.Tf
    abs_h = torch.zeros(Tf, device=data.device)
    sq, cnt, mape, mcnt = 0.0, 0, 0.0, 0
    for i in range(0, data.n(split), bs):
        idx = torch.arange(i, min(i + bs, data.n(split)), device=data.device)
        x, m, y = data.batch(split, idx)
        p = model(x, m) * data.sd0 + data.mu0
        y = y * data.sd0 + data.mu0
        e = p - y
        abs_h += e.abs().sum((0, 2))
        sq += float((e ** 2).sum())
        cnt += e.numel()
        mask = y > 1e-3
        mape += float((e.abs()[mask] / y[mask]).sum())
        mcnt += int(mask.sum())
    per_step = abs_h / (cnt / Tf)
    return {"MAE": float(per_step.mean()), "RMSE": math.sqrt(sq / cnt), "MAPE": 100 * mape / max(mcnt, 1),
            "MAE_h": per_step.cpu().numpy()}


def build_model(data, args, d=None, d_ff=None, variant=None):
    return PaTFuse(data.F, data.K, args.Th, args.Tf, args.P, d or args.d, d_ff or args.d_ff, None, args.R,
                   args.heads, args.dropout, args.tau, variant or args.variant).to(data.device)


def fit(data, args, epochs, d=None, d_ff=None, variant=None, verbose=True):
    model = build_model(data, args, d, d_ff, variant)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = math.ceil(data.n("train") / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=epochs * steps, pct_start=0.3)
    best, best_mse, best_state, t_ep = float("inf"), float("inf"), None, []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(data.n("train"), device=data.device)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            x, m, y = data.batch("train", perm[i:i + args.batch])
            loss = F_.mse_loss(model(x, m), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            tot += float(loss) * len(x)
        if data.device.type == "cuda":
            torch.cuda.synchronize()
        t_ep.append(time.time() - t0)
        val = evaluate(model, data, "val")
        if val["MAE"] < best:
            best, best_mse, best_state = val["MAE"], val["RMSE"] ** 2, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"ep {ep + 1:3d}/{epochs} train_mse {tot / data.n('train'):.4f} val_MAE {val['MAE']:.3f} "
                  f"val_RMSE {val['RMSE']:.3f} ({t_ep[-1]:.1f}s)", flush=True)
    model.load_state_dict(best_state)
    return model, n_params, float(np.mean(t_ep)), best, best_mse


def bco_search(fitness, bounds, pop=6, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = np.array([b[0] for b in bounds], float), np.array([b[1] for b in bounds], float)
    X = lo + rng.random((pop, len(bounds))) * (hi - lo)
    fit = np.array([fitness(x) for x in X])
    for it in range(iters):
        b = X[fit.argmin()].copy()
        for i in range(pop):
            r1, r2 = rng.choice([j for j in range(pop) if j != i], 2, replace=False)
            t, mode = rng.random(), rng.integers(3)
            p0, p1, p2, p3 = X[i], X[r1], X[r2], b
            if mode == 0:
                c = (1 - t) * p0 + t * p3
            elif mode == 1:
                c = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p3
            else:
                c = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
            c = np.clip(c, lo, hi)
            fc = fitness(c)
            if fc < fit[i]:
                X[i], fit[i] = c, fc
        print(f"BCO iter {it + 1}/{iters} best {fit.min():.4f} at {X[fit.argmin()].round(1)}", flush=True)
    return X[fit.argmin()], fit.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PEMS04")
    ap.add_argument("--data_path", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--variant", default="full", choices=PaTFuse.VARIANTS)
    ap.add_argument("--Th", type=int, default=12)
    ap.add_argument("--Tf", type=int, default=12)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--d_ff", type=int, default=64)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--bco", action="store_true")
    ap.add_argument("--bco_pop", type=int, default=6)
    ap.add_argument("--bco_iters", type=int, default=30)
    ap.add_argument("--bco_epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    if args.synthetic:
        raw, start = make_synthetic(), PEMS_START["PEMS04"]
    else:
        raw, start = np.load(args.data_path)["data"].astype(np.float32), PEMS_START[args.dataset]
        if raw.ndim == 2:
            raw = raw[..., None]
    data = TrafficData(raw, start, args.Th, args.Tf, dev)
    print(f"data: T={raw.shape[0]} N={data.N} F={data.F} K={data.K} | windows "
          f"{data.n('train')}/{data.n('val')}/{data.n('test')}")

    d, d_ff = args.d, args.d_ff
    if args.bco:
        def fitness(x):
            dd, ff = int(round(x[0] / 8) * 8), int(round(x[1]))
            return fit(data, args, args.bco_epochs, dd, ff, verbose=False)[4]
        (bd, bf), _ = bco_search(fitness, [(32, 128), (32, 128)], args.bco_pop, args.bco_iters, args.seed)
        d, d_ff = int(round(bd / 8) * 8), int(round(bf))
        print(f"BCO result: d={d}, d_ff={d_ff}")

    model, n_params, t_ep, best_val, _ = fit(data, args, args.epochs, d, d_ff)
    res = evaluate(model, data, "test")
    print(f"\n[{args.variant}] params {n_params / 1e6:.3f}M | train {t_ep:.1f}s/epoch | best val MAE {best_val:.3f}")
    print(f"TEST MAE {res['MAE']:.3f} RMSE {res['RMSE']:.3f} MAPE {res['MAPE']:.2f}%")
    print("step-wise MAE:", np.round(res["MAE_h"], 3).tolist())


if __name__ == "__main__":
    main()
