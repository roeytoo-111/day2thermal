"""Generate all figures for the day2thermal algorithms PDF.
Run from the repo root:  CUDA_VISIBLE_DEVICES= python3 research/pdf_build/make_figs.py
"""
import os, re, random, json, sys
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 100, "savefig.dpi": 200})
C = {"blue": "#2b6cb0", "orange": "#dd6b20", "green": "#2f855a", "red": "#c53030", "grey": "#4a5568", "purple": "#6b46c1"}

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("wrote", name)

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.5, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    def box(x, y, w, h, text, fc="#ebf4ff", ec=C["blue"], fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color="#1a202c")
    def arrow(x0, y0, x1, y1, col=C["grey"]):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=col))
    box(1, 24, 15, 7, "paired video\nday.ts + thermal.ts", fc="#fefcbf", ec="#b7791f", fs=7.5)
    box(1, 14, 15, 7, "pre-registered\npair folders (RGB/, NIR/)", fc="#fefcbf", ec="#b7791f", fs=7.5)
    box(1, 4, 15, 7, "raw pairs,\nno registration", fc="#fefcbf", ec="#b7791f", fs=7.5)
    box(21, 24, 16, 7, "extract_frames\ntime sync, NUC drop", fs=7.5)
    box(41, 24, 15, 7, "register\nhomography, crop,\nchunk split", fs=7.5)
    box(21, 14, 35, 7, "prep_aligned\npair by name · common-size crop · session split", fs=7.5)
    box(21, 4, 35, 7, "prep_unpaired\ndownscale RGB · hard-link IR", fs=7.5)
    box(61, 12, 14, 19, "train\n\npix2pix\ntwo_stage\ncyclegan", fc="#c6f6d5", ec=C["green"], bold=True)
    box(80, 22, 18, 8, "infer\nany RGB folder →\nsynthetic IR PNGs", fc="#e9d8fd", ec=C["purple"], fs=7.5)
    box(80, 6, 18, 8, "eval /\ncheck_thermalness\nvs real IR hold-out", fc="#e9d8fd", ec=C["purple"], fs=7.5)
    arrow(16, 27.5, 21, 27.5); arrow(37, 27.5, 41, 27.5); arrow(56, 27.5, 61, 27.5)
    arrow(16, 17.5, 21, 17.5); arrow(56, 17.5, 61, 17.5); arrow(16, 7.5, 21, 7.5); arrow(56, 7.5, 61, 7.5)
    arrow(75, 26, 80, 26); arrow(75, 10, 80, 10)
    ax.text(68, 9.2, "→ checkpoints/latest.pt\n   samples/*.png, config.json", ha="center", fontsize=6.5, color=C["grey"], va="top")
    ax.text(28, 32.8, "→ train/{rgb,thermal}, val/{rgb,thermal}: identical filename = the pair", ha="left", fontsize=7.5, color=C["grey"], style="italic")
    save(fig, "fig_pipeline.png")

def fig_l1_median():
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax = axs[0]
    samples = np.concatenate([rng.normal(0.30, 0.04, 5000), rng.normal(0.72, 0.05, 5000)])
    ax.hist(samples, bins=80, color="#bee3f8", edgecolor="none", density=True, label="plausible IR values for ONE RGB input")
    m, med = samples.mean(), np.median(samples)
    ax.axvline(m, color=C["red"], lw=2, label=f"L2 optimum = mean = {m:.2f}")
    ax.axvline(med, color=C["orange"], lw=2, ls="--", label=f"L1 optimum = median = {med:.2f}")
    ax.axvline(0.30, color=C["green"], lw=1.5, ls=":", label="what a GAN can commit to (a mode)")
    ax.set_xlabel("target value (e.g. normalised temperature)"); ax.set_ylabel("density"); ax.set_title("One-to-many target: pixel losses land between the modes"); ax.legend(fontsize=7, loc="upper center")
    ax = axs[1]; x = np.arange(0, 100)
    p1 = 0.2 + 0.6 * (x > 40) - 0.15 * ((x > 60) & (x < 70)); p2 = 0.35 + 0.55 * (x > 44) - 0.15 * ((x > 62) & (x < 72))
    ax.plot(x, p1, color=C["blue"], lw=1.6, label="plausible answer 1 (cool ambient)"); ax.plot(x, p2, color=C["green"], lw=1.6, label="plausible answer 2 (warm ambient, slight offset)")
    ax.plot(x, (p1 + p2) / 2, color=C["red"], lw=2.2, ls="--", label="L1/L2-optimal prediction (their average)")
    ax.set_xlabel("pixel along a row"); ax.set_ylabel("value"); ax.set_title("Averaging incompatible answers softens edges and contrast"); ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig_l1_median.png")

def fig_gan_game():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))
    x = np.linspace(-4, 6, 500); pd = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi); pg = np.exp(-0.5 * (x - 2.0) ** 2) / np.sqrt(2 * np.pi)
    ax = axs[0]; ax.plot(x, pd, color=C["blue"], label="p_data"); ax.plot(x, pg, color=C["orange"], label="p_g (generator)")
    ax.plot(x, pd / (pd + pg), color=C["red"], ls="--", label="optimal D*(x) = p_data / (p_data + p_g)"); ax.set_title("Optimal discriminator for a fixed G"); ax.legend(fontsize=7); ax.set_xlabel("x")
    ax = axs[1]; d = np.linspace(0.001, 0.999, 500)
    ax.plot(d, np.log(1 - d), color=C["grey"], label="minimax  log(1−D)   (G minimises)"); ax.plot(d, -np.log(d), color=C["blue"], label="non-saturating  −log D"); ax.plot(d, (d - 1) ** 2, color=C["green"], label="least-squares  (D−1)²")
    ax.set_ylim(-4, 6); ax.set_xlabel("D(G(z))  — discriminator's belief that the fake is real"); ax.set_title("Generator loss as a function of D's verdict"); ax.legend(fontsize=7)
    ax = axs[2]; ax.plot(d, np.abs(-1 / (1 - d)), color=C["grey"], label="|d/dD| minimax"); ax.plot(d, np.abs(-1 / d), color=C["blue"], label="|d/dD| non-saturating"); ax.plot(d, np.abs(2 * (d - 1)), color=C["green"], label="|d/dD| least-squares")
    ax.set_ylim(0, 8); ax.set_xlabel("D(G(z))"); ax.set_title("Gradient magnitude reaching G"); ax.legend(fontsize=7)
    ax.annotate("minimax: ~0 gradient when D rejects\nfakes confidently (early training)", xy=(0.08, 1.1), xytext=(0.25, 5.5), fontsize=7, arrowprops=dict(arrowstyle="->", color=C["grey"]))
    save(fig, "fig_gan_game.png")

def fig_unet():
    chans = [3, 64, 128, 256, 512, 512, 512, 512, 512]; res = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    n = len(chans); xs_enc = np.linspace(4, 44, n); xs_dec = np.linspace(56, 96, n)[::-1]
    def h_of(c): return 3 + 8 * np.log2(max(c, 2)) / 9.0
    for i, (c, r) in enumerate(zip(chans, res)):
        h = h_of(c); y = 20 - h / 2
        ax.add_patch(Rectangle((xs_enc[i] - 1.6, y), 3.2, h, fc="#bee3f8", ec=C["blue"], lw=1))
        ax.text(xs_enc[i], y - 1.6, f"{c}ch\n{r}²", ha="center", va="top", fontsize=6.5)
        if i > 0:
            ax.add_patch(Rectangle((xs_dec[i] - 1.6, y), 3.2, h, fc="#c6f6d5", ec=C["green"], lw=1))
            dec_c = c * 2 if i < n - 1 else c; stagger = 0 if i % 2 else 3.2
            ax.text(xs_dec[i], y - 1.6 - stagger, f"{dec_c}→{chans[i-1] if i>1 else 64}\n@{r}²", ha="center", va="top", fontsize=6)
    for i in range(1, n - 1):
        y = 20 + h_of(chans[i]) / 2 + 0.6
        ax.add_patch(FancyArrowPatch((xs_enc[i] + 1.8, y), (xs_dec[i] - 1.8, y), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=C["grey"], connectionstyle="arc3,rad=-0.25", ls="--"))
    ax.text(50, 36, "skip connections: encoder features concatenated to decoder input at every level", ha="center", fontsize=8, color=C["grey"], style="italic")
    ax.text(xs_enc[0], 34, "input A\n(3 ch, or 4/12 in two_stage)", ha="center", fontsize=7)
    ax.text(xs_dec[1] + 3, 34, "extra head:\nConvT(128→64) → IN → ReLU\n→ Conv3×3(64→1) → tanh", ha="center", fontsize=7)
    ax.text(50, 3, "encoder block: Conv4×4 s2 → InstanceNorm → LeakyReLU(0.2)     decoder block: ReLU → ConvT4×4 s2 → InstanceNorm  (dropout 0.5 optional in the 512-ch blocks)", ha="center", fontsize=7.2, color=C["grey"])
    ax.text(50, 0.5, "num_downs = log2(crop_size) = 8 for a 256 crop → 1×1 bottleneck;  54.5 M parameters at ngf=64", ha="center", fontsize=7.5, fontweight="bold")
    save(fig, "fig_unet.png")

def fig_patchgan(rgb_crop):
    fig = plt.figure(figsize=(10, 3.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    layers = [("Conv 4×4 s2 (in→64)", 4, 2), ("Conv 4×4 s2 (64→128)", 10, 4), ("Conv 4×4 s2 (128→256)", 22, 8), ("Conv 4×4 s1 (256→512)", 46, 8), ("Conv 4×4 s1 (512→1)", 70, 8)]
    ax.set_xlim(0, 190); ax.set_ylim(0, 6)
    for i, (name, rf, st) in enumerate(layers):
        y = 5 - i; ax.barh(y, rf, height=0.6, color=plt.cm.Blues(0.35 + 0.12 * i))
        ax.text(rf + 2, y, f"{name}\nRF {rf} px, stride {st}", va="center", fontsize=6.8)
    ax.set_title("70×70 PatchGAN: receptive field growth (n_layers_d = 3)", fontsize=9)
    ax = fig.add_subplot(gs[1]); ax.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.add_patch(Rectangle((60, 60), 70, 70, fill=False, ec=C["red"], lw=2))
    for k in range(0, 257, 32): ax.axhline(k, color="white", lw=0.3, alpha=0.6); ax.axvline(k, color="white", lw=0.3, alpha=0.6)
    ax.set_title("one D logit judges a 70×70 patch\n(30×30 logits per 256² crop)", fontsize=8)
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    ax.text(0, 0.95, "What D receives (conditional):", fontsize=8.5, fontweight="bold", va="top")
    ax.text(0, 0.80, "pix2pix   D([A, B]) — 4 channels\ntwo_stage D1([A, T-plane, S]) — 5 ch\n          D2([A, R]) — 4 ch\ncyclegan  D_B(B) 1 ch, D_A(A) 3 ch\n          (unconditional)", fontsize=7.8, va="top", family="monospace")
    ax.text(0, 0.30, "Loss = mean over the 30×30 map of\n(logit − target)²  (LSGAN)  — i.e. D acts\nas a texture / local-statistics critic;\nglobal layout is L1's job.", fontsize=7.8, va="top")
    ax.text(0, 0.02, "--num-scales-d 2: a second PatchGAN on the\n2× average-pooled input → 140-px view.", fontsize=7.8, va="top", color=C["grey"])
    save(fig, "fig_patchgan.png")

def fig_decomp(nir_crop_u8):
    import torch
    from day2thermal.torch_utils import gaussian_blur
    B = torch.from_numpy(nir_crop_u8.astype(np.float32) / 127.5 - 1.0)[None, None]; S = gaussian_blur(B, 8.0); R = B - S
    b, s, r = B[0, 0].numpy(), S[0, 0].numpy(), R[0, 0].numpy()
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, im, t, cm, vr in [(axs[0], b, f"B (target)  std={b.std():.3f}", "gray", (-1, 1)), (axs[1], s, f"S = blur(B, σ=8)  std={s.std():.3f}", "gray", (-1, 1)),
                              (axs[2], r, f"R = B − S  std={r.std():.3f}, range [{r.min():.2f},{r.max():.2f}]", "coolwarm", (-0.6, 0.6))]:
        ax.imshow(im, cmap=cm, vmin=vr[0], vmax=vr[1]); ax.set_title(t, fontsize=8); ax.axis("off")
    ax = axs[3]; ax.hist(b.ravel(), bins=100, color=C["blue"], alpha=0.6, label="B", density=True); ax.hist(r.ravel(), bins=100, color=C["red"], alpha=0.6, label="R", density=True)
    ax.set_title("R has a much narrower distribution than B", fontsize=8); ax.legend(fontsize=7); ax.set_xlabel("value ([−1,1] scale)")
    fig.suptitle("Two-stage target decomposition exactly as train.py builds it (illustrated on a NIR crop; the operator is the same for thermal)", fontsize=8.5)
    save(fig, "fig_decomp.png"); return float(b.std()), float(s.std()), float(r.std())

def fig_curves(log_path, iters_per_epoch=916):
    it_pat = re.compile(r"^\[e(\d+) i(\d+)\] D=([\d.]+) G_gan=([\d.]+) G_L1=([\d.]+)"); val_pat = re.compile(r"^\[e(\d+)\] val L1 = ([\d.]+) \| ([\d.]+)s")
    its, D, Gg, Gl1, ve, vl, vt = [], [], [], [], [], [], []
    for line in open(log_path):
        m = it_pat.match(line)
        if m: its.append(int(m.group(1)) * iters_per_epoch + int(m.group(2))); D.append(float(m.group(3))); Gg.append(float(m.group(4))); Gl1.append(float(m.group(5)))
        m = val_pat.match(line)
        if m: ve.append(int(m.group(1)) + 1); vl.append(float(m.group(2))); vt.append(float(m.group(3)))
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
    ax = axs[0]; ax.plot(its, D, "o-", ms=3, color=C["red"], label="D loss (LSGAN, ½·[real+fake])"); ax.plot(its, Gg, "s-", ms=3, color=C["blue"], label="G adversarial (D(A,fake)−1)²")
    ax.axhline(0.25, color=C["grey"], ls=":", lw=1); ax.text(its[-1], 0.27, "0.25 = D maximally confused", ha="right", fontsize=7, color=C["grey"])
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_title("Adversarial terms (runs/nir_p2p)"); ax.legend(fontsize=7)
    ax = axs[1]; ax.plot(its, np.array(Gl1) / 100.0, "o-", ms=3, color=C["green"]); ax.set_xlabel("iteration"); ax.set_title("Train L1 = G_L1/100  (per-batch, [−1,1] scale)"); ax.set_ylim(0, 0.45)
    ax = axs[2]; ax.plot(ve, vl, "o-", color=C["purple"], label="val L1 (400 centre crops, held-out sessions)"); ax.set_xlabel("epoch"); ax.set_title("Validation L1 per epoch ([−1,1] scale)"); ax.set_ylim(0, 0.2)
    for e_, v_ in zip(ve, vl): ax.annotate(f"{v_:.3f}", (e_, v_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.legend(fontsize=7); save(fig, "fig_curves.png")
    return {"iters": its, "D": D, "Gg": Gg, "Gl1": Gl1, "val_epochs": ve, "val": vl, "val_t": vt}

def fig_samples(sample_dir):
    first = cv2.imread(os.path.join(sample_dir, "e000_i00000.png")); names = sorted(os.listdir(sample_dir)); last = cv2.imread(os.path.join(sample_dir, names[-1]))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, im, t in [(axs[0], first[:512], "epoch 0, iteration 0 (untrained G):\nRGB | real NIR | generated"), (axs[1], last[:512], f"{names[-1][:-4]}:\nRGB | real NIR | generated")]:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(t, fontsize=8.5); ax.axis("off")
    save(fig, "fig_samples.png"); return names[-1]

def fig_fullframe(ckpt, val_rgb_dir, val_th_dir, names):
    import torch
    from day2thermal.infer import load_model, translate
    from day2thermal.torch_utils import pad_to_multiple, unpad
    from day2thermal.utils import read_rgb, rgb_to_norm
    from day2thermal.eval import ssim_pair
    dev = torch.device("cpu"); nets, targs, tn, nd = load_model(ckpt, dev)
    fig, axs = plt.subplots(len(names), 4, figsize=(11, 2.15 * len(names))); fig.subplots_adjust(hspace=0.28, wspace=0.05); stats = []
    for r, n in enumerate(names):
        bgr = read_rgb(os.path.join(val_rgb_dir, n)); gt = cv2.imread(os.path.join(val_th_dir, n), -1)
        A = torch.from_numpy(rgb_to_norm(bgr))[None]; A, hw = pad_to_multiple(A, 2 ** nd)
        with torch.no_grad(): pred = unpad(translate(nets, targs, A, 0.0), hw)[0, 0].numpy()
        p01 = (np.clip(pred, -1, 1) + 1) / 2; g01 = gt.astype(np.float32) / 255.0
        l1 = float(np.abs(p01 - g01).mean()); ss = ssim_pair(p01, g01); stats.append((n, l1, ss))
        for c, (im, t, cm) in enumerate([(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "RGB input", None), (p01, "generated NIR (epoch-6 ckpt)", "gray"), (g01, "real NIR", "gray"), (np.abs(p01 - g01), f"|error|  L1={l1:.3f}  SSIM={ss:.3f}", "magma")]):
            ax = axs[r, c]; ax.imshow(im, cmap=cm, vmin=0, vmax=1 if cm != "magma" else 0.5); ax.set_title(t, fontsize=8); ax.axis("off")
        axs[r, 0].text(0, -0.02, n, transform=axs[r, 0].transAxes, fontsize=6.5, va="top", color=C["grey"])
    save(fig, "fig_fullframe.png"); return stats

def fig_parallax():
    Z = np.logspace(1, 3.4, 200); f = 1444.0; fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for b, col in [(0.05, C["green"]), (0.10, C["blue"]), (0.20, C["orange"])]: ax.plot(Z, f * b / Z, color=col, label=f"baseline b = {b*100:.0f} cm")
    ax.axhline(1, color=C["red"], ls="--", lw=1); ax.text(12, 1.15, "1 px", color=C["red"], fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("scene range Z [m]"); ax.set_ylabel("residual parallax [px]")
    ax.set_title("Single-homography error d ≈ f·b/Z  (f = 1444 px: 640 px across ~25° HFOV)", fontsize=8.5); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both"); save(fig, "fig_parallax.png")

def fig_lr(n_epochs=40, n_decay=20):
    e = np.arange(0, n_epochs + n_decay); lam = np.array([max(0.0, 1.0 - max(0, k + 1 - n_epochs) / max(1, n_decay)) for k in e])
    fig, ax = plt.subplots(figsize=(5.2, 2.6)); ax.step(e + 1, lam * 2e-4, where="post", color=C["blue"]); ax.set_xlabel("epoch (1-based)"); ax.set_ylabel("learning rate")
    ax.set_title(f"lr schedule as coded: constant for --n-epochs={n_epochs}, then linear to 0 over --n-epochs-decay={n_decay}", fontsize=8)
    ax.annotate("last decay epoch runs at lr = 0", xy=(60, 0), xytext=(38, 6e-5), fontsize=7.5, arrowprops=dict(arrowstyle="->")); save(fig, "fig_lr.png")

def fig_target_stats(train_th_dir):
    names = sorted(os.listdir(train_th_dir)); pick = random.Random(0).sample(names, 300); hist = np.zeros(256); per_sess = {}
    for n in pick:
        im = cv2.imread(os.path.join(train_th_dir, n), -1); h, _ = np.histogram(im, bins=256, range=(0, 256)); hist += h
        per_sess.setdefault(re.match(r"(.+?)_\d{6}\.png$", n).group(1), []).append(float(im.mean()))
    hist /= hist.sum(); fig, axs = plt.subplots(1, 2, figsize=(10, 3.0))
    ax = axs[0]; ax.bar(np.arange(256), hist, width=1, color=C["blue"]); ax.set_xlabel("NIR pixel value (8-bit)"); ax.set_ylabel("fraction"); ax.set_title(f"Target histogram, 300 random train frames — {hist[0]*100:.2f}% at 0, {hist[255]*100:.2f}% at 255", fontsize=8.5)
    ax = axs[1]; ks = sorted(per_sess); ax.bar(range(len(ks)), [np.mean(per_sess[k]) for k in ks], yerr=[np.std(per_sess[k]) for k in ks], color=C["green"], capsize=2)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=60, ha="right", fontsize=6.5); ax.set_ylabel("mean NIR value"); ax.set_title("Per-session mean target (±σ over frames): sessions differ → hold out whole sessions", fontsize=8.5)
    save(fig, "fig_target_stats.png"); return float(hist[0]), float(hist[255])

if __name__ == "__main__":
    fig_pipeline(); fig_l1_median(); fig_gan_game(); fig_unet(); fig_parallax(); fig_lr()
    val_rgb = f"{REPO}/data/agri_rgb_nir/aligned/val/rgb"; val_th = f"{REPO}/data/agri_rgb_nir/aligned/val/thermal"; n0 = "wheat_27072019_000500.png"
    rgb = cv2.imread(os.path.join(val_rgb, n0)); nir = cv2.imread(os.path.join(val_th, n0), -1); y, x = (rgb.shape[0] - 256) // 2, (rgb.shape[1] - 256) // 2
    fig_patchgan(rgb[y:y + 256, x:x + 256]); stds = fig_decomp(nir[y:y + 256, x:x + 256])
    curves = fig_curves(f"{REPO}/runs/nir_p2p/train.log"); last_sample = fig_samples(f"{REPO}/runs/nir_p2p/samples")
    stats = fig_fullframe(f"{REPO}/runs/nir_p2p/checkpoints/latest.pt", val_rgb, val_th, ["wheat_27072019_000500.png", "drybean-30072020_1_000200.png"])
    tstats = fig_target_stats(f"{REPO}/data/agri_rgb_nir/aligned/train/thermal")
    json.dump({"decomp_std_B_S_R": stds, "curves": curves, "last_sample": last_sample, "fullframe": stats, "target_frac_0_255": tstats}, open(os.path.join(HERE, "figdata.json"), "w"), indent=1)
    print("all figures done")
