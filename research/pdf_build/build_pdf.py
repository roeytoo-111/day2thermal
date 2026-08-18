# -*- coding: utf-8 -*-
"""Build research/day2thermal_algorithms_explained.pdf with reportlab.
Run after make_figs.py:  python3 research/pdf_build/build_pdf.py
"""
import os, hashlib, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, Preformatted)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); EQ = os.path.join(HERE, "eq"); os.makedirs(EQ, exist_ok=True)
OUT = os.path.abspath(os.path.join(HERE, "..", "day2thermal_algorithms_explained.pdf"))
FD = json.load(open(os.path.join(HERE, "figdata.json")))

FONTDIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DV", f"{FONTDIR}/DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("DV-B", f"{FONTDIR}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-M", f"{FONTDIR}/DejaVuSansMono.ttf")); pdfmetrics.registerFont(TTFont("DV-S", f"{FONTDIR}/DejaVuSerif.ttf"))
registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-S", boldItalic="DV-B")

BODY = ParagraphStyle("Body", fontName="DV", fontSize=9.3, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5)
TITLE = ParagraphStyle("Title", fontName="DV-B", fontSize=22, leading=27, spaceAfter=8, textColor=colors.HexColor("#1a202c"))
SUBT = ParagraphStyle("Subt", fontName="DV", fontSize=12, leading=16, textColor=colors.HexColor("#4a5568"), spaceAfter=14)
H1 = ParagraphStyle("H1", fontName="DV-B", fontSize=15, leading=19, spaceBefore=14, spaceAfter=7, textColor=colors.HexColor("#1a365d"))
H2 = ParagraphStyle("H2", fontName="DV-B", fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#2b6cb0"))
H3 = ParagraphStyle("H3", fontName="DV-B", fontSize=9.8, leading=13, spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#2d3748"))
PART = ParagraphStyle("Part", fontName="DV-B", fontSize=20, leading=25, spaceBefore=6, spaceAfter=10, textColor=colors.HexColor("#1a365d"))
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2.5, alignment=TA_LEFT)
CAP = ParagraphStyle("Cap", fontName="DV-S", fontSize=8, leading=10.5, textColor=colors.HexColor("#4a5568"), alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)
CODE = ParagraphStyle("Code", fontName="DV-M", fontSize=7.4, leading=9.6, backColor=colors.HexColor("#f7fafc"), borderColor=colors.HexColor("#cbd5e0"), borderWidth=0.5, borderPadding=4, leftIndent=4, spaceBefore=3, spaceAfter=7)
NOTE = ParagraphStyle("Note", parent=BODY, backColor=colors.HexColor("#fffbea"), borderColor=colors.HexColor("#d69e2e"), borderWidth=0.6, borderPadding=5, leftIndent=3, spaceBefore=4, spaceAfter=9)
KEY = ParagraphStyle("Key", parent=BODY, backColor=colors.HexColor("#ebf8ff"), borderColor=colors.HexColor("#2b6cb0"), borderWidth=0.6, borderPadding=5, leftIndent=3, spaceBefore=4, spaceAfter=9)
TCELL = ParagraphStyle("TCell", fontName="DV", fontSize=7.6, leading=9.8, alignment=TA_LEFT); TCELLB = ParagraphStyle("TCellB", parent=TCELL, fontName="DV-B")
TOC0 = ParagraphStyle("TOC0", fontName="DV-B", fontSize=9.5, leading=13, leftIndent=0); TOC1 = ParagraphStyle("TOC1", fontName="DV", fontSize=8.8, leading=12, leftIndent=14)
W = A4[0] - 4 * cm

def eq(tex, size=12.5, maxw=W):
    h = hashlib.md5((tex + str(size)).encode()).hexdigest()[:10]; p = os.path.join(EQ, f"eq_{h}.png")
    if not os.path.exists(p):
        f = plt.figure(figsize=(0.01, 0.01)); f.text(0, 0, tex, fontsize=size); f.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True); plt.close(f)
    im = PILImage.open(p); w_px, h_px = im.size; w_pt, h_pt = w_px / 300 * 72, h_px / 300 * 72
    if w_pt > maxw: s = maxw / w_pt; w_pt, h_pt = maxw, h_pt * s
    img = Image(p, width=w_pt, height=h_pt); img.hAlign = "CENTER"; return [Spacer(1, 2), img, Spacer(1, 5)]
def fig(name, width_cm, caption):
    p = os.path.join(FIG, name); im = PILImage.open(p); w_px, h_px = im.size; w = min(width_cm * cm, W); h = w * h_px / w_px
    img = Image(p, width=w, height=h); img.hAlign = "CENTER"; return [KeepTogether([Spacer(1, 4), img, Paragraph(caption, CAP)])]
def P(t, st=BODY): return [Paragraph(t, st)]
def H(t, lvl): return [Paragraph(t, {0: PART, 1: H1, 2: H2, 3: H3}[lvl])]
def BL(items, st=BUL): return [Paragraph(f"• {i}", st) for i in items]
def NL(items, st=BUL): return [Paragraph(f"{k+1}. {i}", st) for k, i in enumerate(items)]
def CD(t): return [Preformatted(t, CODE)]
def KEYB(t): return [Paragraph(t, KEY)]
def TB(rows, widths=None, header=True):
    data = [[Paragraph(str(c), TCELLB if (header and r == 0) else TCELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths] if widths else None, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header: style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0"))]
    for i in range(1, len(rows)):
        if i % 2 == 0: style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7fafc")))
    t.setStyle(TableStyle(style)); return [Spacer(1, 3), t, Spacer(1, 7)]

class Doc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        BaseDocTemplate.__init__(self, fn, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=1.8 * cm, **kw)
        self.addPageTemplates([PageTemplate("normal", [Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="F")], onPage=self._deco)]); self._n = 0
    def _deco(self, canv, doc):
        canv.saveState(); canv.setFont("DV", 7.5); canv.setFillColor(colors.HexColor("#718096"))
        canv.drawString(2 * cm, 1.1 * cm, "day2thermal — RGB → IR translation with GANs: algorithms, objectives, data, tuning"); canv.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"page {doc.page}")
        canv.setStrokeColor(colors.HexColor("#cbd5e0")); canv.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm); canv.restoreState()
    def beforeDocument(self): self._n = 0
    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("Part", "H1", "H2"):
            lvl = {"Part": 0, "H1": 0, "H2": 1}[fl.style.name]; self._n += 1; key = f"h{self._n}"; self.canv.bookmarkPage(key); self.canv.addOutlineEntry(fl.getPlainText(), key, level=lvl, closed=False)
            self.notify("TOCEntry", (lvl, fl.getPlainText(), self.page, key))

S = []
S += P("RGB → IR Image Translation with GANs", TITLE)
S += P("The <b>day2thermal</b> pipeline explained from first principles: how GANs work, the loss and target functions, what the code does exactly, what is critical about the data, which parameters matter, and what to expect while training.", SUBT)
S += P("<b>Scope and provenance.</b> Everything stated about the implementation was checked against the source at commit <font name='DV-M'>bec21e4</font> of <font name='DV-M'>github.com/roeytoo-111/day2thermal</font> (files: dataset.py, networks.py, losses.py, train.py, infer.py, eval.py, extract_frames.py, register.py, prep_aligned.py, prep_unpaired.py, torch_utils.py, utils.py, check_thermalness.py). Every number quoted from experiments was measured on this machine on 2026-08-17: the RGB→NIR run <font name='DV-M'>runs/nir_p2p</font> (pix2pix, stopped after 6 epochs), its 60-iteration sanity check, and the properties of the two datasets. Where something is an expectation rather than a measurement, it is labelled as such. Document generated 2026-08-17.")
S += [Spacer(1, 6)]
S += KEYB("<b>How to read this document.</b> Part I builds the theory from zero (pixel losses → GANs → conditional GANs → pix2pix → LSGAN → multimodal translation → ThermalGAN → CycleGAN). Part II walks through the code exactly. Part III is the data checklist. Part IV is the parameter reference, a tuning procedure, and how to read training signals. If you only train models and want the practical view, read §2, §4, then Parts III–IV; the figures in §14 show what a healthy run looks like.")
toc = TableOfContents(); toc.levelStyles = [TOC0, TOC1]
S += [Spacer(1, 8), Paragraph("Contents", H1), toc, PageBreak()]

S += H("Notation used throughout", 1)
S += TB([["Symbol", "Meaning (as in the code)"],
         ["A", "RGB input, tensor 3×H×W in [−1, 1] (dataset.py: <font name='DV-M'>rgb_to_norm</font>: x/127.5 − 1)"],
         ["B", "IR target, tensor 1×H×W in [−1, 1] (utils.py: <font name='DV-M'>ThermalNorm.to_norm</font>) — thermal (LWIR) or NIR, whatever the folder <font name='DV-M'>thermal/</font> holds"],
         ["G, D", "generator, discriminator (networks.py: <font name='DV-M'>UnetGenerator</font>, <font name='DV-M'>NLayerDiscriminator</font>)"],
         ["B̂ = G(A)", "generated IR (train.py: <font name='DV-M'>fake</font>)"],
         ["T", "scalar temperature condition in [−1, 1] (two_stage only): logged °C via --temps-csv, else mean(B) of the crop"],
         ["S, R", "low-frequency base S = blur(B, σ) and residual R = B − S (two_stage); Ŝ, R̂ their predictions"],
         ["p_data, p_g", "distribution of real (A,B) pairs / of generated pairs"],
         ["‖·‖₁", "mean absolute value over all pixels (torch.nn.L1Loss, reduction 'mean')"],
         ["λ_L1", "--lambda-l1, default 100"],
         ["[−1, 1] vs [0, 1] scale", "training/validation losses are on [−1, 1]; eval.py reports on [0, 1] (half the value)"]], widths=[3.2, 13.8])

# ================================================================== PART I
S += [PageBreak()] + H("Part I — Foundations: from pixel losses to ThermalGAN", 0)
S += H("1. The problem: paired image-to-image translation", 1)
S += P("We are given pairs (A, B): a colour image A and the infrared image B of the same scene at the same instant, registered so that pixel (x, y) in A and pixel (x, y) in B look along the same ray. We want a function G that maps a new A to a plausible B. Two things make this different from ordinary regression. First, B is a whole image with strong spatial structure — a good answer must have sharp edges, the right texture statistics and coherent objects, not just the right value at each pixel on average. Second, for some modalities the answer is genuinely <i>not unique</i>: several different B are compatible with the same A.")
S += H("1.1 One-to-many (thermal) versus nearly one-to-one (NIR)", 2)
S += P("<b>Longwave thermal (LWIR, 8–14 µm)</b> measures <i>emitted</i> radiance: emissivity × Planck(surface temperature) plus a reflected-ambient term. The visible image tells you shape, class and context, but not the temperature: the same airframe against the same sky at 09:00 and 15:00 is identical in RGB and differs by many °C in LWIR (solar loading, ambient temperature, sky state). So p(B | A) has several modes. ThermalGAN (Kniaz et al., 2018) measured the consequence: generic conditional GANs recover absolute object temperature only to about 5 °C, while the fine contrasts that carry the useful signal live at the 1 °C scale.")
S += P("<b>Near-infrared (NIR, 0.7–1.5 µm)</b> — the band in the agricultural dataset used for the run in this document — is <i>reflected</i> sunlight, like RGB. Vegetation reflects NIR strongly (chlorophyll/cell structure), soil weakly; NIR is therefore largely determined by what RGB already shows (greenness, material), up to illumination and exposure. p(B | A) is close to unimodal. The same code trains both, but the parts of the method built for ambiguity (temperature conditioning, the two-stage split) only have meaning for thermal.")
S += KEYB("<b>Why this matters for you.</b> The choice of model follows from the physics of the target: for thermal, ambiguity is the central problem and pix2pix is a baseline; for NIR, pix2pix is the right model and the extra machinery is unnecessary. Section 5 explains the thermal-specific parts; Parts III–IV say when to use them.")

S += H("2. Pixel losses and what they converge to", 1)
S += P("Suppose you train G with a pixel loss only. For a fixed input A, the network can output one value per pixel; the loss averages over all the B that occur with that A in the training set. The minimiser is a well-defined statistic of p(B | A):")
S += eq(r"$\arg\min_b \; \mathrm{E}[\,|B-b|\,] = \mathrm{median}(B\,|\,A), \qquad \arg\min_b \; \mathrm{E}[(B-b)^2] = \mathrm{E}[B\,|\,A]$")
S += P("(Sketch for L1: the derivative of E|B − b| with respect to b is P(B &lt; b) − P(B &gt; b), which is zero exactly at the median.) When p(B | A) is unimodal and narrow, mean and median are the answer you want. When it is bimodal — cool-ambient thermal versus warm-ambient thermal — both statistics land <i>between</i> the modes, in a region of low probability: the prediction is an image nobody would ever record. Spatially the same thing happens: if plausible answers put an edge at slightly different places or with different contrast, their average is a softened, lower-contrast edge. That is the “blur” of pure-L1/L2 translation.")
S += fig("fig_l1_median.png", 16.5, "Figure 1. Left: for one RGB input with two plausible IR answers, the L2 optimum (mean) and L1 optimum (median) sit between the modes. Right: along an image row, the L1/L2-optimal prediction is the average of the plausible answers — softened edges and reduced contrast. A generator trained adversarially is pushed to <i>commit</i> to one plausible answer instead.")
S += P("<b>Frequency view.</b> The average of plausible answers gets the low spatial frequencies roughly right (global level, layout) and destroys the high frequencies (edges, texture), because those are exactly what differs between the answers. This observation is the design principle behind pix2pix: keep L1 for the low frequencies, add an adversarial term for the high frequencies, and make the discriminator look at <i>local patches</i> because high-frequency realism is a local property (§4.2).")

S += H("3. Generative adversarial networks from scratch", 1)
S += H("3.1 What a GAN is", 2)
S += P("A GAN (Goodfellow et al., 2014) learns to <i>sample</i> from a data distribution without ever writing down its density. It trains two networks against each other: a generator G that turns an input (noise z, or here the conditioning image A) into a candidate sample, and a discriminator D, a binary classifier asked to tell real samples from generated ones. The two are trained with opposite objectives:")
S += eq(r"$\min_G \max_D \; V(D,G) = \mathrm{E}_{x\sim p_{\mathrm{data}}}[\log D(x)] + \mathrm{E}_{z\sim p_z}[\log(1-D(G(z)))]$")
S += P("D maximises V (assign high probability to real x, low to G(z)); G minimises it (make D assign high probability to G(z)). The essential idea is that <b>the loss function is learned</b>: instead of a fixed formula such as L1, G is scored by an adaptive, data-dependent critic that can detect whatever makes samples look unreal — blur, wrong texture statistics, impossible combinations.")
S += H("3.2 What the game converges to", 2)
S += P("Fix G. For each x the discriminator solves max<sub>y</sub> a·log y + b·log(1 − y) with a = p_data(x), b = p_g(x); the maximiser is y = a/(a+b):")
S += eq(r"$D^*(x) = \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}, \qquad C(G) = \max_D V(D,G) = -\log 4 + 2\,\mathrm{JSD}(p_{\mathrm{data}}\,\|\,p_g)$")
S += P("Substituting D* back into V gives C(G), which is −log 4 plus twice the Jensen–Shannon divergence between the real and generated distributions. So, at the game's equilibrium, the generator minimises the JS divergence and the global optimum is p_g = p_data — the generator reproduces the data distribution and D*(x) = ½ everywhere (maximally confused). This is the sense in which a GAN's “target function” is a distribution, not a pixel value.")
S += fig("fig_gan_game.png", 16.8, "Figure 2. Left: the optimal discriminator for a fixed generator is the ratio p_data/(p_data + p_g). Middle: three generator losses as a function of D's verdict on a fake. Right: their gradients. The original minimax loss gives G almost no gradient when D rejects fakes confidently — the state early in training; the non-saturating (−log D) and least-squares losses do not have that problem. The code's <font name='DV-M'>--gan-mode vanilla</font> is the non-saturating cross-entropy; the default <font name='DV-M'>lsgan</font> is the least-squares form.")
S += H("3.3 Training in practice, and why it is delicate", 2)
S += P("Nobody solves the inner maximisation exactly. Training alternates one gradient step on D (with G frozen) and one on G (with D frozen), on the same batch. This is a two-player game, not a single objective being minimised, and it has characteristic pathologies:")
S += BL(["<b>Vanishing gradients.</b> When p_data and p_g barely overlap (always true early on for high-dimensional images), a near-perfect D exists, JSD sits at its maximum log 2, and the minimax generator loss log(1 − D(G(z))) is flat — G learns nothing. The non-saturating loss (maximise log D(G(z))) fixes the flatness at the cost of noisier gradients; the least-squares loss (§3.4) fixes it more gently.",
         "<b>Oscillation / non-convergence.</b> Each player's loss surface moves as the other updates; simultaneous gradient descent can circle instead of converging. Practical dampers: low momentum (Adam β₁ = 0.5 instead of 0.9 — the DCGAN finding), a slower D (the ½ factor on the D loss in this code), a history buffer of old fakes for D (CycleGAN's <font name='DV-M'>ImagePool</font>).",
         "<b>Mode collapse.</b> G maps many inputs to a few outputs that fool D, D adapts, G jumps to another few. Strongly <i>conditioned</i> GANs with a reconstruction term (paired A → B with L1) are largely immune, because L1 pins each output to its own target; unpaired GANs (CycleGAN) are more exposed.",
         "<b>D wins outright.</b> If D becomes perfect, G's gradient dies (see above). Symptoms in the log: D loss → 0, G adversarial loss climbing."])
S += P("Everything this code does that looks like ‘folklore’ is a response to one of these: LSGAN by default, Adam(2e-4, β = (0.5, 0.999)), alternating single steps with the fake detached in the D step, N(0, 0.02) weight initialisation, InstanceNorm, the L1 anchor, and the image pool in the unpaired mode.")
S += H("3.4 Least-squares GAN (the default here)", 2)
S += P("LSGAN (Mao et al., 2017) replaces the sigmoid/cross-entropy classifier by a regression: D outputs a raw score and is trained to output 1 on real inputs and 0 on fakes with a squared loss; G is trained to make D output 1 on its fakes:")
S += eq(r"$\mathcal{L}_{LSGAN}(D) = \mathrm{E}_{x\sim p_{data}}[(D(x)-1)^2] + \mathrm{E}_{\hat x\sim p_g}[(D(\hat x)-0)^2], \qquad \mathcal{L}_{LSGAN}(G) = \mathrm{E}_{\hat x\sim p_g}[(D(\hat x)-1)^2]$")
S += P("The penalty grows with the distance of D's score from the target, so a fake that D rejects confidently still produces a large, well-behaved gradient for G (Figure 2, right). Mao et al. show that for a particular coding of the targets the LSGAN objective minimises a Pearson-χ² divergence; with the 0/1 coding used in this code the correspondence is not exact, but the property that matters — smooth, non-saturating gradients that push fakes toward the decision boundary — holds. In <font name='DV-M'>losses.py</font>: <font name='DV-M'>GANLoss('lsgan')</font> is <font name='DV-M'>nn.MSELoss</font> against a tensor of ones or zeros shaped like D's output map; <font name='DV-M'>'vanilla'</font> is <font name='DV-M'>nn.BCEWithLogitsLoss</font> (D outputs logits; no sigmoid layer exists in <font name='DV-M'>NLayerDiscriminator</font>). When D is multi-scale the loss is averaged over the scales.")

S += H("4. Conditional GANs and pix2pix", 1)
S += H("4.1 Conditioning the discriminator", 2)
S += P("If D only ever saw B, G could satisfy it with <i>any</i> realistic IR image — including one unrelated to A. pix2pix (Isola et al., 2017) therefore feeds D the pair: D(A, B). D then learns the joint distribution and rejects outputs that are realistic but inconsistent with the input. In this code the pair is literally a channel concatenation, <font name='DV-M'>torch.cat([A, B], 1)</font> → 4 input channels for D. The full pix2pix objective adds the L1 anchor:")
S += eq(r"$G^* = \arg\min_G \max_D \; \mathcal{L}_{cGAN}(G,D) + \lambda\,\mathcal{L}_{L1}(G), \qquad \mathcal{L}_{L1}(G) = \mathrm{E}_{A,B}\left[\|B - G(A)\|_1\right],\quad \lambda = 100$")
S += P("The two terms divide the work: L1 makes the output <i>faithful</i> (correct global level and layout, small average error) and the adversarial term makes it <i>plausible</i> (sharp, correct local statistics). λ = 100 with images in [−1, 1] and an LSGAN term of order 0.1–1 means L1 dominates the gradient — the GAN term is a sharpening prior, not the main driver. This is why paired GANs are so much more stable than unconditional ones, and it is the first parameter to move when outputs are too soft (lower λ) or too ‘creative’ (raise λ).")
S += H("4.2 The PatchGAN discriminator", 2)
S += P("Because the adversarial term is responsible for high frequencies, D does not need to see the whole image at once. The 70×70 PatchGAN is a small fully-convolutional net whose every output logit has a 70×70-pixel receptive field; on a 256×256 crop it emits a 30×30 map of logits and the loss is the mean over that map. Interpretation: D models the image as a Markov random field with 70-px neighbourhoods — it judges texture and local structure, and treats pixels more than 70 px apart as independent. Consequences: few parameters (2.77 M here), works at any image size, and its verdict is precisely about the things L1 cannot judge.")
S += fig("fig_patchgan.png", 17, "Figure 3. Left: receptive-field growth through the five layers of <font name='DV-M'>NLayerDiscriminator</font> (n_layers_d = 3): 4 → 10 → 22 → 46 → 70 px. Middle: one logit judges one 70×70 window of the (A, B) pair. Right: what D receives in each mode.")
S += P("<font name='DV-M'>--num-scales-d 2</font> (<font name='DV-M'>MultiScaleDiscriminator</font>, from pix2pixHD) applies a second PatchGAN to the 2× average-pooled input, i.e. a critic with a 140-px view for coherence beyond a single patch (plot boundaries, tramlines, large objects); the fine-scale critic is unchanged and the two losses are averaged.")
S += H("4.3 The U-Net generator", 2)
S += P("Input and output share all their low-level geometry — every edge in the IR image is at an edge of the RGB image. An encoder–decoder with <i>skip connections</i> lets that structure bypass the bottleneck: at each resolution the decoder receives the encoder's features concatenated to its own up-sampled features, so it never has to re-invent where things are; the deep bottleneck (1×1 spatial for a 256 crop) carries the semantics — <i>what</i> this region is, hence how bright it should be in IR.")
S += fig("fig_unet.png", 17, "Figure 4. <font name='DV-M'>UnetGenerator</font> as built for a 256 crop: 8 stride-2 encoder stages, mirror decoder, skip concatenation at every level, ThermalGAN-style extra head. Labels under the decoder give the transposed-conv input→output channels.")
S += P("Two details matter later. The output goes through <b>tanh</b>, hence the [−1, 1] normalisation of all targets. And the default <b>InstanceNorm</b> standardises every feature map per sample: it makes training independent of batch size and removes per-image contrast from the features — welcome for a style-like mapping, but it means the absolute output level has to be carried by the skips and the last layers (relevant when you care about calibrated temperatures; <font name='DV-M'>--norm batch|none</font> exists for that experiment).")
S += H("4.4 What pix2pix cannot do", 2)
S += P("pix2pix's output is deterministic: one B per A. Isola et al. tried injecting noise and found the generator learns to ignore it (dropout was the only ‘noise’ that survived). For a one-to-many target that means the model can only ever produce one of the modes — or, if L1 wins, the average of them. That is the gap ThermalGAN closes (§5).")

S += H("5. Multimodal translation: BicycleGAN → ThermalGAN", 1)
S += H("5.1 BicycleGAN", 2)
S += P("BicycleGAN (Zhu et al., 2017) forces a latent code z to matter by two constraints. <b>cVAE-GAN</b>: an encoder E maps the real B to a distribution q(z | B) = N(μ, σ²); a sample z is fed to G together with A; the reconstruction G(A, z) must match B (L1 + adversarial) and q(z | B) is pulled toward N(0, I) by a KL term, so that at test time random z ~ N(0, I) produce valid outputs. <b>cLR-GAN</b>: sample z, generate G(A, z), re-encode with E and require ẑ ≈ z, so different z cannot collapse to the same output. Together (the ‘bicycle’) they give a controllable but <i>arbitrary</i> latent.")
S += eq(r"$\mathcal{L}_{KL} = -\frac{1}{2}\,\mathrm{mean}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$")
S += H("5.2 ThermalGAN's two ideas", 2)
S += P("<b>Idea 1 — replace the arbitrary latent by physics.</b> The multimodality of thermal images is not style, it is temperature. ThermalGAN replaces z by a temperature vector T (background and object temperatures). Now (A, T) → B is nearly deterministic: the L1 target is unambiguous, the output is controllable, and sweeping T at inference yields the multimodal ‘probe sets’ — repurposed here as free weather augmentation for detector training.")
S += P("<b>Idea 2 — split the target into a hard part and an easy part.</b> Write B = S + R with S the low-frequency ‘base’ (average object temperatures; in the paper a semantic thermal segmentation) and R the residual of relative contrasts. Physically S carries the ambient-dependent absolute level, i.e. the multimodality; R (edges, local contrasts from emissivity, heat capacity, internal sources) is nearly invariant to ambient temperature — the paper's Figure 5 shows the same person at 17/25/28 °C ambient with almost identical R. So stage 1, G₁: (A, T) → Ŝ, gets all the ambiguity and is handed T; stage 2, G₂: (A, Ŝ) → R̂, sees a unimodal target of small dynamic range and low variance, for which plain pix2pix suffices and whose entire L1 budget goes to the fine contrasts. The two GAN pairs are trained independently and the output is the sum:")
S += eq(r"$S = k_\sigma * B,\quad R = B - S,\quad \hat S = G_1([A, T]),\quad \hat R = G_2([A, \hat S]),\quad \hat B = \mathrm{clip}(\hat S + \hat R, -1, 1)$")
S += fig("fig_decomp.png", 17, f"Figure 5. The decomposition exactly as <font name='DV-M'>train.py::step_two_stage</font> computes it (<font name='DV-M'>torch_utils.gaussian_blur</font>, σ = 8 px, separable, reflect-padded), applied to a 256-px NIR crop for illustration: std(B) = {FD['decomp_std_B_S_R'][0]:.3f}, std(S) = {FD['decomp_std_B_S_R'][1]:.3f}, std(R) = {FD['decomp_std_B_S_R'][2]:.3f}. R's distribution is half as wide as B's — that is the variance reduction stage 2 benefits from.")
S += H("5.3 How this code adapts it (and what that changes)", 2)
S += BL(["<b>Base S = Gaussian low-pass of the real B</b> (<font name='DV-M'>--lowpass-sigma</font>, default 8 px at crop scale; kernel radius round(3σ) = 24 → 49 taps) instead of a semantic segmentation supervised by pixel labels — no labels exist for this data. Structures smaller than ~2σ live entirely in R.",
         "<b>T is a scalar plane</b>: the value from <font name='DV-M'>--temps-csv</font> (°C, mapped through the abs16 normalisation) if given, otherwise <b>mean(B) of the crop</b> — read off the target. That is intentional (it is how the ambiguity becomes an input), but note: validation uses the same oracle, so <font name='DV-M'>two_stage</font> val L1 is optimistic relative to deployment where you supply T.",
         "<b>Temperature loss</b> λ_temp·|mean(Ŝ) − T| (default 10) as a simple proxy for the paper's latent-temperature-domain loss.",
         "<b>Optional VAE</b> (<font name='DV-M'>--use-vae</font>): only the cVAE-GAN half — E(S) → (μ, log σ²), z = μ + σ·ε broadcast to <font name='DV-M'>--nz</font> = 8 extra input planes of G₁, KL weight 0.01; z = 0 at inference. Marked experimental.",
         "<b>G₂ trains on the predicted Ŝ, detached</b> — the same imperfect input it will see at test time; the two GAN pairs stay independent.",
         "<b>Representational caveat.</b> R = B − S can exceed ±1 next to saturated regions while G₂ ends in tanh; on 8-bit AGC data that saturates at 0/255 (the arsuf set does, on every frame) this clips a little contrast at saturated edges. Negligible in practice, real in principle."])

S += H("6. Unpaired translation: CycleGAN (the fallback)", 1)
S += P("When registration is impossible, no pixel correspondence exists and neither L1 nor a conditional D can be used. CycleGAN (Zhu et al., 2017) trains two generators, G: RGB → IR and F: IR → RGB, with unconditional discriminators on each domain, and substitutes <i>cycle consistency</i> for correspondence — the only thing tying G's output to its input is that F must be able to undo it:")
S += eq(r"$\mathcal{L}_{cyc} = \|F(G(A)) - A\|_1 + \|G(F(B)) - B\|_1, \qquad \mathcal{L}_{idt} = \|G(B\otimes\mathbf{1}_3) - B\|_1 + \|F(\bar A) - A\|_1$")
S += P("The identity term in this code is channel-adapted (G is 3→1, F is 1→3): the IR image replicated to three channels through G should come back unchanged, and the channel-mean of RGB through F should reconstruct RGB. It anchors global intensity and prevents contrast inversion — but it also literally teaches G that a grey-looking input maps to itself, i.e. it nudges toward the <b>greyscale shortcut</b>: emitting something close to the input's luminance satisfies cycle consistency perfectly (trivially invertible) and partly fools D, yet it is a desaturation, not a modality change; real LWIR is nearly uncorrelated with visible brightness because temperature is not albedo. <font name='DV-M'>check_thermalness.py</font> measures exactly this: corr(G(RGB), luminance(RGB)) versus corr(real IR, luminance(RGB)); &gt; 0.75 = shortcut, &lt; 0.5 = decoupled. D is trained on an <font name='DV-M'>ImagePool</font> (default 50) mixing current and older fakes to damp G/D oscillation. Unpaired output can also hallucinate content — acceptable for style, dangerous for radiometry and for detector training data.")

S += H("7. Why a GAN and not a diffusion model (for this problem)", 1)
S += P("Diffusion models learn to invert a fixed noising process: a network ε<sub>θ</sub>(x<sub>t</sub>, t, cond) predicts the noise added at step t and sampling runs the reverse chain (≈1000 steps for DDPM, 20–50 with DDIM/DPM-Solver samplers, 1–4 with distilled variants). Training is a plain regression — no adversary, no collapse, mode-covering by construction — and sample quality/diversity is generally higher. They are the natural phase-2 candidates (Palette, BBDM, ControlNet on a latent model). The reasons a conditional GAN is the right first tool here:")
S += BL(["<b>Inference cost scales with the payload.</b> The generator is one U-Net pass per frame (~0.1 s at 1098×798 on the laptop GPU); diffusion is 20–1000 passes of a comparable network. The pipeline exists to translate a large RGB-only set and to iterate on detector-in-the-loop experiments; 30× per frame is the difference between minutes and days.",
         "<b>Small paired data.</b> 316 pairs (arsuf) or 7 k pairs of one crop domain: diffusion from scratch memorises; fine-tuning a visible-domain prior imports the wrong statistics (shadows dark, sky bright, texture = albedo). GAN + a strong L1 signal converges fast and predictably.",
         "<b>Determinism and control.</b> two_stage is a function of (A, T): repeatable, sweepable, frame-consistent; diffusion samples are stochastic and per-frame flicker is worse.",
         "<b>Calibration.</b> An explicit temperature loss and additive base + contrast map directly onto a regression-style GAN; enforcing radiometric consistency through a stochastic denoising chain is possible but non-standard.",
         "<b>Hallucination is a liability here.</b> Rich priors invent plausible detail; for detector training data an invented hot spot is a false positive the detector learns. GAN + L1 is the more conservative regressor.",
         "<b>The brief was ThermalGAN.</b> Its contributions are GAN-native; a diffusion version would be a different method."])
S += P("When diffusion would win: much larger and more diverse paired sets, an 8-bit visual-realism goal rather than radiometric fidelity, a need for diverse augmentation beyond a scalar T, or a temporal model — and the speed gap keeps closing (latent diffusion, consistency/rectified-flow distillation).")

# ================================================================== PART II
S += [PageBreak()] + H("Part II — The implementation, exactly", 0)
S += fig("fig_pipeline.png", 17, "Figure 6. Data flow. Three entry points feed the same training layout; the pairing key is the filename.")
S += H("8. Data preparation", 1)
S += H("8.1 extract_frames.py — temporal synchronisation of two videos", 2)
S += P("<font name='DV-M'>FrameSource</font> reads frames sequentially and assigns time t = index / fps — a constant-frame-rate assumption (container timestamps are not read; check your streams' PTS if in doubt). <font name='DV-M'>NearestFetcher</font> is a streaming nearest-in-time lookup; the thermal fetcher is built with a time shift of −offset, so a query at RGB time t returns the thermal frame at thermal time t + offset — hence the definition <font name='DV-M'>--offset-ms</font> = thermal clock − RGB clock for the same event. The main loop samples t = start, start + 1/fps, … on the RGB clock, fetches the nearest RGB and thermal frames, and accepts the pair only if both lie within <font name='DV-M'>--tol-ms</font> of t (default: half the thermal frame period, so the nearest thermal frame always qualifies, and the nearest RGB frame does whenever the RGB rate ≥ thermal rate). <font name='DV-M'>--drop-static-thermal</font> skips a pair when mean|thermal<sub>t</sub> − thermal<sub>t−1</sub>| &lt; 0.05 (0–255 scale) — the frozen frames of the microbolometer's NUC/FFC shutter. Output <font name='DV-M'>rgb/000000.png</font>, <font name='DV-M'>thermal/000000.png</font>, thermal single-channel; the shared name is the pair.")
S += P("<b>Sync precision</b> is therefore bounded by ± tol plus the error of your offset. For the arsuf streams the offset can be read from the embedded MISB ST 0601 KLV timestamps (both files share one encoder clock): +893 ms at the start, +897 ms at the end — a telemetry-derived value that still deserves one hot-object confirmation.")
S += H("8.2 register.py — one homography for the whole recording", 2)
S += P("A pixel loss needs pixel (x, y) in A and B to look along the same ray. The two cameras differ in mount, focal length, FOV and resolution, so RGB is warped into the thermal grid with a 3×3 projective transform H (<font name='DV-M'>cv2.warpPerspective</font>, bilinear); thermal is never resampled (radiometric integrity; and RGB is the higher-resolution side, so downsampling it loses nothing). One H is exact if the cameras share a centre of projection or the scene is planar/distant; for a rigid pair with baseline b looking at range Z, the residual parallax is about")
S += eq(r"$d \approx \frac{f\,b}{Z}\ \mathrm{px}$")
S += fig("fig_parallax.png", 10.5, "Figure 7. Residual parallax of a single homography versus scene range for three baselines (f = 1444 px assumes 640 px across ~25° HFOV). Air-to-air and ground-from-altitude scenes are far below 1 px; near-field ground scenes are not — crop them away.")
S += BL(["<b>--mode manual</b> (recommended): ≥ 4 correspondences in native pixel coordinates (<font name='DV-M'>pick_points.py</font> enforces native coords), <font name='DV-M'>cv2.findHomography(..., RANSAC, 3.0)</font> — a 3-px reprojection threshold rejects a bad click.",
         "<b>--mode ecc</b> (the code's default): automatic. Since intensities are not comparable across modalities, it aligns Sobel gradient-magnitude maps of both images with <font name='DV-M'>cv2.findTransformECC</font> (affine model, initialised from a scale/letterbox guess, best of <font name='DV-M'>--calib-frames</font>). Fragile cross-modally.",
         "<b>--mode identity</b>: scale + letterbox only (co-boresighted optics).",
         "<b>--auto-crop</b>: the largest axis-aligned rectangle inside the warped-RGB footprint ∩ thermal frame (2-px margin) — no black borders reach the network. Every frame in <font name='DV-M'>--raw</font> is written; this stage never discards frames.",
         "<b>Split</b>: the frame list is cut into blocks of <font name='DV-M'>--chunk</font> (200) frames; n_val = max(1, round(val_frac · n_chunks)) blocks, evenly spaced (<font name='DV-M'>np.linspace</font>), go to val. Consecutive frames are near-duplicates; a random split would leak.",
         "<b>registration.json</b> (H, crop, ECC score, sizes) is the calibration record — keep it. (It was lost for the arsuf 316-pair set, which is why that calibration cannot be audited or reapplied.)"])
S += H("8.3 prep_aligned.py — datasets that already ship registered pairs", 2)
S += P("Walks a root, treats any directory holding <font name='DV-M'>RGB*</font> and <font name='DV-M'>NIR*</font> sub-folders as a session, pairs sub-folders by suffix (RGB-1 ↔ NIR-1) and files by name; crops each pair to its common size when they differ by ≤ <font name='DV-M'>--max-size-diff</font> (4) px — the off-by-one crops common in source sets — and <b>skips</b> larger mismatches as mis-registrations; writes IR as single-channel 8-bit (3-channel replicated greys collapsed), hard-links RGB; renames to <font name='DV-M'>&lt;session&gt;[_k]_&lt;index:06d&gt;.png</font> so lexicographic order is session-then-frame; splits by <b>whole session</b> (<font name='DV-M'>--val-sessions</font> explicit, or <font name='DV-M'>--val-frac</font> evenly spaced sessions); writes <font name='DV-M'>split.json</font>. On the agricultural set: 8 970 pairs → 7 331 train / 1 632 val, 7 pairs skipped (RGB/NIR sizes differed by 15–190 px), NIR's consistent +1 px cropped away, residual mis-registration measured at 0.3–0.7 px by phase correlation on edge maps.")
S += H("8.4 prep_unpaired.py and the unpaired loader", 2)
S += P("For CycleGAN the loaders draw A and B independently. <font name='DV-M'>prep_unpaired</font> downsizes RGB once to <font name='DV-M'>--width</font> (default the thermal width) and hard-links thermal; <font name='DV-M'>UnpairedDayThermalDataset</font> re-checks the width. Reason: a 256-px crop of a 3840-wide frame covers ~7 % of the field of view, of a 640-wide frame ~40 %; without matching, the discriminator separates the domains by <i>zoom level</i>, a shortcut that has nothing to do with modality. The train/val split uses the same temporal-chunk rule as register.py, applied independently to each domain.")
S += H("8.5 ThermalNorm — how pixels become network targets", 2)
S += eq(r"$x_{rel8} = \frac{p}{127.5} - 1, \qquad x_{abs16} = 2\,\frac{(p\cdot s + o) - T_{min}}{T_{max}-T_{min}} - 1$")
S += P("<b>rel8</b> (default): 8-bit pixels are relative intensities (a 16-bit file is crudely reduced by /257 first). <b>abs16</b>: radiometric counts p → °C via p·s + o (FLIR TLinear defaults s = 0.04 K/count, o = −273.15) → linear map of [T_min, T_max] (defaults −20 … 80 °C) onto [−1, 1]; the network then predicts calibrated temperatures and eval reports °C. Both are clipped to [−1, 1]. Inverses (<font name='DV-M'>norm_to_uint8</font>, <font name='DV-M'>norm_to_uint16</font>, <font name='DV-M'>norm_to_celsius</font>) are used by infer/eval. The choice must match the data: rel8 for AGC video, abs16 only for genuine radiometric 16-bit frames.")
S += H("8.6 One training sample (dataset.py)", 2)
S += NL(["Read RGB (<font name='DV-M'>IMREAD_COLOR</font>) and IR (<font name='DV-M'>IMREAD_UNCHANGED</font>, colour collapsed to grey); IR → [−1, 1] via ThermalNorm.",
         "<font name='DV-M'>--load-size L</font> (286): both squash-resized to max(L, crop)² with INTER_AREA — note this distorts aspect and discards resolution; <font name='DV-M'>--load-size 0</font>: native resolution, up-scaled only if smaller than the crop.",
         "A random <font name='DV-M'>--crop-size</font> (256) window — the <b>same window</b> from A and B — is cut (centre crop for validation).",
         "50 % horizontal flip on both. Then, on 80 % of samples, photometric jitter on <b>RGB only</b>: gain α ∈ [0.85, 1.15], bias β ∈ [−15, 15] (0–255 units), clipped. The IR target is never perturbed — it is the quantity being regressed.",
         "RGB → [−1, 1]; T = <font name='DV-M'>--temps-csv</font> value through <font name='DV-M'>celsius_to_norm</font> if present for that file stem, else mean(B) of the crop."])

S += H("9. Networks as built (networks.py)", 1)
S += TB([["Level (crop 256)", "Encoder op", "Encoder out", "Decoder op (input → output)", "Decoder out"],
         ["256² → 128²", "Conv4×4 s2 (3→64)", "64 @128²", "ConvT(128→64) → IN → ReLU → Conv3×3(64→1) → tanh  (extra head)", "1 @256²"],
         ["128² → 64²", "LReLU, Conv(64→128), IN", "128 @64²", "ReLU, ConvT(256→64), IN", "64 @128² (+skip 64 → 128)"],
         ["64² → 32²", "LReLU, Conv(128→256), IN", "256 @32²", "ReLU, ConvT(512→128), IN", "128 @64² (+skip → 256)"],
         ["32² → 16²", "LReLU, Conv(256→512), IN", "512 @16²", "ReLU, ConvT(1024→256), IN", "256 @32² (+skip → 512)"],
         ["16² → 8², 8² → 4², 4² → 2²", "LReLU, Conv(512→512), IN  (×3; dropout 0.5 optional)", "512", "ReLU, ConvT(1024→512), IN  (×3)", "512 (+skip → 1024)"],
         ["2² → 1² (innermost)", "LReLU, Conv(512→512)", "512 @1²", "ReLU, ConvT(512→512), IN", "512 @2² (+skip → 1024)"]], widths=[3.0, 4.2, 2.0, 5.6, 2.8])
S += P("Parameters (measured by instantiation): generator 54.54 M with the extra head (54.41 M without); input channels 3 (pix2pix, cyclegan G), 4 (two_stage G₁ = RGB + T-plane, G₂ = RGB + Ŝ), 12 with the VAE (+8 z-planes); cyclegan F is 1→3. <font name='DV-M'>--crop-size</font> must be a power of two ≥ 32 because num_downs = log₂(crop). At inference the input is reflect-padded to a multiple of 2<sup>num_downs</sup>.")
S += TB([["PatchGAN layer", "kernel/stride", "channels", "receptive field", "cumulative stride"],
         ["1", "4×4 / 2", "in → 64", "4", "2"], ["2", "4×4 / 2", "64 → 128 (+IN)", "10", "4"], ["3", "4×4 / 2", "128 → 256 (+IN)", "22", "8"],
         ["4", "4×4 / 1", "256 → 512 (+IN)", "46", "8"], ["5 (output)", "4×4 / 1", "512 → 1 logit", "70", "8"]], widths=[3.0, 2.6, 3.6, 3.0, 3.6])
S += P("2.77 M parameters; 30×30 logits for a 256² input; no sigmoid (LSGAN regresses the raw logit, vanilla uses BCE-with-logits). Two scales: 5.53 M. <font name='DV-M'>TemperatureEncoder</font> (VAE only): 5 stride-2 convs, global average pool, two linear heads → μ, log σ² of dimension nz. All conv/linear weights N(0, 0.02), biases 0 (<font name='DV-M'>init_weights</font>). Checkpoint sizes follow from the counts (fp32 weights + Adam m and v): pix2pix with a 2-scale D = 60.1 M params → 0.72 GB (measured 720 866 033 bytes), two_stage 114.6 M → 1.4 GB, cyclegan 120.1 M → 1.44 GB.")

S += H("10. The training loops, line by line (train.py)", 1)
S += H("10.1 Common machinery", 2)
S += P("One Adam optimiser per network, lr 2e-4, β = (0.5, 0.999); LambdaLR schedule stepped once per epoch; DataLoader with shuffle, <font name='DV-M'>drop_last=True</font>, <font name='DV-M'>--num-workers</font> (4); seed via <font name='DV-M'>set_seed</font>. <font name='DV-M'>--out/config.json</font> is written at start with all args and the thermal normalisation.")
S += eq(r"$\eta(e) = \eta_0 \cdot \max\!\left(0,\; 1 - \frac{\max(0,\, e+1-N_{\mathrm{const}})}{N_{\mathrm{decay}}}\right)$")
S += fig("fig_lr.png", 10.5, "Figure 8. The learning-rate schedule as coded (<font name='DV-M'>lr_lambda</font>): flat for --n-epochs, then linear decay; note that the factor reaches exactly 0 in the last decay epoch, so that epoch does no learning (harmless; the original pix2pix uses N_decay + 1 in the denominator).")
S += H("10.2 --mode pix2pix (step_pix2pix)", 2)
S += CD("""fake = G(A)
# --- discriminator step (G frozen by detach) ---
loss_D = 0.5 * ( gan(D(cat[A, fake.detach()]), False) + gan(D(cat[A, B]), True) )
loss_D.backward(); opt_D.step()
# --- generator step (D frozen via set_requires_grad) ---
loss_G = gan(D(cat[A, fake]), True) + lambda_l1 * L1(fake, B)
loss_G.backward(); opt_G.step()
# printed: D=loss_D  G_gan=adversarial term  G_L1=100*L1(fake,B)   (all on the [-1,1] scale)""")
S += eq(r"$\mathcal{L}_D = \frac{1}{2}\left[(D(A,\hat B)-0)^2 + (D(A,B)-1)^2\right], \qquad \mathcal{L}_G = (D(A,\hat B)-1)^2 + 100\,\|\hat B - B\|_1$")
S += P("The <font name='DV-M'>detach()</font> cuts the graph so D's loss cannot move G; <font name='DV-M'>set_requires_grad(D, False)</font> during the G step avoids computing D gradients and prevents G's loss from moving D. The ½ halves D's effective learning rate. Note that <font name='DV-M'>G_L1</font> in the log is 100 × the mean absolute error on the [−1, 1] scale: G_L1 = 12 means MAE 0.12, i.e. 6 % of the full range, or ≈ 15 grey levels of an 8-bit target.")
S += H("10.3 --mode two_stage (step_two_stage)", 2)
S += CD("""S = gaussian_blur(B, sigma);  R = B - S;  Tpl = T broadcast to 1xHxW
in1 = cat[A, Tpl]  (+ z-planes from E(S) if --use-vae)
S_hat = G1(in1)
# stage 1
loss_D1 = 0.5*( gan(D1(cat[A,Tpl,S_hat.detach()]),False) + gan(D1(cat[A,Tpl,S]),True) )   -> step D1
loss_G1 = gan(D1(cat[A,Tpl,S_hat]),True) + 100*L1(S_hat,S) + lambda_temp*|mean(S_hat)-T|  (+ lambda_kl*KL)  -> step G1 (+E)
# stage 2, on the DETACHED prediction
R_hat = G2(cat[A, S_hat.detach()])
loss_D2 = 0.5*( gan(D2(cat[A,R_hat.detach()]),False) + gan(D2(cat[A,R]),True) )   -> step D2
loss_G2 = gan(D2(cat[A,R_hat]),True) + 100*L1(R_hat,R)                            -> step G2
B_hat = clamp(S_hat.detach() + R_hat.detach(), -1, 1)
# printed: D1 G1_gan G1_L1 G1_T [G1_kl] D2 G2_gan G2_L1""")
S += H("10.4 --mode cyclegan (step_cyclegan)", 2)
S += CD("""fake_B = G(A); rec_A = F(fake_B); fake_A = F(B); rec_B = G(fake_A)
# generators (both D frozen)
loss_G = gan(D_B(fake_B),True) + gan(D_A(fake_A),True)
       + lambda_cyc*( L1(rec_A,A) + L1(rec_B,B) )
       + lambda_idt*( L1(G(B.expand(3ch)),B) + L1(F(A.mean(ch)),A) )      # if lambda_idt > 0
# discriminators, on pooled fakes (ImagePool of --pool-size)
loss_D = 0.5*(gan(D_B(B),True)+gan(D_B(pool_B(fake_B)),False)) + 0.5*(gan(D_A(A),True)+gan(D_A(pool_A(fake_A)),False))
# printed: D G_gan G_cyc G_idt""")
S += H("10.5 Validation, samples, checkpoints, resume", 2)
S += BL(["<b>val L1</b> each epoch on up to <font name='DV-M'>--val-max-batches</font> (50) batches of centre crops: paired modes report mean |prediction − B| on the [−1, 1] scale (two_stage with the T oracle); cyclegan reports the cycle reconstruction |F(G(A)) − A|, explicitly ‘not a fidelity metric’. Compare with eval.py by halving.",
         "<b>Sample strips</b> every <font name='DV-M'>--log-every</font> iterations (first 4 images of the batch): pix2pix RGB | real | generated; two_stage RGB | real | S | Ŝ | B̂; cyclegan RGB | unrelated real IR | G(RGB) | F(G(RGB)) | F(IR).",
         "<b>Checkpoints</b>: <font name='DV-M'>latest.pt</font> every epoch (overwritten — do not read it mid-write), <font name='DV-M'>epoch_NNN.pt</font> every <font name='DV-M'>--save-freq</font>; contents: epoch, args, thermal_norm, all nets, all optimiser states → infer.py rebuilds the exact architecture from it.",
         "<b>--resume</b> restores nets, optimisers, epoch counter and fast-forwards the lr schedulers (otherwise a resumed run would restart the decay)."])

S += H("11. Inference and evaluation", 1)
S += H("11.1 infer.py", 2)
S += P("Rebuilds G (or G₁ + G₂) from the checkpoint args, reflect-pads the input to a multiple of 2<sup>num_downs</sup> (chunked so each pad is smaller than the current size), translates, un-pads → output geometry = input geometry, one 8-bit PNG per input with the same name; <font name='DV-M'>--save-16bit</font> adds <font name='DV-M'>_16bit.png</font> (calibrated counts in abs16, a plain 16-bit stretch in rel8). <font name='DV-M'>--temps</font> (two_stage only): each value is mapped to a T-plane — °C through <font name='DV-M'>celsius_to_norm</font> in abs16, t·2 − 1 for t ∈ [0, 1] in rel8 — and written to its own sub-folder (<font name='DV-M'>T15C/</font>, <font name='DV-M'>T0.5/</font>); without --temps, T = 0 (mid-range). Ignored for pix2pix/cyclegan. <font name='DV-M'>--max-size</font> caps the long side first; device is CUDA if available unless <font name='DV-M'>--cpu</font>.")
S += H("11.2 eval.py", 2)
S += P("Matches files by stem (dropping a <font name='DV-M'>_16bit</font> suffix), resizes the prediction to the ground-truth size if needed, maps both through the same ThermalNorm and then to [0, 1], and reports per-image averages of L1, RMSE, PSNR = 10·log₁₀(1/MSE) and SSIM (single channel, 11×11 Gaussian window σ = 1.5):")
S += eq(r"$\mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2+\mu_y^2 + C_1)(\sigma_x^2+\sigma_y^2 + C_2)},\quad C_1 = 0.01^2,\; C_2 = 0.03^2$")
S += P("With <font name='DV-M'>--thermal-mode abs16</font> it also reports L1_degC = L1 · (T_max − T_min) and RMSE_degC likewise. Because it needs aligned ground truth, it exists only for the paired routes.")
S += H("11.3 check_thermalness.py", 2)
S += P("Samples n frames, translates them (T = 0), computes r_gen = corr(G(RGB), luminance(RGB)) and, where an unregistered thermal twin exists, r_real = corr(real, luminance). Verdicts: r_gen &gt; 0.75 greyscale shortcut (‘more epochs will not fix this — change the objective’), 0.5–0.75 partial, &lt; 0.5 decoupled. It is a cheap sanity check for unpaired models, not a fidelity metric.")

# ================================================================== PART III
S += [PageBreak()] + H("Part III — Data: what you must know", 0)
S += P("A GAN's output can only be as good as the correspondence in its training pairs. In practice most failed translation runs are data failures that no hyper-parameter can fix. The items below are ordered by how often they are the real problem.")
S += H("D1. Pixel alignment (registration) — the assumption everything rests on", 2)
S += P("The L1 term compares pixel (x, y) of the prediction with pixel (x, y) of the target. If the pair is misaligned by k pixels, the ‘correct’ target for an edge is a smeared edge; L1 pushes toward that smear while D pushes toward sharpness, and the two fight — outputs come out either blurry (L1 wins) or sharp-but-shifted with ghost edges (D wins). Rule of thumb: sub-pixel to ~1 px residual is harmless; a few pixels visibly hurt high-frequency fidelity; tens of pixels make pairs poison. Check it: blend the warped RGB over the IR, phase-correlate Sobel edge maps (the agricultural set: 0.3–0.7 px), keep <font name='DV-M'>registration.json</font>. Time misalignment is spatial misalignment for anything that moves — see D2.")
S += H("D2. Temporal synchronisation", 2)
S += P("Different frame rates (30 vs 25 fps in the arsuf streams) and different start times mean the offset must be calibrated: hot-object event, or shared-clock telemetry (KLV). Precision is ± half a frame period plus offset error; NUC/FFC frozen thermal frames must be dropped; the constant-fps assumption must hold (check PTS spacing — arsuf: thermal exactly 40 ms, day 33 ± 5 ms jitter, no drops).")
S += H("D3. What the target physically is", 2)
S += BL(["<b>Thermal vs NIR.</b> Emitted radiance versus reflected light. A NIR pair set trains an RGB→NIR reflectance model that cannot stand in for thermal (agricultural set = NIR). Confirm the band from the sensor, not the folder name.",
         "<b>Radiometric vs AGC.</b> 8-bit video from an uncooled core has been through automatic gain: the same temperature maps to different grey levels from frame to frame → the regression target is <i>non-stationary</i>. Lock the AGC span, or record radiometric 16-bit and use abs16 — that is what makes ThermalGAN's calibrated-°C output possible. rel8 models learn relative appearance only.",
         f"<b>Saturation.</b> The arsuf 8-bit set touches 0 and 255 on essentially every frame (stretched encode); the agricultural NIR barely does ({FD['target_frac_0_255'][0]*100:.2f} % of pixels at 0, {FD['target_frac_0_255'][1]*100:.2f} % at 255 in 300 random train frames). Saturated targets are unrecoverable and clip R in two_stage."])
S += fig("fig_target_stats.png", 16.5, "Figure 9. Know your target: histogram of NIR values over 300 random training frames (left) and the mean target level per session (right). Sessions differ by exposure and canopy state — precisely why validation must hold out whole sessions.")
S += H("D4. Diversity beats quantity", 2)
S += P("ThermalGAN's own dataset lesson: training on its narrow ReID split (15 k pairs, one mall) generalised poorly; the fix was the varied VOC split (5 k pairs, four cities, all seasons, −20 … +40 °C), not more frames of the same scene. Consecutive video frames are nearly the same sample. The arsuf aligned set (316 pairs, one continuous stream, one scene) is a plumbing check, not supervision for a deployable translator; the agricultural set (15 sessions, 4 crops, 3 years, 8 970 pairs) is a real dataset. Count <i>sessions × conditions</i>, not frames.")
S += H("D5. Split hygiene", 2)
S += P("Never shuffle frames across train/val. register.py splits by contiguous chunks, prep_aligned by whole sessions; hold out entire flights for the number you report. A leaked split gives beautiful validation curves and a model that fails on the next flight. (The arsuf set's val = 200 &gt; train = 116 is a side effect of chunk = 200 on 316 frames — recorded, not endorsed.)")
S += H("D6. Resolution, field of view and crop", 2)
S += BL(["<font name='DV-M'>--load-size 286</font> squash-resizes a 1098×798 frame to 286×286 — aspect distortion plus a ~4× loss of resolution before the 256 crop; fine for facades, wrong when the fine texture is the content. <font name='DV-M'>--load-size 0</font> trains on native crops.",
         "The 70-px patch and the 256-px crop set the scale D and G reason about. Objects much larger than the crop are only ever seen in pieces (multi-scale D and larger crops help); objects of a few pixels are dominated by background in L1 (train on crops around them if boxes exist).",
         "Unpaired training must match angular scale between domains (<font name='DV-M'>--rgb-width</font>), or D learns zoom level.",
         "At inference the padded border (reflect) can show artefacts; crop or overlap-tile if it matters."])
S += H("D7. The temperature condition (two_stage only)", 2)
S += P("With no <font name='DV-M'>--temps-csv</font>, T is the crop's mean target — an oracle at training and validation, a guess at deployment. If you want physically meaningful control, log ambient temperature (or a proxy) per frame and pass it; otherwise treat T as an abstract ‘level’ knob and expect the val number to flatter the model.")
S += H("D8. Pairing mechanics", 2)
S += P("Same filename in <font name='DV-M'>rgb/</font> and <font name='DV-M'>thermal/</font>; PNG (any of .png/.jpg/.tif/.bmp are listed); IR may be 8- or 16-bit, colour IR is collapsed to grey; with <font name='DV-M'>--load-size 0</font> the two files must have identical size (prep_aligned guarantees it; a 1-px mismatch crashes the collate). Names sort lexicographically — zero-pad indices.")
S += H("D9. How much data", 2)
S += P("pix2pix works from a few hundred pairs (400 facades) <i>within</i> a narrow domain; generalising across conditions needs the diversity of D4. For a first thermal model aim for thousands of pairs across several flights, times of day and backgrounds; for the agricultural NIR set 7 k pairs of 4 crops is comfortable, and the model already scores well after 6 epochs (§14).")
S += H("D10. Downstream use and hallucination", 2)
S += P("Synthetic IR is a prior, not a measurement. Before trusting a set for detector training: audit false-positive hotspots on synthetic vs real, run detector-in-the-loop (train on synthetic, test on a real IR hold-out from unseen flights), and prefer conservative settings (higher λ_L1) for radiometric use.")
S += H("The two datasets in this repository — the facts", 2)
S += TB([["", "arsuf day + thermal (streams → 316 aligned pairs)", "agricultural RGB → NIR (ir-rgb-dataset)"],
         ["modality of B", "LWIR thermal, 8-bit AGC video (H.264), 640×512 @25 fps", "NIR reflectance, 8-bit, stored as 3-ch replicated grey"],
         ["input A", "3840×2160 @30 fps H.264 (jitter ±5 ms, no drops)", "RGB 1098×798 (a tail of smaller frames)"],
         ["registration", "one homography, manual points, auto-crop → 460×445; registration.json lost", "shipped registered; NIR +1 px cropped; 0.3–0.7 px residual"],
         ["sync", "KLV-derived offset +893 ms (validate)", "same instant by construction"],
         ["pairs / split", "316: 116 train / 200 val (chunk 200), first 316 raw pairs of 4 897", "8 970 → 7 331 / 1 632, 4 held-out sessions, 7 skipped"],
         ["saturation", "0 and 255 on every frame", "≈0.1 % of pixels"],
         ["right mode", "pix2pix baseline → two_stage (T = mean(B) unless °C logged); rel8", "pix2pix; rel8; --load-size 0"],
         ["what it is good for", "bring-up / plumbing; too narrow for deployment", "a real RGB→NIR model; not a thermal model"]], widths=[2.6, 7.2, 7.2])

# ================================================================== PART IV
S += [PageBreak()] + H("Part IV — Parameters, tuning, and what to expect", 0)
S += H("12. Parameter reference (what each flag does in the code, and when to move it)", 1)
S += TB([["Flag [default]", "Effect (as coded)", "When / how to change; symptoms"],
         ["--mode [pix2pix]", "pix2pix | two_stage | cyclegan (§10)", "pix2pix for one-to-one targets (NIR) and as the thermal baseline; two_stage for thermal with controllable level; cyclegan only without registration"],
         ["--load-size [286] / --crop-size [256]", "squash to L² then random crop; 0 = native crops; crop must be a power of two ≥ 32 and sets num_downs", "native crops when texture matters; larger crop (512) for more context at 4× cost; smaller crop for tiny GPUs"],
         ["--batch-size [8]", "images per step; drop_last", "GPU-bound (pix2pix 256/2-scale D ≈ 3.8 GB at 8); with InstanceNorm batch size does not change normalisation statistics"],
         ["--lambda-l1 [100]", "weight of ‖B̂ − B‖₁ in G's loss (also G1/G2 in two_stage)", "the single most important knob: raise (150–200) for fidelity/less hallucination, lower (30–50) if outputs are soft and D loss stays high; symptoms of too high: blur, muted texture; too low: sharp but unfaithful, unstable"],
         ["--gan-mode [lsgan]", "MSE-to-target vs BCE-with-logits adversarial loss", "keep lsgan; vanilla only to reproduce the paper"],
         ["--num-scales-d [1]", "extra PatchGAN on 2× pooled input", "2 when structures larger than 70 px matter (used in both recorded runs)"],
         ["--n-layers-d [3]", "PatchGAN depth → receptive field 70 (3), 142 (4), 34 (2)", "rarely; larger RF for very large-scale texture"],
         ["--ngf / --ndf [64]", "base width of G / D", "halve for memory (quality drops); doubling is rarely worth it"],
         ["--norm [instance]", "IN / BN / none in G and D", "try batch or none only for calibrated-level experiments; BN needs batch ≥ 8"],
         ["--dropout [off]", "0.5 dropout in the inner 512-ch U-Net blocks", "mild regulariser on very small sets; also the only ‘noise’ pix2pix ever used"],
         ["--no-extra-head", "removes ConvT(128→64)→Conv3×3 head", "keep the head (paper's extra layer); flag exists for ablation"],
         ["--lr [2e-4], --beta1 [0.5], --beta2 [0.999]", "Adam for every net", "do not raise lr; β₁ 0.5 is the stability setting; NaNs → lower lr"],
         ["--n-epochs [100] + --n-epochs-decay [100]", "flat then linear-to-zero epochs", "scale to dataset size: 7 k pairs converge in tens of epochs; watch val L1 plateau; the last decay epoch has lr 0"],
         ["--thermal-mode [rel8], --tmin/--tmax [−20/80], --raw-scale/--raw-offset", "target normalisation (§8.5)", "abs16 only for radiometric 16-bit; set tmin/tmax to the real range (a wide range wastes tanh resolution)"],
         ["--temps-csv [none]", "T from logged °C per file stem", "always, if you have it (two_stage)"],
         ["--lowpass-sigma [8]", "σ of the base S at crop scale (radius 3σ)", "larger σ → more goes to R (stage 2 does more, ambient in S only); smaller → S carries structure; 6–12 is the sensible band; ablate on val"],
         ["--lambda-temp [10]", "weight of |mean(Ŝ) − T|", "raise if sweeping T at inference barely changes the output level; lower if S loses structure"],
         ["--use-vae, --nz [8], --lambda-kl [0.01]", "BicycleGAN-style latent on G₁", "experimental; only if T alone cannot express the variability"],
         ["--lambda-cyc [10], --lambda-idt [5], --pool-size [50]", "cyclegan cycle, identity, D history buffer", "identity 0 if check_thermalness reports a shortcut; lower cyc if the mapping is too conservative; pool 0 to disable damping"],
         ["--rgb-width [thermal width], --skip-first [0], --val-frac [0.1], --chunk [200]", "unpaired data handling", "match angular scale; skip lead-in ground footage; chunk relative to how fast the scene changes"],
         ["--val-max-batches [50], --log-every [100], --save-freq [10]", "bookkeeping", "raise val batches for smoother curves; samples are the fastest diagnostic"],
         ["--seed [0], --num-workers [4], --resume, --max-iters", "reproducibility, loader, resume, debug cap", "different seeds change fine texture, not the level of quality; num_workers ≈ CPU cores"],
         ["infer: --temps, --max-size, --save-16bit, --cpu", "T sweep, resolution cap, 16-bit out, device", "generate at the training scale (native if trained native)"]], widths=[3.6, 5.6, 7.8])

S += H("13. A tuning procedure that respects the order of importance", 1)
S += NL(["<b>Data first</b> (Part III): verify alignment on a blend, the target's modality and bit depth, saturation, and that val holds out whole sessions/flights. No parameter compensates for a bad pair.",
         "<b>Smoke run</b>: <font name='DV-M'>--max-iters 60 --n-epochs 1 --n-epochs-decay 0</font>, then infer + eval on ~100 val images. It catches shape/name/bit-depth problems in a minute and gives a floor to beat (agricultural set: L1 0.098 / SSIM 0.66 after 60 iterations).",
         "<b>Baseline</b>: pix2pix, LSGAN, λ_L1 = 100, InstanceNorm, 2-scale D, native crops if texture matters. Train until val L1 flattens; keep the samples.",
         "<b>Fidelity vs sharpness</b>: move λ_L1 (50 / 100 / 200) and compare eval L1/SSIM <i>and</i> the samples — the metric prefers blur, your downstream task usually does not. Decide with the task metric when you have one.",
         "<b>Scale</b>: if large structures are wrong, add the multi-scale D or a 512 crop; if small objects vanish, crop around them.",
         "<b>Thermal only</b>: switch to two_stage; ablate σ (6/8/12) and λ_temp; check that sweeping T actually moves the output level; use --temps-csv if you can.",
         "<b>Unpaired only</b>: run check_thermalness after a few epochs; if r &gt; 0.75, drop λ_idt, lower λ_cyc — do not train longer.",
         "<b>Length</b>: for ~7 k pairs, tens of epochs; for ~300 pairs, hundreds (and expect overfitting — watch val). Use --resume rather than restarting.",
         "<b>Repeat with 2–3 seeds</b> before concluding a change helped: adversarial training has run-to-run variance."])

S += H("14. Reading the training signals — what a healthy run looks like", 1)
S += P("Numbers below are from <font name='DV-M'>runs/nir_p2p</font> (agricultural RGB→NIR, pix2pix, LSGAN, 2-scale D, native 256 crops, batch 8, 916 iterations/epoch, ≈0.23 s/iteration on an RTX 5050 Laptop GPU, ≈215 s/epoch), which ran for 6 epochs before the machine restarted and was not resumed.")
S += fig("fig_curves.png", 17, "Figure 10. Left: the adversarial terms. With LSGAN and 0/1 targets, a maximally confused D scores ½·(0.5² + 0.5²) = 0.25 and G's adversarial term is also 0.25 — both hover around it after the first hundred iterations, which is what equilibrium looks like. Middle: G's L1 (÷100) per batch. Right: validation L1 per epoch on the held-out sessions, 0.143 → 0.108 and still falling.")
S += BL(["<b>D loss</b> (LSGAN): starts high, drops within ~200 iterations to 0.2–0.5 and stays noisy there. D → 0 with G_gan rising = D winning (reduce D's capacity/scales, or raise λ_L1); D stuck ≈ 0.5 with G_gan ≈ 0 = D too weak/G collapsed to something D cannot judge.",
         "<b>G_gan</b>: order 0.3–0.9 in this run — larger than the L1 term's gradient only rarely; that is intended (§4.1).",
         "<b>G_L1</b> = 100 × MAE ([−1, 1]): 41 at iteration 0 (untrained tanh output), 10–18 after the first epoch (MAE 0.10–0.18, i.e. 13–23 grey levels), trending down; per-batch noise of ±0.03 is normal because crops differ.",
         "<b>val L1</b> should fall monotonically-ish and then flatten; rising val with falling train = overfitting to the training sessions.",
         "<b>Sample strips</b> are the fastest diagnostic: modality relationship (vegetation bright / soil dark here; sky cold / engines hot for thermal) appears within a few hundred iterations, texture takes epochs, tiling/checkerboard patterns or identical outputs across rows are failure signatures."])
S += fig("fig_samples.png", 17, f"Figure 11. Real sample strips from the run: at iteration 0 (left) the untrained generator emits noise; by <font name='DV-M'>{FD['last_sample'][:-4]}</font> (right) the outputs are close to the real NIR. Columns: RGB | real NIR | generated.")
S += fig("fig_fullframe.png", 17, f"Figure 12. The epoch-6 checkpoint on two full-resolution held-out frames (CPU inference, 1098×798): L1 = {FD['fullframe'][0][1]:.3f} / SSIM = {FD['fullframe'][0][2]:.3f} (wheat, unseen session) and L1 = {FD['fullframe'][1][1]:.3f} / SSIM = {FD['fullframe'][1][2]:.3f} (drybean, unseen session), on the [0, 1] scale. Errors concentrate on fine canopy texture and plot edges — the high-frequency residue L1 cannot resolve and D has not yet fully sharpened.")
S += H("14.1 What to expect for better results", 2)
S += BL(["<b>On the agricultural set</b> (measured up to epoch 6; the rest is expectation): val L1 was 0.108 ([−1, 1]) and full-frame L1 0.04–0.05 / SSIM 0.82–0.88 ([0, 1]) at epoch 6 of 60. A converged run should push full-frame L1 to roughly 0.03–0.04 and SSIM toward 0.9; the sanity-check model (60 iterations) sat at L1 0.098 / SSIM 0.66. Beyond that point the differences are in texture sharpness — judge them on the strips and on whatever downstream task consumes the NIR.",
         "<b>On thermal data</b> the bar is different: the paper's baselines reach ≈5 °C absolute error, useful fine contrast needs ≈1 °C; with 8-bit AGC targets you cannot state °C at all, only relative fidelity (L1/SSIM in [0, 1]) — and the decisive number is detector-in-the-loop.",
         "<b>Improvements come from, in order:</b> better/more diverse pairs (D4), correct scale (native crops, multi-scale D), λ_L1 tuned to the task, then epochs. Architecture changes rank last.",
         "<b>Signs you are done</b>: val L1 flat for many epochs, samples indistinguishable in structure from the target, D and G_gan hovering near 0.25 without trends, and — for thermal — a T sweep that moves the level without changing structure."])
S += H("15. Metrics: what the numbers mean", 1)
S += TB([["Metric (eval.py, [0, 1] scale)", "Meaning", "Reading it"],
         ["L1", "mean |pred − gt|", "× 255 = grey levels; × (T_max − T_min) = °C in abs16; 0.04 = 10 grey levels"],
         ["RMSE", "√mean (pred − gt)²", "penalises large local errors (hot spots, edges) more than L1"],
         ["PSNR", "10·log₁₀(1/MSE)", "20 dB ≈ RMSE 0.1; +6 dB = RMSE halved; saturates in meaning above ~35 dB"],
         ["SSIM", "structural similarity, 11×11 Gaussian window", "sensitive to texture/edge structure L1 ignores; 0.66 → 0.82–0.88 between the 60-iteration and epoch-6 models"],
         ["train-log val L1", "MAE on [−1, 1], centre crops, ≤ 400 images", "≈ 2 × eval L1; optimistic for two_stage (T oracle)"],
         ["r_gen (check_thermalness)", "corr(G(RGB), luminance)", "&gt; 0.75 shortcut; &lt; 0.5 decoupled; only meaningful for unpaired models"],
         ["detector-in-the-loop", "train detector on synthetic, test on real IR", "the metric that answers the actual question; audit false positives"]], widths=[4.0, 4.6, 8.4])

# ================================================================== Appendices
S += [PageBreak()] + H("Appendix A — Command quick reference", 1)
S += CD("""# ---- pre-registered pair folders (agricultural RGB->NIR) ----
python -m day2thermal.prep_aligned --root ir-rgb-dataset --rgb-prefix RGB --ir-prefix NIR \\
    --out data/agri_rgb_nir/aligned \\
    --val-sessions canola_06082019,drybean-30072020,Lentwheat_29082018,wheat_27072019
python -m day2thermal.train --data-root data/agri_rgb_nir/aligned --out runs/nir_p2p \\
    --mode pix2pix --load-size 0 --crop-size 256 --batch-size 8 --num-scales-d 2 \\
    --norm instance --gan-mode lsgan --lambda-l1 100 --thermal-mode rel8 \\
    --n-epochs 40 --n-epochs-decay 20 --save-freq 10 --log-every 200 --val-max-batches 50 --num-workers 4 --seed 0
#   (resume: add --resume runs/nir_p2p/checkpoints/latest.pt)
python -m day2thermal.infer --ckpt runs/nir_p2p/checkpoints/latest.pt --input data/agri_rgb_nir/aligned/val/rgb --out preds/nir_p2p_val
python -m day2thermal.eval  --pred preds/nir_p2p_val --gt data/agri_rgb_nir/aligned/val/thermal --out-json preds/nir_p2p_val/metrics.json
python -m day2thermal.infer --ckpt runs/nir_p2p/checkpoints/latest.pt --input /path/to/rgb_images --out /path/to/synthetic_nir

# ---- paired video (day + thermal) ----
python -m day2thermal.extract_frames --rgb day.ts --thermal thermal.ts --out data/raw --fps 7 --offset-ms 893 --drop-static-thermal
python -m day2thermal.register --raw data/raw --out tmp --dump-pair 0            # then pick >=4 points -> calib/points.json
python -m day2thermal.register --raw data/raw --out data/aligned --mode manual --points calib/points.json --auto-crop --val-frac 0.1
python -m day2thermal.train --data-root data/aligned --out runs/p2p    --mode pix2pix  --crop-size 256 --batch-size 8
python -m day2thermal.train --data-root data/aligned --out runs/2stage --mode two_stage --crop-size 256 --lowpass-sigma 8 --lambda-temp 10 [--temps-csv weather.csv]
python -m day2thermal.infer --ckpt runs/2stage/checkpoints/latest.pt --input big_rgb_set/ --out synth_thermal --temps 0,15,30 --save-16bit
python -m day2thermal.eval  --pred preds/ --gt data/aligned/val/thermal   [--thermal-mode abs16 --tmin -20 --tmax 80]

# ---- unpaired fallback ----
python -m day2thermal.prep_unpaired --raw data/raw --out data/unpaired --width 640 --skip-first 900
python -m day2thermal.train --data-root data/unpaired --out runs/cyc --mode cyclegan --load-size 286 --crop-size 256 --batch-size 4 \\
    --lambda-cyc 10 --lambda-idt 5 --pool-size 50 --num-scales-d 2
python -m day2thermal.check_thermalness --ckpt runs/cyc/checkpoints/latest.pt --data-root data/unpaired --n 24""")

S += H("Appendix B — Glossary", 1)
S += TB([["Term", "Meaning"],
         ["Adversarial loss", "The term that scores G through D's verdict; here LSGAN: (D(A, B̂) − 1)² for G"],
         ["AGC", "Automatic gain control of an IR core: per-frame stretch of the 8-bit output — makes 8-bit thermal non-stationary"],
         ["cGAN", "Conditional GAN: D sees the input A together with the candidate B"],
         ["Cycle consistency", "F(G(A)) ≈ A and G(F(B)) ≈ B — CycleGAN's substitute for paired supervision"],
         ["ImagePool", "Buffer of past fakes shown to D to damp oscillation (cyclegan)"],
         ["InstanceNorm", "Per-sample, per-channel standardisation of feature maps"],
         ["JSD", "Jensen–Shannon divergence; what the original GAN generator minimises at equilibrium"],
         ["KLV / MISB 0601", "Telemetry stream in UAV MPEG-TS; carries per-packet timestamps and sensor position"],
         ["LSGAN", "Least-squares GAN: squared loss to targets 1/0 instead of cross-entropy"],
         ["Mode collapse", "G producing few distinct outputs regardless of input/latent"],
         ["NUC / FFC", "Non-uniformity correction / flat-field correction: shutter events that freeze thermal video"],
         ["PatchGAN", "Fully-convolutional D whose logits each judge one 70×70 patch"],
         ["Radiometric", "IR data calibrated to temperature (16-bit counts → °C)"],
         ["rel8 / abs16", "ThermalNorm modes: relative 8-bit vs absolute-temperature 16-bit"],
         ["S / R (base / residual)", "Low-pass of the target and its residual; the two_stage decomposition"],
         ["T-plane", "The scalar temperature condition broadcast to a constant image plane and concatenated to RGB"],
         ["U-Net", "Encoder–decoder with skip connections between equal resolutions"]], widths=[3.6, 13.4])

S += H("Appendix C — References", 1)
S += BL(["Kniaz, Knyaz, Hladůvka, Kropatsch, Mizginov. ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-Identification in Multispectral Dataset. ECCV 2018 Workshops, LNCS 11134, 606–624.",
         "Goodfellow et al. Generative Adversarial Nets. NeurIPS 2014.",
         "Isola, Zhu, Zhou, Efros. Image-to-Image Translation with Conditional Adversarial Networks (pix2pix). CVPR 2017.",
         "Mao et al. Least Squares Generative Adversarial Networks. ICCV 2017.",
         "Wang et al. High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (pix2pixHD). CVPR 2018.",
         "Zhu, Zhang, Pathak, Darrell, Efros, Wang, Shechtman. Toward Multimodal Image-to-Image Translation (BicycleGAN). NeurIPS 2017.",
         "Zhu, Park, Isola, Efros. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN). ICCV 2017.",
         "Radford, Metz, Chintala. Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN). ICLR 2016.",
         "Arjovsky, Bottou. Towards Principled Methods for Training Generative Adversarial Networks. ICLR 2017.",
         "Shrivastava et al. Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN; history buffer). CVPR 2017.",
         "Wang, Bovik, Sheikh, Simoncelli. Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM). IEEE TIP 2004.",
         "Saharia et al. Palette: Image-to-Image Diffusion Models. SIGGRAPH 2022; Li et al. BBDM. CVPR 2023; Zhang, Rao, Agrawala. ControlNet. ICCV 2023.",
         "Repository: github.com/roeytoo-111/day2thermal (commit bec21e4); research/DayToThermal_Research_Report.md for the literature review and LWIR physics."])

doc = Doc(OUT, title="RGB → IR Image Translation with GANs — day2thermal explained", author="day2thermal")
doc.multiBuild(S)
print("wrote", OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")
