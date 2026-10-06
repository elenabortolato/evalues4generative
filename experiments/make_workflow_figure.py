"""Draw the workflow scheme of the procedure (Algorithm 1) as figures/workflow.png.

The diagram is laid out in HTML with absolutely positioned boxes, the
mathematics is typeset with the KaTeX copy vendored under paper/vendor/katex,
the arrows are an SVG layer, and Chromium (via Playwright) renders the page
to a PNG at three times screen resolution.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
KATEX = (ROOT / "paper" / "vendor" / "katex").resolve().as_uri()
OUT = ROOT / "figures" / "workflow.png"

W, H = 1180, 520
INK, INK2, LINE = "#0b0b0b", "#52514e", "#3a3936"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# (key, x, y, w, h, accent, title, body lines); $...$ is typeset by KaTeX.
BOXES = [
    ("policy", 20, 150, 210, 120, INK2, "1. Action",
     [r"The design chooses $A_t$ from the past $\mathcal F_{t-1}$",
      "interim analyses, arm dropping, re-assignment"]),
    ("real", 290, 40, 250, 95, BLUE, "2a. Real system",
     [r"$X_t \sim Q_t(\cdot \mid \mathcal F_{t-1}, A_t)$"]),
    ("twin", 290, 185, 250, 120, BLUE, "2b. Simulator / digital twin",
     [r"$Y_{t,1},\dots,Y_{t,M} \sim P_t(\cdot \mid \mathcal F_{t-1}, A_t)$",
      "same action, from the current state"]),
    ("bet", 610, 95, 250, 130, ORANGE, "3&ndash;4. Score and bet",
     [r"$E_t = \dfrac{(M+1)\,e^{s_{t-1}(X_t)}}{e^{s_{t-1}(X_t)} + \sum_{j} e^{s_{t-1}(Y_{t,j})}}$"]),
    ("wealth", 920, 95, 240, 100, ORANGE, "5. Wealth",
     [r"$W_t = \frac{1}{|\Lambda|}\sum_{\lambda} \prod_{i \le t}(1-\lambda+\lambda E_i)$"]),
    ("decision", 940, 245, 200, 60, INK2, "",
     [r"$W_t \ge 1/\alpha\,?$"]),
    ("alarm", 940, 350, 200, 100, ORANGE, "Alarm",
     [r"reject $H_0$: the simulator", "is wrong under the actions taken"]),
    ("learn", 595, 290, 280, 170, GREEN, "6. Learn (after the bet)",
     [r"update $s_{t-1} \to s_t$ with $(A_t, X_t, Y_{t,1:M})$",
      "BLR, simulation likelihood, martingale predictive, forest",
      "<i>power: log score of the forecasts</i>", "<i>(Theorems 2&ndash;3)</i>"]),
]

# (polyline points, label, label x, label y, dashed)
ARROWS = [
    ([(230, 210), (260, 210), (260, 87), (290, 87)], r"$A_t$", 236, 196, False),
    ([(260, 210), (260, 245), (290, 245)], "", 0, 0, False),
    ([(540, 87), (575, 87), (575, 135), (610, 135)], r"$X_t$", 548, 70, False),
    ([(540, 245), (575, 245), (575, 185), (610, 185)], r"$Y_{t,1:M}$", 546, 252, False),
    ([(860, 150), (920, 150)], r"$E_t$", 878, 128, False),
    ([(1040, 195), (1040, 245)], "", 0, 0, False),
    ([(1040, 305), (1040, 350)], "yes", 1048, 316, False),
    ([(700, 290), (700, 225)], r"score $s_{t-1}$", 612, 246, False),
    ([(780, 225), (780, 290)], r"$(A_t, X_t, Y_{t,1:M})$", 788, 246, False),
    ([(940, 275), (900, 275), (900, 492), (125, 492), (125, 270)],
     r"no: next step, $t \leftarrow t+1$", 380, 468, True),
]


def page() -> str:
    boxes = []
    for key, x, y, w, h, accent, title, lines in BOXES:
        shape = "border-radius:30px;" if key == "decision" else "border-radius:6px;"
        head = f'<div class="title" style="color:{accent}">{title}</div>' if title else ""
        body = "".join(f'<div class="line">{ln}</div>' for ln in lines)
        boxes.append(
            f'<div class="box" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;'
            f'border-color:{accent};{shape}">{head}{body}</div>')
    paths, labels = [], []
    for pts, label, lx, ly, dashed in ARROWS:
        d = " ".join(f"{px},{py}" for px, py in pts)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        paths.append(f'<polyline points="{d}" fill="none" stroke="{LINE}" '
                     f'stroke-width="1.6"{dash} marker-end="url(#head)"/>')
        if label:
            labels.append(f'<div class="lab" style="left:{lx}px;top:{ly}px">{label}</div>')
    lane = ('<div class="lane"><div class="lanecap"><b>Validity (Theorem 1)</b>: given '
            r'$\mathcal G_t$, the outcomes $X_t, Y_{t,1}, \dots, Y_{t,M}$ are '
            'exchangeable under $H_0$</div></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script src="{KATEX}/katex.min.js"></script>
<script src="{KATEX}/contrib/auto-render.min.js"></script>
<style>
  html, body {{ margin:0; background:#ffffff; }}
  #c {{ position:relative; width:{W}px; height:{H}px;
        font-family:"Times New Roman", Times, serif; color:{INK}; }}
  .box {{ position:absolute; box-sizing:border-box; border:2px solid; background:#fff;
          padding:8px 10px; display:flex; flex-direction:column; justify-content:center;
          align-items:center; text-align:center; gap:4px; z-index:2; }}
  .title {{ font-weight:bold; font-size:17px; }}
  .line {{ font-size:15px; line-height:1.25; color:{INK}; }}
  .lab {{ position:absolute; font-size:15px; color:{INK}; z-index:3; white-space:nowrap; }}
  .lane {{ position:absolute; left:275px; top:15px; width:280px; height:375px;
           background:#eef4fc; border-radius:10px; z-index:0; }}
  .lanecap {{ position:absolute; left:12px; right:12px; bottom:12px; font-size:13.5px;
              color:{INK2}; text-align:center; line-height:1.3; }}
  svg {{ position:absolute; left:0; top:0; z-index:1; }}
  .katex {{ font-size:1.0em; }}
</style></head><body><div id="c">
{lane}
<svg width="{W}" height="{H}"><defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5"
 markerWidth="7" markerHeight="7" orient="auto-start-reverse">
 <path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>{''.join(paths)}</svg>
{''.join(boxes)}{''.join(labels)}
</div>
<script>renderMathInElement(document.body, {{delimiters:[{{left:"$", right:"$", display:false}}]}});
document.title = "ready";</script></body></html>"""


def main() -> None:
    html = ROOT / "figures" / "_workflow.html"
    html.write_text(page())
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        pg = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=3)
        pg.goto(html.as_uri())
        pg.wait_for_function("document.title === 'ready'")
        pg.wait_for_timeout(300)
        pg.locator("#c").screenshot(path=str(OUT))
        browser.close()
    html.unlink()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
