"""Build the arXiv upload from main.tex and supplement.tex.

The zip contains main.tex, its bibliography already processed (main.bbl, since
arXiv does not run BibTeX), the figures it uses in figures/, and the compiled
supplement as an ancillary file (anc/supplement.pdf), which arXiv lists on the
abstract page. In the copy of main.tex, whole-line comments are removed (arXiv
publishes the source), figures are read from figures/ and \\pdfoutput=1 is set
so that arXiv runs pdflatex. The copy is compiled once more from the zip's
contents alone, without BibTeX, to check that it builds as arXiv will build it.

Run from this folder:

    python make_arxiv.py        # -> arxiv/ and arxiv.zip
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figures"
OUT = HERE / "arxiv"


def latex(name: str, cwd: Path, bibtex: bool) -> str:
    def run(*cmd):
        subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run("pdflatex", "-interaction=nonstopmode", name + ".tex")
    if bibtex:
        run("bibtex", name)
    for _ in range(2):
        run("pdflatex", "-interaction=nonstopmode", name + ".tex")
    return (cwd / (name + ".log")).read_text(errors="replace")


def check(log: str, what: str) -> None:
    errors = [l for l in log.splitlines() if l.startswith("!")]
    undefined = [l for l in log.splitlines() if "undefined" in l]
    if errors or undefined:
        raise SystemExit(f"{what}: {errors[:3]} {undefined[:3]}")


def main() -> None:
    # bibliography and supplement, compiled in place
    check(latex("main", HERE, bibtex=True), "main.tex")
    check(latex("supplement", HERE, bibtex=True), "supplement.tex")

    tex = (HERE / "main.tex").read_text()
    tex = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%")) + "\n"
    tex = tex.replace("\\graphicspath{{../figures/}}", "\\graphicspath{{figures/}}")
    tex = "\\pdfoutput=1\n" + tex

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "anc").mkdir()
    (OUT / "main.tex").write_text(tex)
    shutil.copy(HERE / "main.bbl", OUT / "main.bbl")
    for name in sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", tex))):
        shutil.copy(FIGS / name, OUT / "figures" / name)
    shutil.copy(HERE / "supplement.pdf", OUT / "anc" / "supplement.pdf")
    files = sorted(p.relative_to(OUT) for p in OUT.rglob("*") if p.is_file())

    # build from the zip's contents alone, as arXiv does
    test = HERE / "arxiv_test"
    shutil.rmtree(test, ignore_errors=True)
    shutil.copytree(OUT, test)
    check(latex("main", test, bibtex=False), "arXiv build")
    pages = subprocess.run(["pdfinfo", "main.pdf"], cwd=test, capture_output=True, text=True).stdout
    shutil.rmtree(test)

    with zipfile.ZipFile(HERE / "arxiv.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(OUT / f, f.as_posix())
    print("wrote", HERE / "arxiv.zip", [f.as_posix() for f in files])
    print([l for l in pages.splitlines() if l.startswith("Pages")])
    if "To be completed" in tex:
        print("WARNING: main.tex still contains '[To be completed ...]'")


if __name__ == "__main__":
    main()
