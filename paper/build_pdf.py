"""Render main.tex to PDF without a TeX distribution.

Adapted from PosteriorOperator/paper/build_pdf.py. Additions for this paper:
author-year citations (\\citep, \\citet, \\citealp) with the bibliography
generated from references.bib; theorem-like environments and proofs with
numbered cross-references; \\[ ... \\] display maths; \\item[label]; numbered
subsection references; and a lettered appendix after \\appendix.


No TeX engine can be installed in the container this was written for: the
Debian mirrors are blocked by egress policy, and the two PyPI packages named
`pdflatex` and `tex-engine` are thin wrappers that shell out to a system
binary that is not there. Chromium is present, though, and npm is reachable,
so the route taken here is LaTeX -> HTML -> Chromium print-to-PDF, with KaTeX
(vendored under paper/vendor/katex, fonts included) doing the mathematics.

This is emphatically NOT a LaTeX implementation. It handles exactly the subset
main.tex uses, and it fails loudly rather than silently mangling anything it
does not recognise -- an unknown backslash command left in the output is
reported at the end, so a command added to the paper cannot quietly vanish
from the PDF.

Usage:  python paper/build_pdf.py            # writes paper/main.pdf
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_NAME = sys.argv[1] if len(sys.argv) > 1 else "main"   # e.g. "supplement"
TEX = HERE / f"{_NAME}.tex"
OUT_HTML = HERE / f"{_NAME}.html"
OUT_PDF = HERE / f"{_NAME}.pdf"
KATEX = HERE / "vendor" / "katex"
FIGURES = HERE.parent / "figures"
BIB = HERE / "references.bib"

THM_KINDS = {"theorem": ("Theorem", True), "proposition": ("Proposition", True),
             "corollary": ("Corollary", True), "remark": ("Remark", False)}

ACCENTS = [('{\\"u}', "ü"), ('{\\"a}', "ä"), ('{\\"o}', "ö"), ("{\\'e}", "é"),
           ("{\\'a}", "á"), ("{\\`o}", "ò"), ("{\\`e}", "è"), ("{\\c{c}}", "ç"),
           ("{\\aa}", "å"), ('\\"u', "ü"), ('\\"a', "ä"), ("\\'e", "é"), ("\\'a", "á")]


def delatex(s: str) -> str:
    for a, b in ACCENTS:
        s = s.replace(a, b)
    s = s.replace("\\&", "&").replace("--", "–")
    return re.sub(r"[{}]", "", s).strip()


def parse_bib(text: str) -> dict:
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,\s]+),", text):
        kind, key = m.group(1).lower(), m.group(2)
        pos, fields = m.end(), {}
        while True:
            f = re.compile(r"\s*(\w+)\s*=\s*\{").match(text, pos)
            if not f:
                break
            val, pos = take_braced(text, f.end() - 1)
            fields[f.group(1).lower()] = val
            c = re.compile(r"\s*,").match(text, pos)
            if c:
                pos = c.end()
        fields["_kind"] = kind
        entries[key] = fields
    return entries


def surnames(authors: str) -> list[str]:
    out = []
    for a in re.split(r"\s+and\s+", authors):
        a = a.strip()
        out.append(delatex(a.split(",")[0]) if "," in a else delatex(a.split()[-1]))
    return out


def cite_name(e: dict) -> str:
    s = surnames(e.get("author", "Anon"))
    return s[0] if len(s) == 1 else f"{s[0]} and {s[1]}" if len(s) == 2 else f"{s[0]} et al."


def bib_item(e: dict) -> str:
    names = []
    for a in re.split(r"\s+and\s+", e.get("author", "")):
        if "," in a:
            last, first = a.split(",", 1)
            inits = " ".join(p[0] + "." for p in re.split(r"[\s-]+", delatex(first)) if p)
            names.append(f"{delatex(last)}, {inits}")
        else:
            names.append(delatex(a))
    auth = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    venue = e.get("journal") or e.get("booktitle") or e.get("publisher") or ""
    bits = [f"{html.escape(auth)} ({e.get('year', '')}).",
            f"{html.escape(delatex(e.get('title', '')))}."]
    tail = f"<em>{html.escape(delatex(venue))}</em>" if venue else ""
    if e.get("volume"):
        tail += f", {e['volume']}" + (f"({e['number']})" if e.get("number") else "")
    if e.get("pages"):
        tail += f", {html.escape(delatex(e['pages']))}"
    if e.get("edition"):
        tail += f", {html.escape(delatex(e['edition']))} edition"
    if tail:
        bits.append(tail + ".")
    if e.get("note") and not venue:
        note = re.sub(r"\\url\{([^}]*)\}", r"\1", e["note"])
        bits.append(html.escape(delatex(note)) + ".")
    return " ".join(bits)


MATH_TOKEN = "\x00MATH{}\x00"


# --------------------------------------------------------------------- helpers

def strip_comments(text: str) -> str:
    """Drop LaTeX comments, respecting \\% escapes."""
    out = []
    for line in text.split("\n"):
        cut = None
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def take_braced(text: str, start: int) -> tuple[str, int]:
    """Read a balanced {...} group beginning at ``start``; return body and end."""
    assert text[start] == "{", text[start : start + 20]
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    raise ValueError("unbalanced brace")


def find_env(text: str, name: str):
    """Yield (start, end, body) for each \\begin{name}...\\end{name}."""
    begin, end = f"\\begin{{{name}}}", f"\\end{{{name}}}"
    pos = 0
    while True:
        a = text.find(begin, pos)
        if a < 0:
            return
        b = text.find(end, a)
        if b < 0:
            return
        yield a, b + len(end), text[a + len(begin) : b]
        pos = b + len(end)


# ------------------------------------------------------------------ text macros

def inline_text(s: str) -> str:
    """Convert LaTeX text-mode markup (no math) to HTML."""
    s = html.escape(s, quote=False)
    # escaped characters first, so later rules cannot see them
    s = s.replace("\\%", "\x01PCT\x01").replace("\\&", "\x01AMP\x01")
    s = s.replace("\\_", "\x01USC\x01").replace("\\$", "\x01DLR\x01")
    s = s.replace("\\#", "#")
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"<strong>\1</strong>", s)
    s = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\url\{([^{}]*)\}", r"<a href=\"\1\"><code>\1</code></a>", s)
    s = re.sub(r"\\phantom\{[^{}]*\}", "&numsp;", s)
    s = s.replace("\\newblock", " ").replace("\\ldots", "…").replace("\\dots", "…")
    s = s.replace("---", "&mdash;").replace("--", "&ndash;")
    s = s.replace("``", "&ldquo;").replace("''", "&rdquo;")
    s = re.sub(r"(?<!\\)\\\n", " ", s)  # control space at a line end
    s = s.replace("~", "&nbsp;").replace("\\,", "&thinsp;").replace("\\ ", " ")
    s = s.replace("\x01PCT\x01", "%").replace("\x01AMP\x01", "&amp;")
    s = s.replace("\x01USC\x01", "_").replace("\x01DLR\x01", "$")
    return s


# --------------------------------------------------------------------- the build

class Builder:
    def __init__(self, tex: str):
        self.tex = tex
        self.math: list[tuple[str, bool]] = []      # (source, is_display)
        self.holes: list[tuple[str, bool]] = []     # (html, is_block)
        self.labels: dict[str, str] = {}
        self.cites: dict[str, str] = {}
        self.counters = {"section": 0, "subsection": 0, "equation": 0,
                         "table": 0, "figure": 0}
        self.bib: dict = {}
        self.macros: dict[str, str] = {}

    # -- math protection ---------------------------------------------------
    def stash(self, source: str, display: bool) -> str:
        self.math.append((source, display))
        return MATH_TOKEN.format(len(self.math) - 1)

    def hole(self, markup: str, block: bool = True) -> str:
        """Stash generated HTML behind a token that survives escaping."""
        self.holes.append((markup, block))
        return f"\x03H{len(self.holes) - 1}\x03"

    def protect_math(self, s: str) -> str:
        # display equations are handled by the block pass; here only inline $...$
        out, i = [], 0
        while i < len(s):
            if s[i] == "$" and (i == 0 or s[i - 1] != "\\"):
                j = i + 1
                while j < len(s) and not (s[j] == "$" and s[j - 1] != "\\"):
                    j += 1
                out.append(self.stash(s[i + 1 : j], False))
                i = j + 1
            else:
                out.append(s[i])
                i += 1
        return "".join(out)

    def text(self, s: str) -> str:
        """Protect math, convert text markup, leave math tokens in place."""
        return inline_text(self.protect_math(s))

    # -- preamble ----------------------------------------------------------
    def read_preamble(self, pre: str) -> None:
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}(?:\[(\d+)\])?", pre):
            body, _ = take_braced(pre, m.end())
            self.macros["\\" + m.group(1)] = body

    # -- label resolution (first pass) ------------------------------------
    def number_labels(self, body: str) -> None:
        """Walk the body in order, assigning numbers to every \\label."""
        events: list[tuple[int, str, str]] = []
        app = body.find("\\appendix")
        for m in re.finditer(r"\\(sub)?section\{", body):
            _, after = take_braced(body, m.end() - 1)
            nxt = re.match(r"\s*\\label\{([^}]*)\}", body[after:])
            kind = "subsection" if m.group(1) else "section"
            if kind == "section" and app >= 0 and m.start() > app:
                kind = "appsection"
            events.append((m.start(), kind, nxt.group(1) if nxt else ""))
        for kind in THM_KINDS:
            for a, b, inner in find_env(body, kind):
                lab = re.search(r"\\label\{([^}]*)\}", inner)
                events.append((a, kind, lab.group(1) if lab else ""))
        for kind in ("table", "figure"):
            for a, b, inner in find_env(body, kind):
                lab = re.search(r"\\label\{([^}]*)\}", inner)
                events.append((a, kind, lab.group(1) if lab else ""))
        for a, b, inner in find_env(body, "equation"):
            lab = re.search(r"\\label\{([^}]*)\}", inner)
            events.append((a, "equation", lab.group(1) if lab else ""))
        c = {k: 0 for k in ["section", "subsection", "appsection", "table", "figure",
                            "equation", *THM_KINDS]}
        for pos, kind, lab in sorted(events):
            if kind == "section":
                c["section"] += 1
                c["subsection"] = 0
                n = str(c["section"])
            elif kind == "appsection":
                c["appsection"] += 1
                n = chr(64 + c["appsection"])
            elif kind == "subsection":
                c["subsection"] += 1
                n = f'{c["section"]}.{c["subsection"]}'
            else:
                c[kind] += 1
                n = str(c[kind])
            if lab:
                self.labels[lab] = n

    # -- tabular -----------------------------------------------------------
    def tabular(self, spec: str, body: str) -> str:
        align = [c for c in spec if c in "lcr"]
        rows_html = []
        pending_top = False
        for raw in body.split("\\\\"):
            cls = []
            if "\\toprule" in raw:
                cls.append("toprule")
            if "\\midrule" in raw:
                cls.append("midrule")
            if "\\bottomrule" in raw:
                pending_top = True
            cell_src = re.sub(r"\\(?:top|mid|bottom)rule", "", raw)
            cell_src = re.sub(r"\\cmidrule(?:\([a-z]*\))?\{[^}]*\}", "", cell_src).strip()
            if not cell_src:
                continue
            cells = re.split(r"(?<!\\)&", cell_src)
            tds = []
            col = 0
            for c in cells:
                mc = re.fullmatch(r"\s*\\multicolumn\{(\d+)\}\{([lcr])\|?\}\{(.*)\}\s*", c, re.S)
                if mc:
                    n, al, body_ = int(mc.group(1)), mc.group(2), mc.group(3)
                    a = {"l": "left", "c": "center", "r": "right"}[al]
                    tds.append(f'<td colspan="{n}" style="text-align:{a}">'
                               f'{self.text(body_.strip())}</td>')
                    col += n
                    continue
                a = {"l": "left", "c": "center", "r": "right"}.get(
                    align[col] if col < len(align) else "l", "left"
                )
                tds.append(f'<td style="text-align:{a}">{self.text(c.strip())}</td>')
                col += 1
            rows_html.append(f'<tr class="{" ".join(cls)}">' + "".join(tds) + "</tr>")
        last = ' class="lastrow"' if pending_top else ""
        return f'<table class="booktabs"{last}><tbody>' + "".join(rows_html) + "</tbody></table>"

    # -- blocks ------------------------------------------------------------
    def block(self, body: str) -> str:
        """Convert a run of body text (no float environments) to HTML."""
        # display equations
        def eq(m):
            inner = re.sub(r"\\label\{[^}]*\}", "", m.group(1))
            self.counters["equation"] += 1
            num = self.counters["equation"]
            tok = self.stash(inner.strip(), True)
            return self.hole(f'<div class="eqwrap"><div class="eq">{tok}</div>'
                             f'<div class="eqno">({num})</div></div>')

        body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eq, body, flags=re.S)

        def disp(m):
            tok = self.stash(m.group(1).strip(), True)
            return self.hole(f'<div class="eqwrap"><div class="eq">{tok}</div></div>')

        body = re.sub(r"\\\[(.*?)\\\]", disp, body, flags=re.S)

        # lists
        def render_list(tag, inner):
            items = [x.strip() for x in inner.split("\\item") if x.strip()]
            lis, labelled = [], False
            for x in items:
                m = re.match(r"\[([^\]]*)\]\s*", x)
                if m:
                    labelled = True
                    lis.append(f'<li><span class="itemlab">{self.text(m.group(1))}</span> '
                               f"{self.text(x[m.end():])}</li>")
                else:
                    lis.append(f"<li>{self.text(x)}</li>")
            cls = ' class="labelled"' if labelled else ""
            return self.hole(f"<{tag}{cls}>{''.join(lis)}</{tag}>")

        # innermost lists first, so nesting of any depth and kind works
        innermost = re.compile(
            r"\\begin\{(itemize|enumerate)\}"
            r"((?:(?!\\begin\{(?:itemize|enumerate)\}).)*?)\\end\{\1\}", re.S)
        while True:
            m = innermost.search(body)
            if not m:
                break
            tag = "ul" if m.group(1) == "itemize" else "ol"
            body = body[:m.start()] + render_list(tag, m.group(2)) + body[m.end():]

        # centred standalone tabulars (the \begin{center} blocks)
        for a, b, inner in list(find_env(body, "center"))[::-1]:
            html_inner = inner
            for ta, tb, tbody in list(find_env(inner, "tabular"))[::-1]:
                spec, after = take_braced(
                    inner, inner.index("{", inner.index("\\begin{tabular}") + 15)
                )
                html_inner = (html_inner[:ta]
                              + self.tabular(spec, tbody[len(spec) + 2 :])
                              + html_inner[tb:])
            body = body[:a] + self.hole(f'<div class="center">{html_inner}</div>') + body[b:]

        # sections and paragraphs
        out, pos = [], 0
        app = body.find("\\appendix")
        in_app = False
        for m in re.finditer(r"\\(section|subsection|paragraph)(\*?)\{", body):
            out.append(body[pos : m.start()])
            title, after = take_braced(body, m.end() - 1)
            if m.group(2):  # unnumbered heading, e.g. \section*{Data availability}
                tag = "h2" if m.group(1) == "section" else "h3"
                out.append(self.hole(
                    f'<{tag} class="unnumbered">{self.text(title)}</{tag}>'))
            elif m.group(1) == "section":
                if app >= 0 and m.start() > app and not in_app:
                    in_app = True
                    self.counters["section"] = 0
                self.counters["section"] += 1
                self.counters["subsection"] = 0
                num = (chr(64 + self.counters["section"]) if in_app
                       else str(self.counters["section"]))
                prefix = "Appendix " if in_app else ""
                out.append(self.hole(
                    f'<h2>{prefix}{num}&nbsp;&nbsp;{self.text(title)}</h2>'))
            elif m.group(1) == "subsection":
                self.counters["subsection"] += 1
                n = f'{self.counters["section"]}.{self.counters["subsection"]}'
                out.append(self.hole(f'<h3>{n}&nbsp;&nbsp;{self.text(title)}</h3>'))
            else:
                out.append(self.hole(
                    f'<span class="parahead">{self.text(title)}</span> ', block=False))
            pos = after
        out.append(body[pos:])
        body = "".join(out).replace("\\appendix", "")

        body = re.sub(r"\\label\{[^}]*\}", "", body)

        # paragraph splitting on blank lines, leaving block-level HTML alone
        block_ids = {i for i, (_, b) in enumerate(self.holes) if b}
        rendered = []
        for chunk in re.split(r"\n\s*\n", body):
            if not chunk.strip():
                continue
            # break the chunk at block-level holes so markup never sits in a <p>
            parts = re.split(r"(\x03H\d+\x03|\x02FLOAT\d+\x02)", chunk)
            buf: list[str] = []

            def flush():
                joined = "".join(buf).strip()
                buf.clear()
                if joined:
                    rendered.append(f"<p>{self.text(joined)}</p>")

            for part in parts:
                if not part:
                    continue
                m = re.fullmatch(r"\x03H(\d+)\x03", part)
                is_block = (m and int(m.group(1)) in block_ids) or \
                           bool(re.fullmatch(r"\x02FLOAT\d+\x02", part))
                if is_block:
                    flush()
                    rendered.append(part)
                else:
                    # inline holes (run-in \paragraph heads) stay with their text
                    buf.append(part)
            flush()
        return "\n".join(rendered)

    # -- floats ------------------------------------------------------------
    def float_env(self, kind: str, inner: str) -> str:
        self.counters[kind] += 1
        num = self.counters[kind]
        cap = ""
        m = re.search(r"\\caption\{", inner)
        if m:
            cap_body, after = take_braced(inner, m.end() - 1)
            cap = cap_body
            inner = inner[: m.start()] + inner[after:]
        label = "Table" if kind == "table" else "Figure"

        if kind == "figure":
            g = re.search(r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}", inner)
            src = (FIGURES / g.group(2)).resolve() if g else None
            width = re.search(r"width=([\d.]*)\\textwidth", g.group(1)) if g else None
            pct = float(width.group(1)) * 100 if width and width.group(1) else 100.0
            body_html = (f'<img src="file://{src}" style="width:{pct:.0f}%">'
                         if src and src.exists() else
                         '<div class="missing">missing figure</div>')
        else:
            body_html = ""
            for ta, tb, tbody in find_env(inner, "tabular"):
                head = inner.index("\\begin{tabular}")
                spec, _ = take_braced(inner, inner.index("{", head + 15))
                body_html = self.tabular(spec, tbody[len(spec) + 2 :])
                break

        cap_html = (f'<figcaption><span class="capnum">{label} {num}:</span> '
                    f'{self.text(cap)}</figcaption>')
        order = (cap_html + body_html) if kind == "table" else (body_html + cap_html)
        return f'<figure class="{kind}">{order}</figure>'

    # -- driver ------------------------------------------------------------
    def run(self) -> str:
        tex = strip_comments(self.tex)
        pre, body = tex.split("\\begin{document}", 1)
        body = body.split("\\end{document}")[0]
        self.read_preamble(pre)

        title = take_braced(pre, pre.index("{", pre.index("\\title")))[0]
        author = take_braced(pre, pre.index("{", pre.index("\\author")))[0]
        date = take_braced(pre, pre.index("{", pre.index("\\date")))[0]

        # bibliography from references.bib (author-year)
        self.bib = parse_bib(BIB.read_text()) if BIB.exists() else {}
        cited = []
        for m in re.finditer(r"\\cite(?:p|t|alp)?(?:\[[^\]]*\])*\{([^}]*)\}", body):
            for k in m.group(1).split(","):
                if k.strip() not in cited:
                    cited.append(k.strip())
        self.missing_cites = [k for k in cited if k not in self.bib]
        body = re.sub(r"\\bibliographystyle\{[^}]*\}", "", body)
        bib_html = ""
        if "\\bibliography{" in body:
            body = re.sub(r"\\bibliography\{[^}]*\}", "", body)
            keys = sorted((k for k in cited if k in self.bib),
                          key=lambda k: (surnames(self.bib[k].get("author", ""))[0].lower(),
                                         self.bib[k].get("year", "")))
            items = "".join(f"<li>{bib_item(self.bib[k])}</li>" for k in keys)
            bib_html = f'<h2 class="unnumbered">References</h2><ul class="bib">{items}</ul>'
        # legacy numeric bibliography, if present
        for a, b, inner in find_env(body, "thebibliography"):
            inner = re.sub(r"^\{[^}]*\}", "", inner.strip())
            entries = [x.strip() for x in inner.split("\\bibitem") if x.strip()]
            items = []
            for i, e in enumerate(entries, 1):
                key, after = take_braced(e, e.index("{"))
                rest = e[after:]
                self.cites[key] = str(i)
                items.append(f'<li><span class="bibnum">[{i}]</span> {self.text(rest)}</li>')
            bib_html = ('<h2 class="unnumbered">References</h2>'
                        f'<ol class="bib">{"".join(items)}</ol>')
            body = body[:a] + body[b:]
            break

        # abstract
        abstract_html = ""
        for a, b, inner in find_env(body, "abstract"):
            abstract_html = ('<div class="abstract"><div class="abstracthead">Abstract</div>'
                             + self.block(inner) + "</div>")
            body = body[:a] + body[b:]
            break

        body = body.replace("\\maketitle", "")
        self.number_labels(body)

        # pull floats out, convert, reinsert as placeholders
        floats: list[str] = []
        for kind in ("table", "figure"):
            while True:
                found = next(find_env(body, kind), None)
                if not found:
                    break
                a, b, inner = found
                floats.append(self.float_env(kind, inner))
                body = body[:a] + f"\x02FLOAT{len(floats)-1}\x02" + body[b:]

        # theorem-like environments and proofs become block holes
        for kind, (label, italic) in THM_KINDS.items():
            n = 0
            while True:
                found = next(find_env(body, kind), None)
                if not found:
                    break
                a, b, inner = found
                n += 1
                inner = inner.strip()
                thm_title = ""
                tm = re.match(r"\[([^\]]*)\]", inner)
                if tm:
                    thm_title = f" ({self.text(tm.group(1))})"
                    inner = inner[tm.end():]
                content = self.block(inner)
                head = f'<span class="thmhead">{label} {n}{thm_title}.</span> '
                content = (content.replace("<p>", "<p>" + head, 1) if content.startswith("<p>")
                           else f"<p>{head}</p>" + content)
                cls = "thm italic" if italic else "thm"
                body = body[:a] + self.hole(f'<div class="{cls}">{content}</div>') + body[b:]
        while True:
            found = next(find_env(body, "proof"), None)
            if not found:
                break
            a, b, inner = found
            content = self.block(inner.strip())
            head = '<span class="proofhead">Proof.</span> '
            content = (content.replace("<p>", "<p>" + head, 1) if content.startswith("<p>")
                       else f"<p>{head}</p>" + content)
            content += '<div class="qed">∎</div>'
            body = body[:a] + self.hole(f'<div class="proof">{content}</div>') + body[b:]

        html_body = self.block(body)

        for i, f in enumerate(floats):
            html_body = html_body.replace(f"\x02FLOAT{i}\x02", f)
        def sub(m):
            return self.holes[int(m.group(1))][0]

        for _ in range(4):  # holes may nest (a list inside a section body)
            html_body = re.sub(r"\x03H(\d+)\x03", sub, html_body)
            abstract_html = re.sub(r"\x03H(\d+)\x03", sub, abstract_html)
        html_body = re.sub(r"<p>\s*(<figure)", r"\1", html_body)
        html_body = re.sub(r"(</figure>)\s*</p>", r"\1", html_body)

        # cross-references
        def ref(m):
            return self.labels.get(m.group(1), "??")
        html_body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({ref(m)})", html_body)
        abstract_html = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({ref(m)})", abstract_html)
        html_body = re.sub(r"\\ref\{([^}]*)\}", ref, html_body)
        abstract_html = re.sub(r"\\ref\{([^}]*)\}", ref, abstract_html)
        for key, n in self.cites.items():
            html_body = html_body.replace(f"\\cite{{{key}}}", f"[{n}]")
            abstract_html = abstract_html.replace(f"\\cite{{{key}}}", f"[{n}]")

        def cite(m):
            kind, o1, o2 = m.group(1), m.group(2), m.group(3)
            # natbib: one optional arg is a postnote; two are prenote, postnote
            pre, post = (o1, o2) if o2 is not None else ("", o1 or "")
            pre = (pre + " ") if pre else ""
            post = (", " + post) if post else ""
            keys = [k.strip() for k in m.group(4).split(",")]
            ents = [(self.bib[k] if k in self.bib else {"author": "??", "year": "??"}) for k in keys]
            if kind == "t":
                parts = [f"{cite_name(e)} ({e.get('year')}" for e in ents]
                return "; ".join(p_ + ")" for p_ in parts[:-1]) + \
                    ("; " if len(parts) > 1 else "") + parts[-1] + post + ")"
            inner = pre + "; ".join(f"{cite_name(e)}, {e.get('year')}" for e in ents) + post
            return f"({inner})" if kind == "p" else inner
        pat = r"\\cite(p|t|alp)(?:\[([^\]]*)\])?(?:\[([^\]]*)\])?\{([^}]*)\}"
        html_body = re.sub(pat, cite, html_body)
        abstract_html = re.sub(pat, cite, abstract_html)

        page = TEMPLATE.format(
            title=inline_text(title.replace("\\\\", " ")),
            author=(inline_text(author.replace("\\\\", "\x04BR\x04")
                                      .replace("\\and", "\x04SEP\x04"))
                    .replace("\x04BR\x04", "<br>")
                    .replace("\x04SEP\x04", "&nbsp;&nbsp;&middot;&nbsp;&nbsp;")),
            date=inline_text(date.replace("\\today", "September 2026")),
            abstract=abstract_html,
            body=html_body,
            bib=bib_html,
            katex=KATEX.resolve().as_uri(),
            macros=self.katex_macros(),
        )

        # restore math last, so nothing above can corrupt TeX source
        def put(m):
            src, disp = self.math[int(m.group(1))]
            esc = html.escape(src, quote=True)
            cls = "math-display" if disp else "math-inline"
            return f'<span class="{cls}" data-tex="{esc}"></span>'

        page = re.sub(r"\x00MATH(\d+)\x00", put, page)
        self.report_unknown(page)
        if getattr(self, "missing_cites", None):
            print("  WARNING citations missing from references.bib:", self.missing_cites)
        unresolved = sorted(set(re.findall(r"\\(?:eq)?ref\{([^}]*)\}", self.tex)) - set(self.labels))
        if unresolved:
            print("  WARNING unresolved references:", unresolved)
        return page

    def katex_macros(self) -> str:
        def esc(t: str) -> str:
            return t.replace(chr(92), chr(92) * 2)

        pairs = ",".join(f'"{esc(k)}": "{esc(v)}"' for k, v in self.macros.items())
        return "{" + pairs + "}"

    def report_unknown(self, page: str) -> None:
        """Fail loudly on any LaTeX command that survived into the HTML."""
        scrubbed = re.sub(r'data-tex="[^"]*"', "", page)
        scrubbed = re.sub(r"var macros = \{.*?\};", "", scrubbed, flags=re.S)
        leftovers = sorted(set(re.findall(r"\\[A-Za-z]+", scrubbed)))
        ignore = {"\\n", "\\t"}
        leftovers = [x for x in leftovers if x not in ignore]
        if leftovers:
            print("  WARNING unconverted commands in output:", ", ".join(leftovers))
        else:
            print("  no unconverted LaTeX commands in the output")


TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="{katex}/katex.min.css">
<script defer src="{katex}/katex.min.js"></script>
<style>
  @page {{ size: A4; margin: 22mm 20mm; }}
  body {{ font-family: "Times New Roman", Times, serif; font-size: 10.5pt;
         line-height: 1.42; color: #111; margin: 0; }}
  h1 {{ font-size: 17pt; text-align: center; margin: 0 0 4pt; line-height: 1.25; }}
  .authors {{ text-align: center; font-size: 11pt; margin-bottom: 2pt; }}
  .date {{ text-align: center; font-size: 10pt; color: #444; margin-bottom: 16pt; }}
  h2 {{ font-size: 12pt; margin: 16pt 0 5pt; }}
  h2.unnumbered {{ margin-top: 18pt; }}
  h3 {{ font-size: 10.8pt; margin: 12pt 0 4pt; font-style: italic; }}
  p {{ margin: 0 0 6pt; text-align: justify; hyphens: auto; }}
  .parahead {{ font-weight: bold; }}
  .abstract {{ margin: 0 14mm 14pt; font-size: 9.8pt; }}
  .abstracthead {{ text-align: center; font-weight: bold; margin-bottom: 4pt; }}
  ul, ol {{ margin: 0 0 6pt; padding-left: 18pt; }}
  li {{ margin-bottom: 3pt; text-align: justify; }}
  figure {{ margin: 12pt 0; text-align: center; break-inside: avoid; }}
  figure img {{ max-width: 100%; height: auto; }}
  figcaption {{ font-size: 9pt; text-align: justify; margin: 6pt 4pt 0;
                line-height: 1.32; }}
  figure.table figcaption {{ margin: 0 4pt 6pt; }}
  .capnum {{ font-weight: bold; }}
  table.booktabs {{ border-collapse: collapse; margin: 0 auto; font-size: 9.3pt; }}
  table.booktabs td {{ padding: 2.2pt 7pt; }}
  tr.toprule td {{ border-top: 1.1pt solid #111; }}
  tr.midrule td {{ border-top: 0.5pt solid #111; }}
  table.lastrow tr:last-child td {{ border-bottom: 1.1pt solid #111; }}
  .eqwrap {{ display: flex; align-items: center; margin: 9pt 0; }}
  .eq {{ flex: 1; text-align: center; }}
  .eqno {{ width: 3em; text-align: right; }}
  .center {{ text-align: center; margin: 9pt 0; break-inside: avoid; }}
  .bib {{ list-style: none; padding-left: 0; font-size: 9.6pt; }}
  .bib li {{ padding-left: 2.2em; text-indent: -2.2em; }}
  .bibnum {{ font-weight: normal; }}
  code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; }}
  .missing {{ color: #b00; font-style: italic; }}
  .thm {{ margin: 8pt 0; }}
  .thm.italic p, .thm.italic li {{ font-style: italic; }}
  .thmhead {{ font-weight: bold; font-style: normal; }}
  .proof {{ margin: 4pt 0 10pt; }}
  .proofhead {{ font-style: italic; }}
  .qed {{ text-align: right; margin-top: -14pt; }}
  ol.labelled, ul.labelled {{ list-style: none; padding-left: 10pt; }}
  ol.labelled li, ul.labelled li {{ text-align: left; }}
  .itemlab {{ font-style: normal; }}
  .bib li {{ margin-bottom: 3pt; }}
</style></head><body>
<h1>{title}</h1>
<div class="authors">{author}</div>
<div class="date">{date}</div>
{abstract}
{body}
{bib}
<script>
window.addEventListener("load", function () {{
  var macros = {macros};
  document.querySelectorAll(".math-inline,.math-display").forEach(function (el) {{
    try {{
      katex.render(el.getAttribute("data-tex"), el, {{
        displayMode: el.classList.contains("math-display"),
        throwOnError: false, macros: macros, strict: false
      }});
    }} catch (e) {{ el.textContent = el.getAttribute("data-tex"); }}
  }});
  document.title = "ready";
}});
</script>
</body></html>
"""


def ensure_katex() -> None:
    """Fetch KaTeX from the npm registry if it is not vendored yet."""
    if (KATEX / "katex.min.css").exists():
        return
    import io
    import tarfile
    import urllib.request

    url = "https://registry.npmjs.org/katex/-/katex-0.16.11.tgz"
    print(f"  fetching KaTeX from {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        blob = r.read()
    KATEX.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("package/dist/")]
        for m in members:
            m.name = m.name[len("package/dist/") :]
            if m.name:
                tar.extract(m, KATEX)
    print(f"  vendored KaTeX into {KATEX.relative_to(HERE.parent)}")


def main() -> int:
    ensure_katex()
    print(f"reading {TEX.relative_to(HERE.parent)}")
    builder = Builder(TEX.read_text())
    page = builder.run()
    OUT_HTML.write_text(page)
    print(f"  labels resolved: {len(builder.labels)}   math fragments: {len(builder.math)}")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        page_obj = browser.new_page()
        errors: list[str] = []
        page_obj.on("pageerror", lambda e: errors.append(str(e)))
        page_obj.goto(OUT_HTML.resolve().as_uri(), wait_until="load")
        page_obj.wait_for_function("document.title === 'ready'", timeout=30000)
        page_obj.pdf(path=str(OUT_PDF), format="A4", print_background=True)
        browser.close()
        if errors:
            print("  page errors:", errors[:3])

    size = OUT_PDF.stat().st_size
    print(f"wrote {OUT_PDF.relative_to(HERE.parent)}  ({size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
