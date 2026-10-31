"""Resolve each DOI via doi.org content negotiation -> BibTeX; fail loudly on unresolvable DOIs."""
import sys, re, json, urllib.request
sys.stdout.reconfigure(encoding="utf-8")

DOIS = {
 "kratzert2018": "10.5194/hess-22-6005-2018",
 "kratzert2019ealstm": "10.5194/hess-23-5089-2019",
 "kratzert2019ungauged": "10.1029/2019WR026065",
 "kratzert2024never": "10.5194/hess-28-4187-2024",
 "frame2022extremes": "10.5194/hess-26-3377-2022",
 "nearing2024nature": "10.1038/s41586-024-07145-1",
 "nevo2022google": "10.5194/hess-26-4013-2022",
 "kratzert2023caravan": "10.1038/s41597-023-01975-w",
 "addor2017camels": "10.5194/hess-21-5293-2017",
 "newman2015": "10.5194/hess-19-209-2015",
 "coxon2020camelsgb": "10.5194/essd-12-2459-2020",
 "loritz2024camelsde": "10.5194/essd-16-5625-2024",
 "klingler2021lamah": "10.5194/essd-13-4529-2021",
 "lees2021gb": "10.5194/hess-25-5517-2021",
 "seibert2012hbv": "10.5194/hess-16-3315-2012",
 "gupta2009kge": "10.1016/j.jhydrol.2009.08.003",
 "nash1970": "10.1016/0022-1694(70)90255-6",
 "yilmaz2008fhv": "10.1029/2007WR006716",
 "hochreiter1997lstm": "10.1162/neco.1997.9.8.1735",
 "klotz2022uncertainty": "10.5194/hess-26-1673-2022",
 "kratzert2022nh": "10.21105/joss.04050",
 "cornes2018eobs": "10.1029/2017JD028200",
 "chen2016xgboost": "10.1145/2939672.2939785",
 "miersch2026boris": "10.1126/sciadv.aef7703",
 "athanase2024boris": "10.1038/s43247-024-01847-0",
 "tomczyk2025boris": "10.3390/su172310640",
 "camelspl2026": "10.5281/zenodo.20133183",
 "mosavi2018review": "10.3390/w10111536",
 "wilcoxon1945": "10.2307/3001968",
 "acuna2025hybrid": "10.5194/hess-29-1277-2025",
 "piniewski2021g2dc": "10.5194/essd-13-1273-2021",
 "bergstrom1973hbv": "10.2166/nh.1973.0012",
 "feng2022extrapolation": "10.1029/2022WR032404",
}
# No DOI exists for NeurIPS proceedings; entry checked against the official proceedings page.
MANUAL = [r"""@inproceedings{ke2017lightgbm,
 title={LightGBM: A Highly Efficient Gradient Boosting Decision Tree},
 author={Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
 booktitle={Advances in Neural Information Processing Systems},
 volume={30},
 pages={3146--3154},
 year={2017},
 url={https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree}
}"""]
ACC = {"́": "'", "̀": "`", "̈": '"', "̃": "~", "̌": "v", "̨": "k", "̧": "c",
       "̇": ".", "̂": "^", "̆": "u", "̋": "H"}
SPECIAL = {"ł": r"{\l}", "Ł": r"{\L}", "ø": r"{\o}", "Ø": r"{\O}", "ß": r"{\ss}", "–": "--", "—": "---",
           "‐": "-", "‑": "-", " ": " ", " ": "~", "’": "'", "‘": "`", "“": "``", "”": "''"}


def to_ascii(s):
    """BibTeX is 8-bit: name abbreviation splits multibyte UTF-8, so emit LaTeX macros instead."""
    import unicodedata
    res = []
    for c in s:
        if ord(c) < 128:
            res.append(c); continue
        if c in SPECIAL:
            res.append(SPECIAL[c]); continue
        d = unicodedata.normalize("NFD", c)
        if len(d) == 2 and d[1] in ACC:
            res.append("{\\%s%s}" % (ACC[d[1]], d[0]) if ACC[d[1]].isalpha() is False else "{\\%s{%s}}" % (ACC[d[1]], d[0]))
        else:
            raise ValueError("unmapped char %r" % c)
    return "".join(res)


out, bad = [], []
for key, doi in DOIS.items():
    req = urllib.request.Request("https://doi.org/" + doi, headers={"Accept": "application/x-bibtex; charset=utf-8", "User-Agent": "ref-verify/1.0"})
    try:
        bib = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        bib = re.sub(r"@(\w+)\{[^,]+,", lambda m: "@%s{%s," % (m.group(1), key), bib, count=1)
        title = re.search(r"title\s*=\s*\{(.+?)\},", bib)
        print("OK ", key, "|", title.group(1)[:90] if title else "?")
        bib = re.sub(r",\s*month\s*=\s*\w+", "", bib)  # bare month macros (sept, june) break splncs04
        import html
        bib = re.sub(r"(?<!\\)&", r"\\&", to_ascii(html.unescape(bib)))
        out.append(bib.strip())
    except Exception as e:
        print("BAD", key, doi, e); bad.append(key)
open(sys.argv[1], "w", encoding="utf-8").write("\n\n".join(out + MANUAL) + "\n")
print("bad:", bad)
