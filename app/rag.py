"""Advanced RAG: ingestion (md/html/pdf/csv) -> chunks -> hybrid retrieval (BM25 + vector) -> RRF fusion -> rerank
-> confidence gate ("I don't know") -> citations.  Pure python + numpy, no external services."""
import csv
import hashlib
import html as htmlmod
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

STOP = set("""a an the and or of to in on for with is are was were be been it its this that these those my me i we you your our
do does did can could would should will shall may might have has had at by from as if so but not no yes any all about what when
where which who how why there their they them then than into out up over again please hi hello thanks thank hey long much many something anything come comes get got tell know need want like just also northwick co pound pounds gbp work works give make take show let us here there policy policies company send back certain amount""".split())
DIM = 4096


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokens(text, keep_stop=False):
    ws = re.findall(r"[a-z0-9£]+(?:[.'-][a-z0-9]+)*", text.lower())
    return [stem(w) for w in ws if keep_stop or w not in STOP]


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    text: str
    kind: str
    toks: list = field(default_factory=list)


def _read(path: Path):
    ext = path.suffix.lower()
    if ext == ".md":
        txt = path.read_text(encoding="utf-8")
        return [(txt.splitlines()[0].lstrip("# ").strip(), txt)], "md"
    if ext == ".html":
        raw = path.read_text(encoding="utf-8")
        title = re.search(r"<title>(.*?)</title>", raw, re.S)
        raw = re.sub(r"<h[12][^>]*>", "\n## ", raw)
        raw = re.sub(r"<[^>]+>", " ", raw)
        return [(htmlmod.unescape(title.group(1)) if title else path.stem, htmlmod.unescape(raw))], "web"
    if ext == ".pdf":
        from pypdf import PdfReader
        txt = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        return [(txt.strip().splitlines()[0] if txt.strip() else path.stem, txt)], "pdf"
    if ext == ".csv":
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        out = []
        for r in rows:
            stock = "in stock" if int(r["stock"]) > 0 else "out of stock"
            out.append((r["name"], f"Product {r['name']} (SKU {r['sku']}), category {r['category']}, price GBP {r['price_gbp']}, "
                        f"{stock}, {r['description']}. Material: {r['material']}. Guarantee: {r['guarantee_months']} months."))
        return out, "csv"
    return [], ""


def split_text(title, text, size=600):
    """Section-aware chunking: split on markdown headings, then pack paragraphs up to `size` chars."""
    parts = re.split(r"\n(?=#{1,3} )", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= size:
            out.append(p)
            continue
        cur = ""
        for para in re.split(r"\n+", p):
            if len(cur) + len(para) > size and cur:
                out.append(cur)
                cur = para
            else:
                cur = (cur + "\n" + para).strip()
        if cur:
            out.append(cur)
    return out


def _hash(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")


def embed(text):
    """Hashing-trick embedding over unigrams, bigrams and char-trigrams (robust to typos/morphology). Offline, deterministic."""
    v = np.zeros(DIM, dtype=np.float32)
    ts = tokens(text)
    feats = [("u", t, 1.0) for t in ts] + [("b", a + "_" + b, 1.3) for a, b in zip(ts, ts[1:])]
    for t in ts:
        p = f"^{t}$"
        feats += [("c", p[i:i + 3], 0.35) for i in range(len(p) - 2)]
    for kind, f, w in feats:
        h = _hash(kind + f)
        v[h % DIM] += w * (1 if (h >> 40) & 1 else -1)
    n = np.linalg.norm(v)
    return v / n if n else v


class Index:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.mat = np.zeros((0, DIM), dtype=np.float32)
        self.df = Counter()
        self.avgdl = 1.0

    # ---------- build ----------
    def build(self, docs_dir=config.DOCS_DIR):
        self.chunks = []
        for path in sorted(Path(docs_dir).rglob("*")):
            if not path.is_file():
                continue
            secs, kind = _read(path)
            n = 0
            for title, text in secs:
                for piece in ([text] if kind == "csv" else split_text(title, text)):
                    n += 1
                    piece = piece if (kind == "csv" or piece.lstrip("# ").startswith(title)) else "# " + title + "\n" + piece
                    c = Chunk(id=f"{path.stem}#{n}", doc=path.stem, title=title, text=piece, kind=kind)
                    c.toks = tokens(title + " " + piece)
                    self.chunks.append(c)
        self.df = Counter(t for c in self.chunks for t in set(c.toks))
        self.avgdl = sum(len(c.toks) for c in self.chunks) / max(1, len(self.chunks))
        self.mat = np.vstack([embed(c.title + " " + c.text) for c in self.chunks]) if self.chunks else self.mat
        return self

    @property
    def n_docs(self):
        return len({c.doc for c in self.chunks})

    # ---------- retrieval ----------
    def bm25(self, q, k=1.5, b=0.75):
        qt = tokens(q)
        N = len(self.chunks)
        scores = np.zeros(N)
        for i, c in enumerate(self.chunks):
            tf = Counter(c.toks)
            s = 0.0
            for t in set(qt):
                if t in tf:
                    idf = math.log(1 + (N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (k + 1) / (tf[t] + k * (1 - b + b * len(c.toks) / self.avgdl))
            scores[i] = s
        return scores

    def vector(self, q):
        return self.mat @ embed(q)

    @staticmethod
    def normalize(q):
        """Synonym normalisation for common customer phrasings."""
        q = re.sub(r"\b(send|give|post|take)\s+(?:\w+\s+){0,2}?back\b", "return", q, flags=re.I)
        q = re.sub(r"\bmoney back\b|\bget my money\b", "refund", q, flags=re.I)
        return q

    @staticmethod
    def expand(q):
        ql = q.lower()
        extra = []
        if re.search(r"how long|how many days|how soon|when will", ql):
            extra.append("days")
        if re.search(r"how much|cost|price|charge|fee", ql):
            extra.append("gbp cost price free")
        if re.search(r"how many|points", ql):
            extra.append("per every")
        return q + " " + " ".join(extra) if extra else q

    def search(self, q, top_k=4, pool=12):
        if not self.chunks:
            return []
        q = self.normalize(q)
        q0, q = q, self.expand(q)
        bm, vec = self.bm25(q), self.vector(q)
        rrf = np.zeros(len(self.chunks))
        for scores in (bm, vec):
            order = np.argsort(-scores)[:pool]
            for rank, i in enumerate(order):
                if scores[i] > 0:
                    rrf[i] += 1.0 / (60 + rank)
        cand = [i for i in np.argsort(-rrf)[:pool] if rrf[i] > 0]
        ranked = sorted(((self.rerank(q0, self.chunks[i], rrf[i], bm[i], vec[i]), i) for i in cand), reverse=True)
        return [{"chunk": self.chunks[i], "score": round(float(s), 3), "bm25": round(float(bm[i]), 2),
                 "vec": round(float(vec[i]), 3)} for s, i in ranked[:top_k]]

    def rerank(self, q, c, rrf, bm, vec):
        """Feature reranker: query-term coverage (idf weighted), bigram phrase hits, title match, vector sim, RRF prior.
        Coverage is the main answerability signal used by the 'I don't know' gate."""
        qt = [t for t in dict.fromkeys(tokens(q))]
        if not qt:
            return 0.0
        N = len(self.chunks)
        idf = {t: math.log(1 + (N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5)) for t in qt}
        ctoks = set(c.toks)
        known = [t for t in qt if self.df.get(t, 0) > 0]
        cover_w = sum(idf[t] for t in qt if t in ctoks)
        total_w = sum(idf[t] for t in qt) or 1
        coverage = cover_w / total_w
        ordered = tokens(q)
        bigr = list(zip(ordered, ordered[1:]))
        ctext = " ".join(c.toks)
        phrase = sum(1 for a, b in bigr if f"{a} {b}" in ctext) / max(1, len(bigr))
        title_hit = len(set(qt) & set(tokens(c.title))) / len(qt)
        return 0.45 * coverage + 0.10 * phrase + 0.10 * title_hit + 0.15 * max(0.0, float(vec)) + 0.20 * min(1.0, float(bm) / (2.5 * total_w)) + (0.06 if c.id.endswith('#1') and c.kind != 'csv' else 0.0) + (0.0 if known else -1)


ANSWERABLE_THRESHOLD = 0.31
_index = None


def get_index():
    global _index
    if _index is None:
        _index = Index().build()
    return _index


def retrieve(query, top_k=4):
    """Returns (hits, answerable). Answerable only if the best reranked hit passes the confidence gate."""
    hits = get_index().search(query, top_k=top_k)
    ok = bool(hits) and hits[0]["score"] >= ANSWERABLE_THRESHOLD
    return hits, ok


def citation(hit):
    c = hit["chunk"]
    return {"id": c.id, "doc": c.doc, "title": c.title, "kind": c.kind, "score": hit["score"], "snippet": c.text[:220]}


NUMERIC_Q = re.compile(r"how (long|much|many|soon)|how many days|cost|price", re.I)


def best_sentences(query, hits, n=2):
    """Extractive answer composer (offline mode): pick the sentences with highest query-term overlap."""
    qt = set(tokens(query))
    cands = []
    for hi, h in enumerate(hits[:3]):
        text = re.sub(r"[ \t]+", " ", re.sub(r"[#*|]", " ", h["chunk"].text))
        for s in re.split(r"(?<=[.!?])\s+|\n+", text):
            s = s.strip(" -\t")
            if len(s) < 25:
                continue
            st = set(tokens(s))
            ov = len(qt & st) / (len(qt) or 1)
            if ov > 0 and re.search(r"\d", s) and NUMERIC_Q.search(query):
                ov += 0.25
            cands.append((ov - 0.12 * hi - 0.02 * len(cands), hi, s))
    cands.sort(reverse=True)
    picked = []
    for sc, hi, s in cands[:n]:
        if sc <= 0 or (picked and sc < 0.6 * cands[0][0]):
            continue
        picked.append((hi + 1, s))
    return picked
