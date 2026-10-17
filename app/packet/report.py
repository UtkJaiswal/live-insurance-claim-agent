"""Human-readable report, rendered from claim_packet.json only (plus evidence crops on disk)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from jinja2 import Environment, select_autoescape

from app.schema import ClaimPacket


def save_crops(packet: ClaimPacket, sweep_dir: Path, max_side: int = 360) -> dict[str, str]:
    """Writes crops/<line id>.jpg from each line's frame_ref box. Returns {id: relative path}."""
    out_dir = sweep_dir / "crops"
    out_dir.mkdir(exist_ok=True)
    out: dict[str, str] = {}
    cache: dict[str, Any] = {}
    for line in [*packet.books, *packet.items]:
        path, _, frag = line.frame_ref.partition("#")
        if path not in cache:
            cache[path] = cv2.imread(str(sweep_dir / path))
        img = cache[path]
        if img is None:
            continue
        if frag.startswith("xywh="):
            x, y, w, h = (int(v) for v in frag[5:].split(","))
            img = img[max(0, y) : y + h, max(0, x) : x + w]
        if img.size == 0:
            continue
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        rel = f"crops/{line.id}.jpg"
        cv2.imwrite(str(sweep_dir / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out[line.id] = rel
    return out


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claim Packet {{ p.sweep.id }}</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1f23;--muted:#6b7280;--line:#e5e7eb;--card:#fff;--accent:#2457d6;--warn:#b45309;--bad:#b91c1c;--ok:#15803d}
@media (prefers-color-scheme: dark){:root{--bg:#121417;--fg:#e8eaed;--muted:#9aa1ac;--line:#2a2e35;--card:#191c20;--accent:#7aa2ff;--warn:#f0b35a;--bad:#ff7b72;--ok:#56d364}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}.big{font-size:20px;font-weight:600}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--card);font-weight:600;white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
img.crop{max-height:90px;max-width:70px;border-radius:3px;display:block}a{color:var(--accent)}
.tag{display:inline-block;padding:1px 6px;border-radius:99px;font-size:11px;border:1px solid var(--line)}
.identified{color:var(--ok)}.unidentified{color:var(--muted)}.needs_appraisal{color:var(--warn)}.priced{color:var(--ok)}.range{color:var(--accent)}
.small{font-size:12px}.defect{color:var(--bad)}
</style></head><body><main>
<h1>Contents claim packet</h1>
<div class="muted">Sweep {{ p.sweep.id }} · captured {{ p.sweep.captured_at }} · {{ p.sweep.country }} / {{ p.sweep.currency }} · sweep {{ p.sweep.duration_s }} s</div>
{% if defects %}<p class="defect">Integrity check found {{ defects|length }} defect(s): {{ defects|join('; ') }}</p>{% else %}<p class="muted small">Integrity check passed: every price has a source, URL and date; every frame reference opens; totals equal the sum of the lines.</p>{% endif %}

<h2>Totals</h2>
<div class="grid">
<div class="card"><div class="muted small">Books counted</div><div class="big">{{ t.book_count }}</div><div class="small">{{ t.books_identified }} identified · {{ t.books_unidentified }} unidentified</div></div>
<div class="card"><div class="muted small">Books, replacement</div><div class="big">{{ money(t.books_replacement_cost) }}</div><div class="small">used value {{ money(t.books_used_value) }}</div></div>
<div class="card"><div class="muted small">Other items, replacement</div><div class="big">{{ money(t.items_replacement_cost_low) }} – {{ money(t.items_replacement_cost_high) }}</div></div>
<div class="card"><div class="muted small">Shelf run</div><div class="big">{{ t.shelf_run_m }} m</div></div>
<div class="card"><div class="muted small">Excluded from totals</div><div class="big">{{ t.excluded_from_totals }}</div><div class="small">unpriced, unidentified or for appraisal</div></div>
<div class="card"><div class="muted small">Review queue</div><div class="big">{{ p.review_queue|length }}</div></div>
</div>

<h2>Room</h2>
<div class="grid">
<div class="card"><div class="muted small">Dimensions (L × W × H)</div><div class="big">{{ dim(r.length_m) }} × {{ dim(r.width_m) }} × {{ dim(r.height_m) }} m</div></div>
<div class="card"><div class="muted small">Floor area</div><div class="big">{{ dim(r.floor_area_m2) }} m²</div><div class="small">{{ dim(r.floor_area_ft2) }} ft²</div></div>
<div class="card"><div class="muted small">Wall area</div><div class="big">{{ dim(r.wall_area_m2) }} m²</div><div class="small">{{ dim(r.wall_area_ft2) }} ft²</div></div>
<div class="card"><div class="muted small">Wall covered by shelving</div><div class="big">{{ dim(r.shelved_wall_area_m2) }} m²</div></div>
</div>
<p class="small muted">Scale: {{ r.scale_method or "none" }} · confidence {{ r.confidence }}{% if r.shape_notes %} · {{ r.shape_notes }}{% endif %}</p>

<h2>Books ({{ p.books|length }})</h2>
<div class="tablewrap"><table>
<tr><th></th><th>ID</th><th>Shelf · pos</th><th>Status</th><th>Title / author / edition</th><th>Spine (cm)</th><th>Replacement</th><th>Used</th><th>Conf.</th></tr>
{% for b in p.books %}<tr>
<td>{% if crops.get(b.id) %}<a href="{{ frame(b.frame_ref) }}"><img class="crop" src="{{ crops[b.id] }}" alt=""></a>{% endif %}</td>
<td class="small">{{ b.id }}</td><td class="small">{{ b.shelf or "" }} · {{ b.position or "" }}</td>
<td><span class="tag {{ b.status }}">{{ b.status.replace('_',' ') }}</span></td>
<td>{% if b.title %}<b>{{ b.title }}</b>{% if b.author %} — {{ b.author }}{% endif %}{% if b.edition %}<div class="small muted">{{ b.edition }}{% if b.isbn %} · ISBN {{ b.isbn }}{% endif %}</div>{% endif %}{% if b.catalogue_url %}<div class="small"><a href="{{ b.catalogue_url }}">catalogue</a></div>{% endif %}{% else %}<span class="muted">—</span>{% endif %}
{% if b.spine_text_read %}<div class="small muted">spine: “{{ b.spine_text_read }}”</div>{% endif %}{% if b.appraisal_reason %}<div class="small needs_appraisal">{{ b.appraisal_reason }}</div>{% endif %}</td>
<td class="num">{{ dim(b.spine_height_cm) }} × {{ dim(b.spine_thickness_cm) }}</td>
<td class="num">{% if b.replacement_cost.amount is not none %}<a href="{{ b.replacement_cost.url }}">{{ money(b.replacement_cost.amount) }}</a>{% if b.replacement_cost.converted %}<div class="small muted">converted from {{ b.replacement_cost.original_currency }} {{ b.replacement_cost.original_amount }}</div>{% endif %}<div class="small muted">{{ b.replacement_cost.source }} · {{ b.replacement_cost.retrieved_at[:10] }}</div>{% else %}—{% endif %}</td>
<td class="num">{% if b.used_value.amount is not none %}<a href="{{ b.used_value.url }}">{{ money(b.used_value.amount) }}</a><div class="small muted">{{ b.used_value.condition_assumed }} · {{ b.used_value.retrieved_at[:10] }}</div>{% else %}—{% endif %}</td>
<td class="num">{{ "%.2f"|format(b.id_confidence) }}</td></tr>{% endfor %}
</table></div>

<h2>Other items ({{ p.items|length }})</h2>
<div class="tablewrap"><table>
<tr><th></th><th>ID</th><th>Category</th><th>Description</th><th>Size w × h (cm)</th><th>Status</th><th>Replacement</th><th>Source</th></tr>
{% for it in p.items %}<tr>
<td>{% if crops.get(it.id) %}<a href="{{ frame(it.frame_ref) }}"><img class="crop" src="{{ crops[it.id] }}" alt=""></a>{% endif %}</td>
<td class="small">{{ it.id }}</td><td>{{ it.category }}</td><td>{{ it.description }}{% if it.brand_model %} · {{ it.brand_model }}{% endif %}</td>
<td class="num">{{ dim(it.dimensions_cm.w) }} × {{ dim(it.dimensions_cm.h) }}</td>
<td><span class="tag {{ it.status }}">{{ it.status.replace('_',' ') }}</span></td>
<td class="num">{% if it.replacement_cost.low is not none %}{{ money(it.replacement_cost.low) }}{% if it.replacement_cost.high != it.replacement_cost.low %} – {{ money(it.replacement_cost.high) }}{% endif %}{% else %}—{% endif %}</td>
<td class="small">{% if it.replacement_cost.url %}<a href="{{ it.replacement_cost.url }}">{{ it.replacement_cost.source }}</a> · {{ (it.replacement_cost.retrieved_at or '')[:10] }}{% elif it.appraisal_reason %}{{ it.appraisal_reason }}{% endif %}</td></tr>{% endfor %}
</table></div>

<h2>Review queue ({{ p.review_queue|length }})</h2>
<div class="tablewrap"><table><tr><th>Line</th><th>Reason</th></tr>
{% for e in p.review_queue %}<tr><td class="small">{{ e.ref_id }}</td><td>{{ e.reason }}</td></tr>{% endfor %}</table></div>

{% if p.excluded_by_policyholder %}<h2>Left out at the policyholder's request</h2>
<div class="tablewrap"><table><tr><th>Line</th><th>Statement</th><th>Frame</th></tr>
{% for e in p.excluded_by_policyholder %}<tr><td class="small">{{ e.ref_id }}</td><td>{{ e.statement }}</td><td><a href="{{ frame(e.frame_ref) }}">frame</a></td></tr>{% endfor %}</table></div>{% endif %}

{% if locale %}<h2>Locale check: {{ locale.books|length }} books priced in {{ locale.country_a }} and {{ locale.country_b }}</h2>
<div class="tablewrap"><table><tr><th>Book</th><th>{{ locale.country_a }} ({{ locale.currency_a }})</th><th>{{ locale.country_b }} ({{ locale.currency_b }})</th></tr>
{% for b in locale.books %}<tr><td>{{ b.title }}</td><td class="num">{{ money(b.a) if b.a is not none else "—" }}{% if b.a_conv %} <span class="small muted">converted</span>{% endif %}</td><td class="num">{{ money(b.b) if b.b is not none else "—" }}{% if b.b_conv %} <span class="small muted">converted</span>{% endif %}</td></tr>{% endfor %}
</table></div>{% endif %}

{% if metrics %}<h2>Cost and latency</h2>
<div class="tablewrap"><table><tr><th>Stage</th><th>Seconds</th><th>Model calls</th><th>Tokens in / out</th><th>HTTP (cached)</th><th>USD</th></tr>
{% for name, s in metrics.stages.items() %}<tr><td>{{ name }}</td><td class="num">{{ "%.1f"|format(s.seconds) }}</td><td class="num">{{ s.calls }}</td><td class="num">{{ s.in_tokens }} / {{ s.out_tokens }}</td><td class="num">{{ s.http_calls }} ({{ s.cache_hits }})</td><td class="num">{{ "%.4f"|format(s.usd) }}</td></tr>{% endfor %}
<tr><td><b>Live voice session</b></td><td></td><td></td><td class="num small">{{ metrics.live.tokens }}</td><td></td><td class="num">{{ "%.4f"|format(metrics.live.usd) }}</td></tr>
<tr><td><b>Total per sweep</b></td><td class="num">{{ "%.0f"|format(metrics.time_to_packet_s or 0) }} after sweep end</td><td></td><td></td><td></td><td class="num"><b>{{ "%.4f"|format(metrics.total_usd) }}</b></td></tr>
</table></div>{% endif %}
<p class="small muted">Generated from claim_packet.json. Totals are computed in code from the lines above. Frame links open the saved frame that each line was read from.</p>
</main></body></html>
"""


def render(packet: ClaimPacket, sweep_dir: Path, defects: list[str], locale: dict | None, metrics: dict | None) -> Path:
    crops = save_crops(packet, sweep_dir)
    cur = packet.sweep.currency

    def money(v: float | None) -> str:
        return "—" if v is None else f"{cur} {v:,.0f}"

    def dim(v: float | None) -> str:
        return "—" if v is None else f"{v:g}"

    def frame(ref: str) -> str:
        return ref.split("#", 1)[0]

    env = Environment(autoescape=select_autoescape(["html"]))
    html = env.from_string(TEMPLATE).render(p=packet, t=packet.totals, r=packet.room, crops=crops, defects=defects,
                                            locale=locale, metrics=metrics, money=money, dim=dim, frame=frame)
    out = sweep_dir / "report.html"
    out.write_text(html)
    return out
