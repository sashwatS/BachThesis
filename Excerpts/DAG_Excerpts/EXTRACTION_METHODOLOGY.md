# Chain A excerpt extraction — methodology

This document is written for downstream LLM consumption. It explains, in concrete and reproducible terms, how the six full-chain Chain A excerpts in `ChainA_Excerpts/Bank/*.json` and `ChainA_Excerpts/Industry/*.json` were located inside the source corpus and validated as preserving the full causal chain.

## 1. Objective and ontology

**Chain A** is a fixed 4-node causal DAG used to benchmark LLM causal-extraction over ESG/sustainability reports. The nodes and the directed signed edges are:

| Node | Label                              | Canonical concept                                                                                       |
|------|------------------------------------|---------------------------------------------------------------------------------------------------------|
| CG   | Corporate Governance               | board oversight, committees, voting, exec compensation, risk-management framework, policy ownership     |
| ER   | Environmental Reporting            | TCFD, CDP, SBTi, NZBA, SFDR, EU Taxonomy, CSRD/ESRS, GRI, UNEP FI, sustainability reports/disclosures   |
| GHG  | GHG Emissions                      | Scope 1/2/3 emissions, carbon footprint, net-zero, decarbonisation, climate-neutral economy, fossil exp.|
| MBV  | Market-based Firm Value            | shareholder value, long-term performance, financial strength, profitability, value creation, AUM/loanbook|

Edges with their hypothesised signs:

```
CG ──(+)──▶ ER ──(−)──▶ GHG ──(−)──▶ MBV
```

A passage qualifies as a **full-chain excerpt** only if all four nodes are co-present in coherent prose with explicit causal claims linking them in the order above, and the edge signs match the hypothesis.

## 2. Source corpus

Two directories of `.jsonl` files (one JSON record per physical text block as extracted from PDFs):

```
5050BankIndustryReports/   # 30 reports — banks + industrials
UnusedReports/             # 14 additional reports
```

Each line in a `.jsonl` file has the shape:

```json
{"text": "...full block of text from the PDF...",
 "section": "ESRS2",
 "category": "General",
 "page": 17}
```

`section` carries the report-internal classification (e.g. `ESRS2`, `EU-Tax`, `S4`, `G1`). `page` is the physical PDF page. The block index in the file (1-indexed line number) is treated as a stable row id; we refer to it as `L<n>`. PDF text-extraction artefacts mean two-column layouts can interleave fragments of the left and right columns into a single block, so block-level prose is sometimes noisy.

## 3. Methodology evolution: strict → relaxed

### 3.1 Strict-wording pass (one excerpt)

The first pass used tight per-node regexes that demanded the canonical surface forms (e.g. ER required `tcfd|cdp|sbti|climate disclosure|non-financial report`; MBV required `shareholder value|investor confidence|market capitalisation|firm value`). Combined with a same-section, ≤5-page-span, ≥4-causal-marker constraint, **only one** passage in the entire corpus survived: `Banca Mediolanum S.p.A. MEDIOLANUM_GROUP_NON-FINANCIAL_STATEMENT_2023.jsonl`, lines `L958–L967`, pp. 177–179, ESRS2, MIFL Responsible Investment Policy / Stewardship and voting.

That single hit became `ChainA_Excerpts/Bank/BANCA-MEDIOLANUM_ChainA.json`. It is the only excerpt in the set that matches the strict-wording specification on every node.

### 3.2 Why one excerpt is not enough

Chain A benchmarks need multiple full-chain examples for stability. The user explicitly relaxed the constraint:

> "relax wording requirements and try for 2 more. its okay if the wording is a bit more vague, but there should still be a decent resemblance to all the nodes in the chain."

So a second pass widened each node regex to admit synonymous surface forms commonly used across European bank sustainability reports and industrial corporate-responsibility reports, while keeping the structural requirements (proximity, single section, causal marker density).

## 4. Relaxed-wording scanner

The implementation is a self-contained Python script (`/tmp/relaxed_scan.py`). It scans every `.jsonl` in both directories, applies the relaxed node regexes and the causal-marker regex over a sliding 15-row window (the anchor row plus ±7 rows), and emits a candidate list ranked by causal-marker density.

### 4.1 Relaxed node regexes

```python
RELAXED_CG = re.compile(
    r'corporate governance|\bgovernance structure|\bgovernance\b|board of directors|'
    r'supervisory board|management board|sustainability committee|risk committee|audit committee|'
    r'oversight|stewardship|voting policy|voting rights|voting right|proxy voting|'
    r'remuneration policy|\bremuneration\b|compensation policy|executive compensation|'
    r'internal control|executive director|non-executive|chief risk officer|chief sustainability officer|'
    r'governance structures',
    re.IGNORECASE)

RELAXED_ER = re.compile(
    r'\btcfd\b|task force on climate[- ]related|\bcdp\b|\bsbti\b|science[- ]based targets?|'
    r'climate disclosure|non[- ]financial report|non[- ]financial statement|'
    r'sustainability report|sustainability disclosure|emissions reduction target|'
    r'\bsfdr\b|sustainable finance disclosure|\bnfrd\b|\bcsrd\b|\besrs\b|\bgri\b|\bgri [0-9]|'
    r'transition plan|eu taxonomy|taxonomy regulation|'
    r'unep fi|un environment programme finance initiative|\bprb\b|'
    r'principles for responsible banking|\bpri\b|principles for responsible investment|'
    r'net[- ]zero banking alliance|\bnzba\b|un global compact|ipsf|pcaf|'
    r'partnership for carbon accounting financials|sustainability reporting',
    re.IGNORECASE)

RELAXED_GHG = re.compile(
    r'ghg emission|greenhouse gas|scope [123]|carbon emission|carbon footprint|'
    r'net[- ]?zero|paris agreement|decarboni[sz]|low[- ]carbon|climate[- ]neutral|carbon neutral|'
    r'emission reduction|emissions reduction|reduce emissions|financed emissions|'
    r'emission intensity|emissions intensity|carbon intensity|climate transition|'
    r'climate risk|climate risks|climate change',
    re.IGNORECASE)

RELAXED_MBV = re.compile(
    r'shareholder value|long[- ]term value|long[- ]term performance|firm value|'
    r'investor confidence|market capitalisation|market capitalization|enterprise value|'
    r'\bprofitability\b|profitable growth|earnings per share|return on equity|\broe\b|'
    r'financial performance|sustainable value|creating value|create value|'
    r'long[- ]term financial|long[- ]term shareholder|economic value|value creation|'
    r'financial strength|financial resilience',
    re.IGNORECASE)
```

These regexes preserve the ontological intent of each node but admit:

- For **MBV**: "long-term performance", "financial strength", "value creation", "creating sustainable value", "create value", "financial resilience", "profitability". This was the most-loosened node — strict MBV vocabulary ("shareholder value", "market cap") is rare in bank-sustainability reports, but value-creation framing is universal.
- For **ER**: the broader European disclosure regime — SFDR, EU Taxonomy, CSRD/ESRS, NZBA, UNEP FI / PRB / PRI, UN Global Compact, GRI, Pillar 3 ESG Reporting, plus the strict TCFD / CDP / SBTi.
- For **GHG**: physical climate framing in addition to emission counts — "climate-neutral economy", "low-carbon", "decarbonisation", "climate transition", "climate risk".
- For **CG**: any board-level / committee-level / risk-framework / stewardship surface form, including three-lines-of-defence governance, remuneration / executive compensation, and CRO / CSO offices.

### 4.2 Causal-marker regex

```python
CAUSAL_MARKERS = re.compile(
    r'\bbecause\b|\bthereby\b|\btherefore\b|\bthus\b|\bhence\b|\bas a result\b|'
    r'\bso that\b|leads? to|lead to|drives?|driven by|promotes?|enables?|supports?|'
    r'contributes? to|helps? (?:to )?(?:realise|realize|achieve|deliver|reduce|mitigate|ensure)|'
    r'in order to|to ensure|to reduce|to mitigate|to achieve|to enhance|to support|'
    r'aligns? with|aligned with|in line with|ensures?|requires?|impacts?|affects?|influences?|'
    r'improves?|reinforces?|strengthens?|fosters?|facilitates?|accelerates?|reduces?|mitigates?|'
    r'addresses?|integrates?|embeds?|commits? to|committed to|is essential to|is fundamental to|'
    r'underpins?|depends? on|result in|results in|resulting in|has (?:a )?significant impact',
    re.IGNORECASE)
```

This captures both classical discourse connectives ("therefore", "thus") and the "purpose / effect" verb phrases ESG reports lean on ("aligned with", "in line with", "to mitigate", "drives", "underpins").

### 4.3 Sliding-window aggregation

```python
for i in range(len(recs)):
    lo, hi = max(0, i-7), min(len(recs), i+8)         # 15-row window
    wtext = " ".join(r['text'] for r in recs[lo:hi])
    pages = [r['page'] for r in recs[lo:hi] if r.get('page')]
    page_span = max(pages) - min(pages)               # at most 5
    if page_span > 5:                continue
    if not (CG and ER and GHG and MBV in wtext):     continue
    if causal_marker_count < 4:      continue
    emit(anchor=i+1, lo, hi, page_span, marker_count)
```

Constraints, in plain language:

1. **15-row window** centred on each row (anchor ± 7 rows). This corresponds to roughly a one-page-and-a-half stretch of source PDF text in a typical sustainability report.
2. **Page span ≤ 5** to keep the window inside one logical sub-section, not across major chapters.
3. **All four nodes must match** somewhere in the window (relaxed regex).
4. **At least 4 causal-marker hits** in the window. Most accepted candidates have 40–120.
5. After the per-row emit, overlapping anchors within ≤ 6 rows are collapsed to the highest-marker-count anchor — this dedup avoids reporting the same passage three or four times when adjacent anchor rows all see the same window.

### 4.4 Already-used skip-list

The strict-pass winner (Mediolanum) plus the manually-promoted Deutsche Bank and AS SEB banka excerpts were added to a `SKIP` set so the relaxed scan would not re-suggest them:

```python
SKIP = {
    "Banca Mediolanum S.p.A. MEDIOLANUM_GROUP_NON-FINANCIAL_STATEMENT_2023.jsonl",
    "deutsche-bank.jsonl",
    "AS SEB banka ESG report 2023.jsonl",
}
```

## 5. Output of the relaxed scan

The relaxed scan over the remaining 41 reports produced **117 candidate windows in 35 reports**. The top of the marker-density ranking was:

```
vodafone.jsonl                                L302 p68 span=4 sec=ESRS2 mk=123
ABN AMRO Bank N.V. Impact_Report_2023.jsonl   L73  p38 span=4 sec=ESRS2 mk=106
vodafone.jsonl                                L309 p69 span=2 sec=ESRS2 mk=106
vodafone.jsonl                                L291 p64 span=3 sec=ESRS2 mk=89
bayer.jsonl                                   L128 p46 span=5 sec=ESRS2 mk=75
stellantis.jsonl                              L308 p94 span=3 sec=ESRS2 mk=73
ABN AMRO Bank N.V. Impact_Report_2023.jsonl   L80  p40 span=3 sec=ESRS2 mk=71
carlsberg.jsonl                               L123 p57 span=3 sec=ESRS2 mk=66
carlsberg.jsonl                               L116 p56 span=5 sec=ESRS2 mk=63
ALPHA BANK S.A.-Sustainability-Report-2023... L38  p22 span=5 sec=ESRS2 mk=58
Bank of Ireland Group plc BOI-Sust...         L48  p33 span=5 sec=ESRS2 mk=58
adidas.jsonl                                  L122 p25 span=2 sec=ESRS2 mk=56
stellantis.jsonl                              L316 p97 span=2 sec=ESRS2 mk=56
...
```

`mk` = causal-marker count in the window. `span` = page span. `sec` = section tag at the anchor.

## 6. Manual verification before commit

Marker density alone does not guarantee a real Chain A — a window can score high because it's full of text that uses a lot of "in line with" and "ensures" language without any actual causal narrative across the four ontological concepts. So each shortlisted candidate was opened (read 15–25 surrounding rows in the source `.jsonl`) and checked against four manual criteria:

1. **Coherent narrative**, not a layout artefact. Multi-column PDF interleaving is common; a candidate that scored 100+ markers because it's actually a risk-register table with adjacent columns mashed together is rejected.
2. **All four nodes anchored to actual textual content**, not navigation headers or footers. ABN AMRO L73 is excluded for this reason — its "governance" hit is a meta-comment in the methodology section ("governance is not included as an impact area in the Impact Report") rather than a substantive CG claim.
3. **The signed edges match the Chain A hypothesis** (CG→ER positive, ER→GHG negative, GHG→MBV negative).
4. **Causal markers attach to the right edges**, e.g. the candidate must contain claims like "voting policy supports proposals that require disclosure" (CG→ER) and "GHG mitigation is essential to long-term performance" (GHG→MBV), not just "we report in line with TCFD" in isolation.

Of the top 13 candidates, the three accepted into the final set were Bank of Ireland, Carlsberg, and Bayer. The rejections and their reasons:

| Candidate                          | Reason for rejection                                                                 |
|------------------------------------|--------------------------------------------------------------------------------------|
| Vodafone L302 p68 (mk=123)         | Multi-column risk-register layout; window is dense with markers but not coherent prose. |
| ABN AMRO L73 / L80 (mk=106 / 71)   | "Governance" appears as a methodological caveat ("not included as an impact area"); CG anchor weak. |
| Stellantis L308 / L316             | Strong CG/ER/GHG, but MBV anchor is implicit and dispersed.                          |
| Adidas L122                        | Brand-equity framing of MBV is too marketing-coded; chain is not causal.             |
| Alpha Bank L38                     | Section traverses two unrelated sub-sections; chain is not contiguous.               |

## 7. Per-excerpt rationale

The six final excerpts, with the chain-recognition reasoning that promoted each one.

### 7.1 Banca Mediolanum (strict-wording, Bank)

- **File / window**: `Banca Mediolanum S.p.A. MEDIOLANUM_GROUP_NON-FINANCIAL_STATEMENT_2023.jsonl`, L958–L967, pp. 177–179, section `ESRS2`, sub-section "Stewardship and voting (MIFL Responsible Investment Policy)".
- **Why it qualifies under strict wording**: every node has a canonical surface form. CG = "voting policy / board of directors / oversight"; ER = "Task Force on Climate-related Financial Disclosure (TCFD)"; GHG = "Greenhouse gas emissions / Paris Agreement / greenhouse gas emission reduction targets"; MBV = "long-term shareholder value".
- **Edges**: voting policy supports TCFD-aligned disclosure (CG→ER, +); TCFD-aligned disclosure feeds the assessment that pushes Paris-aligned reduction targets (ER→GHG, −); stewardship that mitigates climate impact "helps to realise long-term shareholder value" (GHG→MBV, −).

### 7.2 Deutsche Bank (relaxed-wording, Bank)

- **File / window**: `deutsche-bank.jsonl`, L44–L67, pp. 15–19, section `ESRS2`, sub-section "Sustainability strategy and implementation (Non-Financial Report 2023)".
- **Promoted by**: hand-promoted before the relaxed scan ran, after a manual read of the Non-Financial Report 2023 strategy chapter. CG anchored on "Supervisory Board / Management Board / Group Sustainability Committee / governance structures / top executives' performance-based compensation". ER anchored on "Non-Financial Report 2023 / GRI 2-22 / UNEP FI / UN Global Compact / PRI / PRB / NZBA". GHG anchored on "climate-neutral economy / Climate Risk Management Framework / Net-Zero Banking Alliance". MBV anchored on "creating sustainable value / loan book of €479 billion / €559 + €896 billion AuM / sustainable society and economy".
- **Edges**: board-level oversight integration drives the disclosure regime (CG→ER, +); the framework portfolio drives Climate Risk Management Framework + net-zero targets (ER→GHG, −); climate/net-zero targets are tied to top-executive compensation and to DWS's "creating sustainable value" mandate across the loan-book and AuM (GHG→MBV, −).

### 7.3 AS SEB banka (relaxed-wording, Bank)

- **File / window**: `AS SEB banka ESG report 2023.jsonl`, L654–L676, pp. 29–30, section `EU-Tax`, sub-section "Net zero aligned 2030 targets / Impact, risk and opportunities management".
- **Promoted by**: hand-promoted on the same pass as Deutsche Bank, after a manual read of the SEB banka EU-Taxonomy / strategy section. CG anchored on "strong governance / robust sustainability policy framework / risk management". ER anchored on "SFDR / EU Taxonomy (Reg 2020/852) / NZBA Guidelines for Climate Target Settings for Banks / UNEP FI PRB". GHG anchored on "Carbon Exposure Index / fossil fuel credit exposure / net zero aligned 2030 interim targets". MBV anchored on "long-term performance / financial strength and resilience / create value for the planet, people and society".
- **Edges**: governance + sustainability policy framework drives the disclosure regime (CG→ER, +); the disclosure regime drives the Carbon Exposure Index and the net-zero 2030 sectoral targets (ER→GHG, −); sustainability commitments are explicitly tied to long-term performance, financial strength/resilience, and value creation (GHG→MBV, −).

### 7.4 Bank of Ireland (relaxed-wording, Bank)

- **File / window**: `Bank of Ireland Group plc BOI-Sustainablilty-Report-2023.jsonl`, L41–L53, pp. 29–34, section `ESRS2`, sub-section "Decarbonising our own Operations / Managing Climate-related Risks".
- **Selected from relaxed scan** (anchor L48 p33, span=5, mk=58). Verified manually: the passage runs continuously from operational Scope 1/2 emissions reporting (L41) through climate-risk-management governance (L43–L50) to regulatory profitability impact (L53). CG anchored on "Group Risk Management Framework / 1LOD / 2LOD / 3LOD / board of management / Group Strategy". ER anchored on "SBTi 2030 target / TCFD recommendations / ECB's guidelines on climate-related and environmental risks / Pillar 3 ESG Reporting / ICAAP / Sustainability Report 2023". GHG anchored on "GHG emissions / Scope 1/2/3 / low carbon economy / financed emissions / climate change". MBV anchored on "financial soundness / financial loss to the Group / asset values / long term franchise impacts / Bank's profitability".
- **Edges**: three-lines-of-defence governance drives TCFD/ECB-aligned disclosure and ICAAP climate-risk reporting (CG→ER, +); SBTi-aligned disclosure + Pillar 3 financed-emissions reporting drive operational GHG reduction (ER→GHG, −); unmitigated climate/GHG risk propagates through credit, strategic and regulatory channels into financial loss, impaired asset values, franchise damage and lower Bank profitability (GHG→MBV, −).

### 7.5 Carlsberg (relaxed-wording, Industry)

- **File / window**: `carlsberg.jsonl`, L123–L130, pp. 57–59, section `ESRS2`, sub-section "Climate change (E1) / Climate transition plan (E1-1) / Targets and actions (E1-3, E1-4)".
- **Selected from relaxed scan** (anchor L123 p57, span=3, mk=66). Carlsberg's E1 climate disclosure has a clean ESRS structure that aligns naturally with Chain A: SBM-3 (impacts), IRO-1 (assessment), GOV-1 (governance), E1-1 (transition plan), E1-3/E1-4 (targets). CG anchored on "Carlsberg governance model / GOV-1 / ESG target sponsors / Group Sustainability & ESG function / Integrated Supply Chain Sustainability function". ER anchored on "TCFD / SBTi / E1-1 / E1-3 / E1-4 / SBM-3 / IRO-1 / Sustainability statement". GHG anchored on "GHG emissions / Scope 1/2/3 / net zero value chain / climate-neutral economy / 30% reduction by 2030 / net zero by 2040". MBV anchored on "financial risk to the business / overall business strategy and financial planning processes / organic growth trajectory / business transformation / significant investment in decarbonising".
- **Edges**: governance model + ESG target sponsors approve and own the transition plan that is framed by TCFD / SBTi / ESRS E1 (CG→ER, +); TCFD/SBTi-aligned disclosure drives the Scope 1–3 net-zero pathway (ER→GHG, −); the transition plan is explicitly anchored in business strategy + financial planning, requires business transformation and decarbonisation investment, and shields the business from carbon-pricing financial risk (GHG→MBV, −).

### 7.6 Bayer (relaxed-wording, Industry)

- **File / window**: `bayer.jsonl`, L122–L139, pp. 43–51, section `ESRS2`, sub-section "Sustainability Strategy — Focus on Agriculture / Regenerative agriculture / Greenhouse Gas Intensities".
- **Selected from relaxed scan** (anchor L128 p46, span=5, mk=75). The Bayer chain is the loosest of the six because the MBV anchor is intermediated through the agricultural customer base ("farm incomes / farm profitability / yield increase / productivity") rather than Bayer's own market cap or shareholder value. The chain is still complete: CG anchored on "Sustainability Strategy chapter 2 = Corporate Governance + executive compensation + stakeholder engagement". ER anchored on "Bayer Sustainability Report 2023 / 2024 Sustainability Report / externally-reviewed methodology / biennial GHG-intensity progress reporting / Bayer Carbon Programs". GHG anchored on "reducing on-field greenhouse gas emissions / increasing carbon sequestration / mitigate climate change / 443 kg CO₂e per metric ton GHG-intensity baseline". MBV anchored on "yield increase / farm incomes / farm profitability / sustainable food production system / transform the agricultural sector at scale".
- **Edges**: the Sustainability-Strategy / Corporate-Governance chapter underpins commitments and the biennial progress-reporting obligation in the Sustainability Report (CG→ER, +); the report regime is tied to the GHG-intensity baseline and the on-field emission-reduction + carbon-sequestration commitments (ER→GHG, −); GHG-mitigating regenerative-agriculture practices are explicitly framed as the mechanism for yield increase, farm incomes, farm profitability and food security — Bayer's value-creation thesis across its agricultural value chain (GHG→MBV, −).

## 8. JSON output schema

Every accepted excerpt is written to one file under `ChainA_Excerpts/Bank/` or `ChainA_Excerpts/Industry/`, named `<COMPANY>_ChainA.json`. The file is a JSON array (currently always with one element) of records with this schema:

```jsonc
{
  "excerpt": "<verbatim or near-verbatim PDF text with [...] marking elision>",
  "source_location": {
    "file":       "<basename of the source .jsonl>",
    "pages":      [<int>, ...],
    "lines":      [<1-indexed L<n> from the .jsonl>, ...],
    "section":    "<source-record section tag, e.g. ESRS2 / EU-Tax / S4>",
    "subsection": "<human label of the report sub-section>"
  },
  "nodes_mentioned": ["CG", "ER", "GHG", "MBV"],
  "surface_forms": {
    "CG":  ["<exact substring 1>", "<exact substring 2>", ...],
    "ER":  [...],
    "GHG": [...],
    "MBV": [...]
  },
  "edges_claimed": [
    {
      "from": "CG",  "to": "ER",
      "sign": "+",
      "direction": "forward",
      "explicit": true,
      "evidence": "<verbatim quote(s) from the excerpt + a square-bracketed paraphrase>"
    },
    { "from": "ER",  "to": "GHG", "sign": "-", ... },
    { "from": "GHG", "to": "MBV", "sign": "-", ... }
  ],
  "causal_markers": ["<short discourse cue 1>", "<short discourse cue 2>", ...],
  "notes": "<free-form: chain narrative + which wording tolerances were relaxed>"
}
```

`surface_forms[node]` is the canonical evidence list for that node — every entry is a literal substring of the source text (or close to it). `edges_claimed[i].evidence` quotes the passage that supports the edge, and the bracketed `[...]` paraphrase is the methodology author's interpretation of the underlying causal claim. `notes` documents which wording tolerances were relaxed for that excerpt and is the place a downstream LLM should look to understand why a particular passage is borderline-acceptable rather than canonical.

## 9. Wording-tolerance summary

Across the five relaxed-wording excerpts, these are the substitutions that needed to be allowed for the chain to remain intact:

| Node | Strict surface form expected            | Relaxed surface forms accepted (examples)                                                               |
|------|-----------------------------------------|---------------------------------------------------------------------------------------------------------|
| CG   | board of directors, voting policy       | Supervisory/Management Board, three-lines-of-defence (1LOD/2LOD/3LOD), GOV-1, governance model, ESG target sponsors, Sustainability Strategy chapter 2 |
| ER   | TCFD, CDP, SBTi                         | Non-Financial Report 2023, GRI 2-22, UNEP FI / PRB / PRI, UN Global Compact, NZBA, SFDR, EU Taxonomy, ICAAP, Pillar 3 ESG, ESRS E1/E1-1/E1-3/E1-4, externally-reviewed Sustainability Report methodology |
| GHG  | greenhouse gas emissions, Paris-aligned | climate-neutral economy, Climate Risk Management Framework, Carbon Exposure Index, fossil fuel credit exposure, net-zero by 2030/2040, GHG-intensity per metric ton crop |
| MBV  | shareholder value, market cap           | creating sustainable value, loan-book / AuM scale, long-term performance, financial strength and resilience, financial loss to the Group, Bank's profitability, organic growth trajectory, business transformation, farm incomes / farm profitability |

The relaxation is recorded explicitly in the `notes` field of every relaxed excerpt so a downstream LLM evaluating the corpus can distinguish strict-canonical (Mediolanum) from interpretation-dependent (Deutsche Bank, SEB banka, Bank of Ireland, Carlsberg, Bayer) examples.

## 10. Reproducing the scan

To re-run the relaxed scan from scratch:

1. Place the corpus in `5050BankIndustryReports/` and `UnusedReports/` siblings of `ChainA_Excerpts/`.
2. Update `SKIP` in `/tmp/relaxed_scan.py` to the set of files already used.
3. Run:
   ```bash
   python3 /tmp/relaxed_scan.py
   ```
4. Inspect the top of the marker-density ranking, manually verify each shortlisted candidate against the four criteria in §6, and write each accepted excerpt to its own `<COMPANY>_ChainA.json` under the appropriate `Bank/` or `Industry/` subdirectory.

The script is intentionally a single-file scanner with no external dependencies beyond the Python standard library, so it is portable across the corpus snapshots.

## 11. Limitations and known noise

- **PDF text-extraction artefacts**. Two-column layouts can produce a single `text` block that interleaves left- and right-column fragments. This inflates causal-marker counts in noisy passages and is the main reason the top-ranked candidate (Vodafone L302 mk=123) was rejected in §6.
- **MBV is the noisiest node**. "Market value" appears frequently in derivative / collateral / asset-valuation contexts that are not Chain A's MBV concept; the relaxed regex deliberately omits the bare phrase "market value" to avoid that noise. The included MBV terms ("financial strength", "long-term performance", "value creation") were chosen to track the firm-value-creation framing that ESG reports actually use.
- **Section labels are not perfectly reliable**. AS SEB banka's section tag at the anchor row is `EU-Tax`, even though the prose is substantively the ESRS2 strategy/governance disclosure. The constraint is "windows must stay within one section" not "windows must match a particular section", so this does not affect candidate selection.
- **Page-span ≤ 5 is a heuristic**. Sustainability reports occasionally interleave a chain across a wider page range; tightening the constraint to ≤ 3 pages would lose Bank of Ireland and Bayer; loosening it to ≤ 8 pages admits more spurious cross-chapter windows. Five is the empirical sweet spot for this corpus.
