# Y-SNP Diversity

Faith's Phylogenetic Diversity on Y-SNP haplogroup trees from YFull, grouped by country / region / language / language family.

Faith's PD (Faith 1992, <https://doi.org/10.1016/0006-3207(92)91201-3>) is the classical metric for phylogenetic diversity — the sum of branch lengths in the minimum subtree connecting a set of tips.

## Quick start

```bash
uv sync
uv run ysnp-diversity G-Z6744 -g language_family
uv run ysnp-diversity G-Z6744 -g language_family --impl fast       # vectorised path
uv run ysnp-diversity G-Z6744 -g language_family --export pd.svg   # or .png / .html
```

YFull HTML is cached to `data/cache/` (TTL 3 days).

## Grouping modes

| Mode | Group key |
|---|---|
| `country` | `country_iso` (ISO 3166-1 alpha-2) |
| `region` | `country_subdivision` (ISO 3166-2) |
| `language` | `language` (ISO 639-3) |
| `language_family` | first matching family in `data/language_families.json` |

Samples missing the requested attribute are dropped from the grouping.

## PD implementations

| `--impl` | Backend | Notes |
|---|---|---|
| `skbio` (default) | `skbio.diversity.alpha.faith_pd` | canonical reference |
| `fast` | `significance.faith_pd_fast` | bit-equal, ~3-30× faster on large trees |

The fast path is cross-validated against scikit-bio on 500+ random subsets in `tests/test_fast_pd.py`. `significance.py` also exposes `null_pd_distribution` and `compute_significance` (SES.PD + permutation p-value), not surfaced via CLI.

## Tree construction

```
branch  edge = formed_ybp - tmrca_ybp             (0 for star branches)
sample  edge = effective_tmrca(branch) - age_ybp  (age 0 for modern tips)
```

Star branches (`G-X*`) carry no SNP or age; samples under them get their terminal edge from the nearest ancestral TMRCA.

`skbio.faith_pd` requires the root to have exactly two children. YFull's root often has more, so `newick.py` wraps the tail under a zero-length synthetic node.

## Credits

`data/languoid.csv` is a [Glottolog](https://glottolog.org/) snapshot (CC-BY 4.0).
