"""
metrics.py
-------------
The semantic layer: every business metric defined exactly once, in code.

The problem this solves is the one that actually breaks analytics teams.
"Loss ratio" gets written in a dashboard, a board pack, a regulatory
filing and an actuary's notebook, and the four disagree - usually because
one includes pharmacy, one nets out reinsurance, and nobody wrote down
which. The numbers are not wrong so much as unowned.

Here a metric is a DEFINITION, not a query:

    Metric(name="loss_ratio",
           numerator="SUM(paid_amount)",
           denominator="SUM(premium_collected)",
           grain="member_month",
           filters=["status = 'paid'"])

Queries are COMPILED from a metric plus a set of dimensions, so a
dashboard cannot express "loss ratio, but computed slightly differently".
Two consequences follow, and both are the point:

  1. Every figure on every screen traces to one definition.
  2. A definition CHANGE becomes a reviewable event with a measurable
     blast radius, which is what `integrity/metric_regression.py` exists
     to compute. Without a semantic layer you cannot even ask "which
     historical numbers did this change move?", because there is no
     single thing that changed.

DIMENSIONS are declared too, so a request for a dimension that does not
exist fails loudly at compile time rather than silently returning a
cartesian product.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dimension:
    name: str
    sql: str
    description: str


@dataclass(frozen=True)
class Metric:
    """A metric is a ratio (or a plain aggregate when denominator is None),
    plus the grain it is valid at and the filters it always carries."""

    name: str
    label: str
    numerator: str
    denominator: str | None
    grain: str
    description: str = ""
    unit: str = "ratio"
    filters: tuple = ()
    higher_is_better: bool | None = None
    owner: str = "actuarial"

    def compile(self, dimensions: list[str], where: list[str] | None = None,
                registry=None) -> str:
        reg = registry or REGISTRY
        unknown = [d for d in dimensions if d not in reg.dimensions]
        if unknown:
            raise ValueError(
                f"unknown dimension(s) {unknown} for metric '{self.name}'; "
                f"declared dimensions are {sorted(reg.dimensions)}")

        selects = [f"{reg.dimensions[d].sql} AS {d}" for d in dimensions]
        conds = list(self.filters) + list(where or [])
        where_sql = f"WHERE {' AND '.join(conds)}" if conds else ""
        group_sql = f"GROUP BY {', '.join(str(i + 1) for i in range(len(dimensions)))}" \
            if dimensions else ""

        if self.denominator:
            value = (f"CASE WHEN {self.denominator} = 0 THEN NULL "
                     f"ELSE {self.numerator} / {self.denominator} END")
        else:
            value = self.numerator

        cols = ",\n       ".join(selects + [f"{value} AS value",
                                            f"{self.numerator} AS numerator"]
                                 + ([f"{self.denominator} AS denominator"]
                                    if self.denominator else []))
        return (f"SELECT {cols}\nFROM {reg.base_relation}\n{where_sql}\n{group_sql}"
                .replace("\n\n", "\n").strip())


@dataclass
class Registry:
    base_relation: str
    metrics: dict = field(default_factory=dict)
    dimensions: dict = field(default_factory=dict)

    def add_metric(self, m: Metric):
        if m.name in self.metrics:
            raise ValueError(f"metric '{m.name}' already defined - a metric may "
                             f"be defined exactly once, that is the whole point")
        self.metrics[m.name] = m

    def add_dimension(self, d: Dimension):
        self.dimensions[d.name] = d

    def get(self, name) -> Metric:
        if name not in self.metrics:
            raise ValueError(f"unknown metric '{name}'; defined metrics are "
                             f"{sorted(self.metrics)}")
        return self.metrics[name]


# =====================================================================
#  The Aegis registry
# =====================================================================
# `vw_member_month` is the conformed base: one row per member per month,
# carrying that month's premium and that month's claims. Defining every
# metric against a single relation at a single grain is what makes the
# ratios comparable - a numerator summed at claim grain over a denominator
# summed at member-month grain is the classic way to produce a loss ratio
# that is wrong by a factor nobody can find.
REGISTRY = Registry(base_relation="vw_member_month")

for d in [
    Dimension("month", "month", "calendar month, YYYY-MM"),
    Dimension("region", "region", "member's region"),
    Dimension("plan", "plan", "metal tier"),
    Dimension("product", "product", "HMO / PPO / EPO"),
    Dimension("channel", "channel", "acquisition channel"),
    Dimension("age_band", "CAST(age_band AS VARCHAR)", "10-year age band"),
]:
    REGISTRY.add_dimension(d)

for m in [
    Metric(
        name="loss_ratio", label="Loss ratio",
        numerator="SUM(paid_amount)", denominator="SUM(premium_collected)",
        grain="member_month", unit="ratio", higher_is_better=False,
        description=("paid claims divided by premium collected. The single "
                     "most-restated metric in insurance, which is why its "
                     "definition lives here and nowhere else."),
    ),
    Metric(
        name="pmpm", label="PMPM (paid per member per month)",
        numerator="SUM(paid_amount)", denominator="COUNT(*)",
        grain="member_month", unit="currency", higher_is_better=False,
        description=("paid claims per member-month. The correct forecasting "
                     "target: total cost confounds price with membership, and "
                     "a shrinking book can hide a rising trend."),
    ),
    Metric(
        name="premium_pmpm", label="Premium PMPM",
        numerator="SUM(premium_collected)", denominator="COUNT(*)",
        grain="member_month", unit="currency",
        description="premium collected per member-month; the loss-ratio denominator",
    ),
    Metric(
        name="denial_rate", label="Denial rate",
        numerator="SUM(denied_lines)", denominator="SUM(total_lines)",
        grain="member_month", unit="ratio", higher_is_better=False,
        description="denied claim lines over all adjudicated lines",
        owner="operations",
    ),
    Metric(
        name="utilisation_per_1000", label="Utilisation per 1,000 members",
        numerator="SUM(total_lines) * 1000.0", denominator="COUNT(*)",
        grain="member_month", unit="count", owner="clinical",
        description="claim lines per 1,000 member-months",
    ),
    Metric(
        name="member_months", label="Member months",
        numerator="COUNT(*)", denominator=None,
        grain="member_month", unit="count",
        description="exposure denominator; every ratio above rests on it",
    ),
    Metric(
        name="paid_amount", label="Paid claims",
        numerator="SUM(paid_amount)", denominator=None,
        grain="member_month", unit="currency",
        description="total paid claims; the loss-ratio numerator",
    ),
]:
    REGISTRY.add_metric(m)


def compile_query(metric_name: str, dimensions: list[str],
                  where: list[str] | None = None, registry=None) -> str:
    reg = registry or REGISTRY
    return reg.get(metric_name).compile(dimensions, where, reg)


def metric_catalog(registry=None) -> list[dict]:
    """Machine-readable catalogue. The AI layer reads THIS rather than the
    database schema, so a natural-language question can only resolve to a
    governed metric and a declared dimension."""
    reg = registry or REGISTRY
    return [{
        "name": m.name, "label": m.label, "unit": m.unit, "grain": m.grain,
        "owner": m.owner, "description": m.description,
        "higher_is_better": m.higher_is_better,
        "definition": (f"{m.numerator} / {m.denominator}" if m.denominator
                       else m.numerator),
        "filters": list(m.filters),
    } for m in sorted(reg.metrics.values(), key=lambda x: x.name)]
