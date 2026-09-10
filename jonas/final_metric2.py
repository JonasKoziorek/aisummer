# %%
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 1. Paths to your two CSV files
# ============================================================
GT_CSV_PATH = Path("/mnt/c/Users/pepaz/Downloads/ai_summer_school_dataset/test_example/gt.csv")
PRED_CSV_PATH = Path.home() / "aisummer" / "aisummer" / "jonas" / "optimistic_prime.csv"

# If you measured the total time it took to generate the predictions (in seconds):
PROCESSING_TIME_S = -1  # Set to your measured time, or 0.0

NUM_POSITIONS = 7

# ============================================================
# 2. Official Evaluation Metric (Loss in EUR)
# ============================================================
SECONDS_PER_HOUR = 3600
MINUTES_PER_HOUR = 60


@dataclass(frozen=True)
class LineConfig:
    plombs_per_hour: float = 3600.0   # Nominal line speed [plombs/hour] (D2)
    eur_per_minute: float = 0.1       # Delay penalty [EUR/min] (D3)
    technician_fee: float = 1.0       # Technician fee per error [EUR] (D5)
    downtime_per_error: float = 2.0   # Line downtime per error [min] (D6)
    batch_size: int = 50              # Batch size (D7)

    @property
    def time_budget_s(self) -> float:
        return SECONDS_PER_HOUR * self.batch_size / self.plombs_per_hour


@dataclass(frozen=True)
class ScoreBreakdown:
    error_cost: float
    overtime_s: float
    overtime_cost: float

    @property
    def total(self) -> float:
        return self.error_cost + self.overtime_cost


def evaluate(errors: int, processing_time_s: float, config: LineConfig) -> ScoreBreakdown:
    cost_per_error = config.downtime_per_error * (
        (config.plombs_per_hour / MINUTES_PER_HOUR) * config.eur_per_minute + config.technician_fee
    )
    error_cost = errors * cost_per_error

    overtime_s = max(0.0, processing_time_s - config.time_budget_s)
    overtime_cost = (
        (config.plombs_per_hour / SECONDS_PER_HOUR) * config.eur_per_minute * overtime_s
    )

    return ScoreBreakdown(
        error_cost=error_cost,
        overtime_s=overtime_s,
        overtime_cost=overtime_cost,
    )


# ============================================================
# 3. Load and Match the CSVs
# ============================================================
df_gt = pd.read_csv(GT_CSV_PATH, delimiter=";", dtype={"filename": str, "number": int})
df_pred = pd.read_csv(PRED_CSV_PATH, delimiter=";", dtype={"filename": str, "number": int})

# Clean whitespace
df_gt["filename"] = df_gt["filename"].str.strip()
df_pred["filename"] = df_pred["filename"].str.strip()

# Merge on filename
merged = pd.merge(df_gt, df_pred, on="filename", suffixes=("_true", "_pred"))

if len(merged) < len(df_gt):
    print(f"Warning: Only {len(merged)} / {len(df_gt)} filenames matched between both CSVs!")

# Check exact matches
merged["exact_match"] = merged["number_true"] == merged["number_pred"]

# ============================================================
# 4. Per-Digit Breakdown
# ============================================================
divisors = 10 ** np.arange(NUM_POSITIONS - 1, -1, -1)

true_digits = (merged["number_true"].to_numpy()[:, None] // divisors) % 10
pred_digits = (merged["number_pred"].to_numpy()[:, None] // divisors) % 10

per_position_acc = (true_digits == pred_digits).mean(axis=0) * 100.0
total_digit_acc = (true_digits == pred_digits).mean() * 100.0

# ============================================================
# 5. Compute Competition Score
# ============================================================
total_evaluated = len(merged)
errors = int((~merged["exact_match"]).sum())
exact_acc = merged["exact_match"].mean() * 100.0

line_config = LineConfig(
    plombs_per_hour=3600.0,
    eur_per_minute=0.1,
    technician_fee=1.0,
    downtime_per_error=2.0,
    batch_size=total_evaluated,
)

score = evaluate(
    errors=errors,
    processing_time_s=PROCESSING_TIME_S,
    config=line_config,
)

# ============================================================
# 6. Report Results
# ============================================================
print("\n" + "=" * 60)
print("                 COMPARISON EVALUATION REPORT")
print("=" * 60)
print(f"Total Matched Samples    : {total_evaluated}")
print(f"Exact Sequence Accuracy  : {exact_acc:.2f}% ({total_evaluated - errors}/{total_evaluated})")
print(f"Total Seal Errors (C9)   : {errors}")
print(f"Assumed Processing Time  : {PROCESSING_TIME_S:.3f} s")
print(f"Time Budget Allowed      : {line_config.time_budget_s:.3f} s")
print("-" * 60)
print(f"Error Penalties Cost     : {score.error_cost:.2f} EUR")
print(f"Overtime Delay           : {score.overtime_s:.3f} s")
print(f"Overtime Penalty Cost    : {score.overtime_cost:.2f} EUR")
print("-" * 60)
print(f"TOTAL LOSS (FINAL SCORE) : {score.total:.2f} EUR  (lower is better)")
print("=" * 60)

print("\nAccuracy by Digit Position (0 to 6):")
for pos, acc in enumerate(per_position_acc):
    print(f"  Position {pos} (10^{6 - pos}s): {acc:.2f}%")
print(f"\nOverall Digit Accuracy   : {total_digit_acc:.2f}%")

# Display incorrect predictions
mismatches = merged[~merged["exact_match"]]
if len(mismatches) > 0:
    print("\n" + "-" * 60)
    print(f"Incorrect Predictions ({len(mismatches)} total):")
    print("-" * 60)
    for _, r in mismatches.iterrows():
        print(f"  {r['filename']:<20} | True: {r['number_true']} | Pred: {r['number_pred']}")
else:
    print("\n🎉 All predictions matched 100%!")
# %%