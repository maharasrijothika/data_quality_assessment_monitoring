"""cProfile hotspot report for profile_dataframe (200k rows x 20 mixed cols)."""

import cProfile
import pstats
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from app.services.profiling import profile_dataframe  # noqa: E402


def build_frame(n: int = 200_000) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    frame = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "customer_key": rng.randint(1, 20_001, n),
        "product_key": rng.randint(1, 5_001, n),
        "territory_key": rng.randint(1, 11, n),
        "order_line_item": rng.randint(1, 9, n),
        "order_quantity": rng.randint(1, 4, n),
        "unit_price": np.round(rng.uniform(5.0, 500.0, n), 2),
        "discount_pct": np.round(rng.uniform(0.0, 0.3, n), 3),
        "order_date": pd.to_datetime(
            "2020-01-01"
        ) + pd.to_timedelta(rng.randint(0, 1800, n), unit="D"),
        "ship_date": pd.to_datetime(
            "2020-01-01"
        ) + pd.to_timedelta(rng.randint(0, 1800, n), unit="D"),
        "postal_code": rng.choice(
            [f"{i:05d}" for i in range(1, 401)], n
        ),
        "country": rng.choice(
            ["US", "CA", "GB", "DE", "FR", "NL", "AU"], n
        ),
        "channel": rng.choice(["online", "store", "partner"], n),
        "is_returned": rng.choice([True, False], n),
        "review_score": rng.randint(1, 6, n),
        "employee_id": rng.randint(1000, 1101, n),
        "invoice_ref": [
            f"INV-{i % 50000:06d}" for i in range(n)
        ],
        "free_note": [
            "note " + str(i % 700) if i % 3 == 0 else ""
            for i in range(n)
        ],
        "ratio_metric": np.round(rng.uniform(0.1, 9.9, n), 2),
        "year": 2020 + rng.randint(0, 5, n),
    })
    frame.iloc[::97, 4] = np.nan  # order_line_item nulls
    frame.iloc[::131, 6] = np.nan  # unit_price nulls
    return frame


def main() -> None:
    frame = build_frame()
    profiler = cProfile.Profile()
    profiler.enable()
    profile_dataframe(frame)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    print("=== TOP 10 BY CUMULATIVE ===")
    stats.print_stats(10)
    print("=== TOP 10 BY TOTTIME ===")
    stats.sort_stats("tottime")
    stats.print_stats(10)


if __name__ == "__main__":
    main()
