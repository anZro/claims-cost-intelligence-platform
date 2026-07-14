"""
Synthetic Medicare Part D-style claims data generator.

Generates two CSVs into seeds/:
  - stg_claims_seed.csv   (claim_id, member_id, drug_id, region, submitted_date,
                            adjudicated_date, spend, denied)
  - stg_members_seed.csv  (member_id, region, enrollment_start, enrollment_end)

Deliberately bakes in a spend anomaly: Region 3 gets an elevated per-claim
spend distribution during April 2025, simulating a specialty drug cost spike.
This gives the eventual NL agent ("why did spend jump in Region 3?") a real
signal to find, rather than a flat synthetic baseline with nothing to explain.

Reproducible: fixed random seed, re-running produces identical output.
"""

import csv
import random
from datetime import date, timedelta

random.seed(42)

REGIONS = ["Region 1", "Region 2", "Region 3", "Region 4"]
DRUGS = [f"D{i}" for i in range(1, 13)]  # 12 distinct drug_ids
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 6, 30)
ANOMALY_MONTH = (2025, 4)  # (year, month) — Region 3 spend spike
ANOMALY_REGION = "Region 3"

N_MEMBERS = 500
N_CLAIMS = 2000

# baseline spend distribution (lognormal-ish via random.lognormvariate)
BASELINE_SPEND_MU = 4.5   # ~ median around $90
BASELINE_SPEND_SIGMA = 0.7
ANOMALY_SPEND_MU = 6.5    # ~ median around $665, simulating specialty drug spike
ANOMALY_SPEND_SIGMA = 0.5

DENIAL_RATE = 0.08


def random_date(start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def in_anomaly_window(d: date) -> bool:
    return (d.year, d.month) == ANOMALY_MONTH


def generate_members(n: int) -> list[dict]:
    members = []
    for member_id in range(1, n + 1):
        region = random.choice(REGIONS)
        enrollment_start = START_DATE - timedelta(days=random.randint(0, 365))
        # most members stay enrolled through the window; some churn mid-period
        if random.random() < 0.1:
            enrollment_end = random_date(START_DATE, END_DATE)
        else:
            enrollment_end = END_DATE + timedelta(days=random.randint(0, 180))
        members.append(
            {
                "member_id": member_id,
                "region": region,
                "enrollment_start": enrollment_start.isoformat(),
                "enrollment_end": enrollment_end.isoformat(),
            }
        )
    return members


def generate_claims(n: int, members: list[dict]) -> list[dict]:
    claims = []
    member_by_region: dict[str, list[int]] = {r: [] for r in REGIONS}
    for m in members:
        member_by_region[m["region"]].append(m["member_id"])

    for claim_id in range(1, n + 1):
        submitted_date = random_date(START_DATE, END_DATE)

        # Region 3 is deliberately over-represented during the anomaly month
        # to simulate a genuine utilization + cost spike, not just a price bump.
        if in_anomaly_window(submitted_date) and random.random() < 0.55:
            region = ANOMALY_REGION
        else:
            region = random.choice(REGIONS)

        member_id = random.choice(member_by_region[region])
        drug_id = random.choice(DRUGS)

        adjudication_lag = random.randint(1, 21)
        adjudicated_date = submitted_date + timedelta(days=adjudication_lag)

        is_anomalous_claim = region == ANOMALY_REGION and in_anomaly_window(submitted_date)
        if is_anomalous_claim:
            spend = round(random.lognormvariate(ANOMALY_SPEND_MU, ANOMALY_SPEND_SIGMA), 2)
        else:
            spend = round(random.lognormvariate(BASELINE_SPEND_MU, BASELINE_SPEND_SIGMA), 2)

        denied = random.random() < DENIAL_RATE

        claims.append(
            {
                "claim_id": claim_id,
                "member_id": member_id,
                "drug_id": drug_id,
                "region": region,
                "submitted_date": submitted_date.isoformat(),
                "adjudicated_date": adjudicated_date.isoformat(),
                "spend": spend,
                "denied": str(denied).lower(),
            }
        )
    return claims


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    members = generate_members(N_MEMBERS)
    claims = generate_claims(N_CLAIMS, members)

    write_csv(members, "seeds/stg_members_seed.csv")
    write_csv(claims, "seeds/stg_claims_seed.csv")

    anomaly_claims = [
        c for c in claims
        if c["region"] == ANOMALY_REGION
        and in_anomaly_window(date.fromisoformat(c["submitted_date"]))
    ]
    print(f"Generated {len(members)} members -> seeds/stg_members_seed.csv")
    print(f"Generated {len(claims)} claims -> seeds/stg_claims_seed.csv")
    print(f"Anomaly claims (Region 3, April 2025): {len(anomaly_claims)}")
    if anomaly_claims:
        avg_anomaly_spend = sum(c["spend"] for c in anomaly_claims) / len(anomaly_claims)
        print(f"  avg spend in anomaly window: ${avg_anomaly_spend:,.2f}")
    other_claims = [c for c in claims if c not in anomaly_claims]
    avg_other_spend = sum(c["spend"] for c in other_claims) / len(other_claims)
    print(f"  avg spend elsewhere: ${avg_other_spend:,.2f}")
