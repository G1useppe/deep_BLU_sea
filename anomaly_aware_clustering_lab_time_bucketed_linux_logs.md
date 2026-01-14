# Anomaly-Aware Clustering with Time-Bucketed Linux Logs

**Duration:** ~3 hours  
**Audience:** IT / Cyber / Security professionals with basic Python familiarity  
**Goal:** Understand how time-bucketed feature engineering changes the behaviour of k-means vs DBSCAN when anomalies are present.

---

## Learning Objectives
By the end of this lab, learners will be able to:

- Generate synthetic Linux-like log data with hidden structure
- Engineer **time-bucketed behavioural features** from logs
- Explain why anomalies distort centroid-based clustering
- Compare k-means and DBSCAN on the same dataset
- Reason about clustering outputs *without ground-truth labels*

---

## Conceptual Setup

We simulate a small Linux fleet with different **latent system roles**:

- Workstations
- Web servers
- Batch / cron-heavy systems

We then inject **time-localised anomalies**, such as:

- SSH brute-force bursts
- Sudden privilege escalation spikes

The key idea is:

> **Normal systems are consistent over time. Anomalies are bursty.**

Time-bucketed features allow us to capture this distinction.

---

## Part 1 — Environment Setup (15 minutes)

```python
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

Set seeds for reproducibility:

```python
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

---

## Part 2 — Synthetic Log Generation (30 minutes)

### System Profiles

```python
SERVICES = ["sshd", "cron", "sudo", "kernel", "nginx"]
DAYS = 7

PROFILES = {
    "workstation": {
        "sshd": 20, "sudo": 5, "cron": 3, "kernel": 50, "nginx": 0
    },
    "web": {
        "sshd": 10, "sudo": 2, "cron": 5, "kernel": 40, "nginx": 300
    },
    "batch": {
        "sshd": 3, "sudo": 1, "cron": 50, "kernel": 60, "nginx": 0
    }
}
```

### Log Generator

```python
def generate_logs(host, profile, anomaly=False):
    logs = []
    start = datetime.now() - timedelta(days=DAYS)

    for day in range(DAYS):
        day_start = start + timedelta(days=day)
        for service, avg in PROFILES[profile].items():
            count = np.random.poisson(avg)
            for _ in range(count):
                logs.append({
                    "timestamp": day_start + timedelta(minutes=random.randint(0, 1439)),
                    "host": host,
                    "service": service
                })

        # Inject time-bucketed anomaly (single day burst)
        if anomaly and day == DAYS - 2:
            for _ in range(200):  # SSH brute force burst
                logs.append({
                    "timestamp": day_start + timedelta(minutes=random.randint(0, 60)),
                    "host": host,
                    "service": "sshd"
                })
            for _ in range(40):  # sudo spike
                logs.append({
                    "timestamp": day_start + timedelta(minutes=random.randint(0, 60)),
                    "host": host,
                    "service": "sudo"
                })

    return logs
```

---

## Part 3 — Build the Fleet Dataset (15 minutes)

```python
logs = []
host_profiles = {}

for i in range(60):
    if i < 20:
        profile = "workstation"
    elif i < 40:
        profile = "web"
    else:
        profile = "batch"

    anomaly = i >= 55  # last 5 hosts are anomalous
    host = f"host_{i}"

    host_profiles[host] = profile
    logs.extend(generate_logs(host, profile, anomaly))

logs_df = pd.DataFrame(logs)
```

---

## Part 4 — Time-Bucketed Feature Engineering (40 minutes)

We aggregate logs into **daily buckets per host**.

```python
logs_df["date"] = logs_df["timestamp"].dt.date

bucketed = (
    logs_df
    .groupby(["host", "date", "service"])
    .size()
    .unstack(fill_value=0)
)
```

### Aggregate Over Time (Behavioural Stability)

```python
features = bucketed.groupby("host").agg([
    "mean",
    "std",
    "max"
])

features.columns = [f"{svc}_{stat}" for svc, stat in features.columns]
```

### Derived Burstiness Features

```python
features["ssh_burst_ratio"] = features["sshd_max"] / (features["sshd_mean"] + 1)
features["sudo_burst_ratio"] = features["sudo_max"] / (features["sudo_mean"] + 1)
```

---

## Part 5 — Scaling and Dimensionality Reduction (20 minutes)

```python
X = features.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

---

## Part 6 — Clustering (30 minutes)

### k-means

```python
kmeans = KMeans(n_clusters=3, random_state=RANDOM_SEED)
kmeans_labels = kmeans.fit_predict(X_scaled)

print("K-means silhouette:", silhouette_score(X_scaled, kmeans_labels))
```

### DBSCAN

```python
dbscan = DBSCAN(eps=1.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

print(pd.Series(dbscan_labels).value_counts())
```

---

## Part 7 — Interpretation and Discussion (30 minutes)

### Expected Observations

- k-means forces anomalous hosts into clusters
- Centroids shift due to burst-heavy features
- DBSCAN labels bursty systems as `-1`
- Time-based features dominate over raw volume

### Key Questions for Learners

- Which features drive separation?
- What happens if you remove `*_max` features?
- How sensitive is DBSCAN to `eps`?
- Are all `-1` points truly malicious?

---

## Part 8 — Optional Extensions (Remaining Time)

- Replace daily buckets with **hourly** buckets
- Visualise PCA scatter with cluster labels
- Export logs as syslog-style text files
- Map features to Splunk SPL:

```spl
index=syslog | stats count by host service date
```

---

## Instructor Notes (Hidden from Learners)

- True anomalies: `host_55` → `host_59`
- DBSCAN should isolate most anomalies if scaled correctly
- k-means silhouette may look reasonable despite semantic failure

---

## Key Takeaway

> **Unsupervised learning finds structure — not truth.**  
> Feature engineering determines what “structure” even means.

Time-awareness is often more important than algorithm choice.

