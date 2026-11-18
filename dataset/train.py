"""시뮬레이터 데이터로 에이전트 모델 4개를 학습하고 규칙 baseline·다른 후보와 비교한다.

    python -m dataset.train                 # 기본: 16가구 × 14일
    python -m dataset.train --households 24 --days 21 --save   # data/에 전체 표 저장(git 제외)

결과
- models/{cleaning,noise,temperature,lighting}.joblib  (채택 모델)
- models/model_card.json                               (후보별 지표, 데이터 크기, 시드)
- dataset/samples/*.jsonl                               (표마다 앞 200행, 형식 확인용)
분할: 가구 단위(앞 75% 학습, 뒤 25% 평가) — 같은 가구가 학습·평가에 섞이지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline

from dataset.simulate import simulate
from orchestrator import agents as A
from orchestrator import features as F

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.1"

SPECS = {
    # agent: (특징 이름, 분류?, 규칙 baseline, {후보 이름: 생성 함수}, 추천 후보)
    "cleaning": (F.CLEAN_FEATURES, False, A.rule_clean, {
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0,
                                                     monotonic_cst=[1, 1] + [0] * (len(F.CLEAN_FEATURES) - 2)),
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=24.0), Ridge(alpha=1.0)),
    }, "hgb"),
    "noise": (F.NOISE_FEATURES, True, A.rule_noise, {
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                                        LogisticRegression(C=1.0, max_iter=2000)),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
    }, "logreg"),
    "temperature": (F.TEMP_FEATURES, False, A.rule_vote, {
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), Ridge(alpha=1.0)),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0),
    }, "ridge"),
    "lighting": (F.LIGHT_FEATURES, True, A.rule_sleep, {
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                                        LogisticRegression(C=1.0, max_iter=2000)),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
    }, "logreg"),
}


def metrics(y, pred, is_clf: bool) -> dict:
    if is_clf:
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        return {"auc": round(float(roc_auc_score(y, p)), 4), "log_loss": round(float(log_loss(y, p)), 4),
                "brier": round(float(brier_score_loss(y, p)), 4)}
    return {"mae": round(float(mean_absolute_error(y, pred)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", type=int, default=16)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", action="store_true", help="전체 표를 data/에 jsonl로 저장")
    args = ap.parse_args()

    t0 = time.time()
    data = simulate(args.households, args.days, args.seed)
    print(f"시뮬레이션 {time.time() - t0:.0f}초: " + ", ".join(f"{k} {len(v)}행" for k, v in data.items()))
    split = int(args.households * 0.75)

    samples = ROOT / "dataset" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    import platform

    import sklearn

    card = {"version": VERSION, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "env": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__, "joblib": joblib.__version__},
            "data": {"households": args.households, "days": args.days, "seed": args.seed,
                     "split": f"가구 0~{split - 1} 학습, {split}~{args.households - 1} 평가"},
            "agents": {}}

    for agent, (names, is_clf, rule, candidates, pick) in SPECS.items():
        rows = data[agent]
        with (samples / f"{agent}.jsonl").open("w", encoding="utf-8") as f:
            for hid, row, y in rows[:200]:
                f.write(json.dumps({"household": hid, **{k: (None if isinstance(v, float) and math.isnan(v) else v)
                                                         for k, v in row.items()}, "label": y}) + "\n")
        if args.save:
            (ROOT / "data").mkdir(exist_ok=True)
            with (ROOT / "data" / f"{agent}.jsonl").open("w", encoding="utf-8") as f:
                for hid, row, y in rows:
                    f.write(json.dumps({"household": hid, **row, "label": y}, default=str) + "\n")

        X = F.to_matrix([r for _, r, _ in rows], names)
        y = np.array([lab for _, _, lab in rows], dtype=float)
        tr = np.array([hid < split for hid, _, _ in rows])
        report = {"rows_train": int(tr.sum()), "rows_test": int((~tr).sum()),
                  "rule": metrics(y[~tr], [rule(r) for (_, r, _), m in zip(rows, ~tr) if m], is_clf)}
        fitted = {}
        for name, make in candidates.items():
            model = make().fit(X[tr], y[tr])
            pred = model.predict_proba(X[~tr])[:, 1] if is_clf else model.predict(X[~tr])
            report[name] = metrics(y[~tr], pred, is_clf)
            fitted[name] = model
        report["chosen"] = pick
        card["agents"][agent] = report
        joblib.dump({"model": fitted[pick], "kind": pick, "version": VERSION, "features": names},
                    ROOT / "models" / f"{agent}.joblib", compress=3)
        print(f"{agent:12s} 규칙 {report['rule']}  " + "  ".join(f"{n} {report[n]}" for n in candidates))

    (ROOT / "models" / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료 {time.time() - t0:.0f}초 → models/")


if __name__ == "__main__":
    main()
