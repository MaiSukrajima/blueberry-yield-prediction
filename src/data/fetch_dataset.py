#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 Kaggle 公开数据集获取 Wild Blueberry Yield 原始数据。

数据来源
--------
Kaggle Datasets：``saurabhshahane/wild-blueberry-yield-prediction``
（备用同一份：``shashwatwork/wild-blueberry-yield-prediction-dataset``）

两者内含的 CSV **逐字节相同**，md5 = ``feffa3af1cbb3c8860f4ae34241adc3d``，
即 Mendeley DOI 10.17632/p5hvjzsvn8.1 的 777 行原始学术数据。

关键：Kaggle **数据集**（Datasets）的下载接口无需登录，也不需要 API Token。
这与 Kaggle **竞赛**（Competitions）不同，后者必须认证、且邀请制竞赛无法下载。

用法
----
    python src/data/fetch_dataset.py              # 下载 + 校验 + 落地
    python src/data/fetch_dataset.py --check-only # 只校验本地文件，不联网
    python src/data/fetch_dataset.py --proxy http://127.0.0.1:7897

输出
----
``data/raw/WildBlueberryPollinationSimulationData.csv``
``reports/dataset_fetch.md``
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

PRIMARY = "saurabhshahane/wild-blueberry-yield-prediction"
MIRROR = "shashwatwork/wild-blueberry-yield-prediction-dataset"

CSV_NAME = "WildBlueberryPollinationSimulationData.csv"
CSV_MD5 = "feffa3af1cbb3c8860f4ae34241adc3d"
CSV_BYTES = 85259
EXPECTED_ROWS = 777
EXPECTED_COLS = 18

API = "https://www.kaggle.com/api/v1/datasets/download"


def md5_of(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def build_opener(proxy):
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    op = urllib.request.build_opener(*handlers)
    op.addheaders = [
        ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"),
    ]
    return op


def fetch_zip(slug: str, proxy) -> tuple[bool, bytes | str]:
    try:
        op = build_opener(proxy)
        with op.open(f"{API}/{slug}", timeout=180) as r:
            return True, r.read()
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)
    dst = os.path.join(args.data_dir, CSV_NAME)

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("# 数据获取报告 dataset_fetch")
    say()
    say(f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"- 主源：`https://www.kaggle.com/datasets/{PRIMARY}`")
    say(f"- 备源：`https://www.kaggle.com/datasets/{MIRROR}`")
    say()

    # 1) 取得 CSV
    if args.check_only:
        if not os.path.isfile(dst):
            say(f"**失败**：本地无 `{dst}`，无法只校验。去掉 --check-only 重跑。")
            _write(args.reports, lines)
            return 1
        say("## 1. 获取方式")
        say()
        say("`--check-only`：未联网，仅校验本地文件。")
        data = open(dst, "rb").read()
    else:
        say("## 1. 获取方式")
        say()
        say("Kaggle **Datasets** 下载接口，**无需登录 / 无需 API Token**：")
        say()
        say(f"```\nGET {API}/{{owner}}/{{dataset}}\n```")
        say()
        say("> 注意：Kaggle **Competitions**（竞赛）的数据接口必须认证，"
            "且 `deep-blueberry-challenge-2025` 为邀请制，无法下载。"
            "本题改用公开 Dataset，属同源数据。")
        say()

        data = None
        for slug in (PRIMARY, MIRROR):
            ok, res = fetch_zip(slug, args.proxy)
            if not ok:
                say(f"- `{slug}` → 失败：{res}")
                continue
            blob = res
            try:
                z = zipfile.ZipFile(io.BytesIO(blob))
            except Exception as e:
                say(f"- `{slug}` → 下载成功但不是 zip：{type(e).__name__}")
                continue
            hit = [n for n in z.namelist() if n.lower().endswith(".csv")]
            say(f"- `{slug}` → zip {len(blob):,} 字节，含 {len(hit)} 个 CSV")
            if not hit:
                continue
            cand = z.read(hit[0])
            if md5_of(cand) == CSV_MD5:
                data = cand
                say(f"  - 采用 `{hit[0]}`，md5 `{CSV_MD5}` 与预期一致 ✅")
                break
            say(f"  - `{hit[0]}` md5 `{md5_of(cand)}` 与预期不符，跳过")

        if data is None:
            say()
            say("**失败**：两个源均未取得预期文件。请检查网络或代理。")
            _write(args.reports, lines)
            return 1

        with open(dst, "wb") as f:
            f.write(data)
        say()

    # 2) 校验
    got_md5 = md5_of(data)
    say("## 2. 文件校验")
    say()
    say("| 项 | 预期 | 实际 | 结果 |")
    say("| --- | --- | --- | --- |")
    rows = [
        ("md5", CSV_MD5, got_md5, got_md5 == CSV_MD5),
        ("字节数", f"{CSV_BYTES:,}", f"{len(data):,}", len(data) == CSV_BYTES),
    ]
    for name, exp, act, ok in rows:
        say(f"| {name} | `{exp}` | `{act}` | {'✅' if ok else '❌'} |")

    # 结构校验
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(data))
        shape_ok = df.shape == (EXPECTED_ROWS, EXPECTED_COLS)
        say(f"| 形状 | `{EXPECTED_ROWS}×{EXPECTED_COLS}` | `{df.shape[0]}×{df.shape[1]}` | "
            f"{'✅' if shape_ok else '❌'} |")
        say(f"| 缺失值 | 0 | {int(df.isna().sum().sum())} | "
            f"{'✅' if df.isna().sum().sum() == 0 else '❌'} |")
        say()
        say("列名：")
        say()
        say("```\n" + ", ".join(df.columns) + "\n```")
    except Exception as e:
        say()
        say(f"结构校验失败：{type(e).__name__}: {e}")

    say()
    say("## 3. 复现方式")
    say()
    say("```bash")
    say("python src/data/fetch_dataset.py          # 重新下载并校验")
    say("python src/data/validate_data.py          # 结构校验")
    say("python src/data/make_folds.py             # 生成折划分")
    say("python tests/smoke_test.py                # 冒烟测试")
    say("```")

    _write(args.reports, lines)
    return 0


def _write(reports_dir: str, lines: list) -> None:
    p = os.path.join(reports_dir, "dataset_fetch.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    sys.exit(main())
