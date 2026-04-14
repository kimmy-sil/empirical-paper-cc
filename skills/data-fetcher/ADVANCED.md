# 数据获取助手 — 高级内容

本文件按需加载（ADVANCED.md）。包含：
- R端（fredr / WDI）完整代码
- Census API 详细用法
- Wind 详细说明
- 频率对齐（月→季→年）完整代码
- 跨源数据合并模板
- 增量更新策略

---

## R端（fredr + WDI）

```r
# install.packages(c("fredr","WDI"))
library(fredr)
library(WDI)

# FRED（fredr包）
fredr_set_key(Sys.getenv("FRED_API_KEY"))

fetch_fred_r <- function(series_ids, start_date="2000-01-01") {
  results <- lapply(series_ids, function(id) {
    tryCatch({
      df <- fredr(series_id=id, observation_start=as.Date(start_date))
      df$series_id <- id
      df
    }, error = function(e) {
      message("Error: ", id, " - ", e$message); NULL
    })
  })
  do.call(rbind, Filter(Negate(is.null), results))
}

series_ids <- c("FEDFUNDS","GDP","CPIAUCSL","UNRATE")
df_fred_r <- fetch_fred_r(series_ids)

# World Bank（WDI包）
df_wb_r <- WDI(
  country   = c("CN","IN","BR","RU","ZA"),
  indicator = c(
    gdp_per_capita = "NY.GDP.PCAP.CD",
    gdp_growth     = "NY.GDP.MKTP.KD.ZG",
    trade          = "NE.TRD.GNFS.ZS",
    fdi_inflow     = "BX.KLT.DINV.WD.GD.ZS"
  ),
  start = 2000, end = 2023,
  extra = TRUE  # 加入区域/收入组
)
head(df_wb_r)
WDIsearch("foreign direct investment")
```

---

## Census API（美国人口普查）

```python
import requests, pandas as pd, os

CENSUS_KEY = os.environ.get("CENSUS_API_KEY","")

def fetch_acs5(variables: list, geo: str="state:*",
               year: int=2022) -> pd.DataFrame:
    """
    下载ACS 5年估计数据
    geo: "state:*"（全部州）, "county:*&in=state:06"（加州所有县）
    """
    var_str = ",".join(["NAME"] + variables)
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": var_str, "for": geo}
    if CENSUS_KEY:
        params["key"] = CENSUS_KEY
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data[1:], columns=data[0])

# 常用变量代码
CENSUS_VARS = {
    "B01003_001E": "total_pop",
    "B19013_001E": "median_income",
    "B17001_002E": "poverty_pop",
    "B15003_022E": "education_ba",
    "B25064_001E": "median_gross_rent",
}

df_census = fetch_acs5(list(CENSUS_VARS.keys()), year=2022)
df_census = df_census.rename(columns=CENSUS_VARS)

# census包（另一选择）
# pip install census
from census import Census
c = Census(CENSUS_KEY)
data = c.acs5.get(("NAME","B01003_001E"), {"for":"state:*"})
```

---

## Wind（万得，本地终端）

```python
# 需本地安装Wind终端（机构订阅）
# pip install WindPy（仅在安装Wind终端的机器上有效）
# ⚠️ Wind不支持服务器/云端/Docker部署，只能本地使用

try:
    from WindPy import w
    w.start()

    # 股票数据（wsd = Wind Sequence Data）
    data = w.wsd(
        "600000.SH,600036.SH",
        "open,high,low,close,volume",
        "2020-01-01", "2023-12-31",
        "PriceAdj=B"                   # 前复权
    )
    df_wind = pd.DataFrame(
        data.Data, index=data.Fields, columns=data.Times
    ).T

    # 宏观数据库（edb = Economic Database）
    # M0001385=CPI, M0001227=GDP, M0043121=M2
    macro = w.edb("M0001385", "2000-01-01", "2023-12-31")

    # 横截面数据（wss = Wind Snapshot Data）
    snapshot = w.wss(
        "600000.SH,600036.SH",
        "pe_ttm,pb_lf,ps_ttm",
        "tradeDate=20231231"
    )

    w.stop()

except ImportError:
    print("Wind终端未安装（或非本地机器）")
    print("处理方式：在本地Wind终端下载数据 → 上传文件")
    print("  df_wind = pd.read_excel('data/raw/wind_data.xlsx')")
```

---

## 频率对齐（月→季→年）

```python
import pandas as pd

def align_frequency(df: pd.DataFrame, date_col: str, value_cols: list,
                    target_freq: str, agg_method: str="mean") -> pd.DataFrame:
    """
    频率对齐（降采样）
    target_freq: 'QE'=季度末, 'YE'=年末
    agg_method: 'mean'（价格/利率）, 'last'（存量/期末值）, 'sum'（流量）

    判断原则：
    - 流量变量（贸易差额/新增贷款）→ sum
    - 价格/利率（CPI/利率）→ mean
    - 存量变量（M2/外汇储备）→ last（期末值）
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    agg_func_map = {"mean":"mean", "last":"last", "sum":"sum"}
    df_res = df[value_cols].resample(target_freq).agg(agg_func_map[agg_method])
    return df_res.reset_index()

# 月度CPI → 季度均值
df_cpi_q = align_frequency(df_cpi_monthly, "date", ["cpi"], "QE", "mean")

# 季度GDP → 年度（求和）
df_gdp_a = align_frequency(df_gdp_quarterly, "date", ["gdp"], "YE", "sum")

# 日利率 → 月度（期末值）
df_rate_m = align_frequency(df_rate_daily, "date", ["rate"], "ME", "last")
```

---

## 跨源数据合并模板

```python
def merge_multi_source(sources: dict, merge_on: list=["year"],
                       how: str="outer") -> pd.DataFrame:
    """
    合并多个数据源（年度/季度面板）
    sources: {"fred":df_fred, "wb":df_wb, "akshare":df_cn}

    ⚠️ 合并后的缺失诊断 + 质量检查 → data-cleaning skill EDA 节
    """
    df_all = None
    for name, df in sources.items():
        if df_all is None:
            df_all = df.copy()
        else:
            n_before = len(df_all)
            df_all = pd.merge(df_all, df, on=merge_on, how=how)
            print(f"合并{name}: {n_before} → {len(df_all)} 行")
    return df_all

df_final = merge_multi_source({
    "fred":     df_fred,
    "worldbank":df_wb,
    "akshare":  df_cn,
}, merge_on=["year","country"])
df_final.to_parquet("data/processed/merged_panel.parquet", index=False)
```

---

## 增量更新策略

```python
import os
import pandas as pd

def incremental_update(existing_path: str, fetch_fn,
                       date_col: str="date") -> pd.DataFrame:
    """
    已有历史数据时，只下载最新数据并追加
    """
    if os.path.exists(existing_path):
        df_existing = (pd.read_parquet(existing_path)
                       if existing_path.endswith(".parquet")
                       else pd.read_csv(existing_path))
        df_existing[date_col] = pd.to_datetime(df_existing[date_col])
        last_date = df_existing[date_col].max()
        print(f"已有数据至 {last_date.date()}，获取增量...")
        df_new = fetch_fn(start=last_date.strftime("%Y-%m-%d"))
        df_new[date_col] = pd.to_datetime(df_new[date_col])
        df_new = df_new[df_new[date_col] > last_date]
        if len(df_new) == 0:
            print("无新增数据")
            return df_existing
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"新增 {len(df_new)} 条，总计 {len(df_updated)} 条")
    else:
        print("无历史数据，全量下载...")
        df_updated = fetch_fn()

    if existing_path.endswith(".parquet"):
        df_updated.to_parquet(existing_path, index=False)
    else:
        df_updated.to_csv(existing_path, index=False)
    return df_updated
```
