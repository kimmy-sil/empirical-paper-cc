# 数据获取助手

## 概述

主流经济/财经数据库API接入、代码模板、手动下载指引。  
**语言**：Python + R（NO Stata）

**高级内容**：R端（fredr/WDI）完整代码、Census详细用法、CNRDS/Wind说明、缓存策略、频率对齐 → 见 ADVANCED.md

---

## Step 0: 环境检测

```python
import importlib.util, os, requests

def check_network(test_url: str="https://api.worldbank.org", timeout: int=5) -> bool:
    try:
        requests.head(test_url, timeout=timeout)
        print("✓ 网络可用")
        return True
    except requests.RequestException:
        print("✗ 网络不可用 → 请手动上传数据文件（见场景C）")
        return False

def get_api_key(env_var: str, source: str, signup_url: str="") -> str:
    key = os.environ.get(env_var, "")
    if key:
        print(f"✓ {source} API Key已加载（{env_var}）")
    else:
        print(f"⚠️ {source} API Key未设置（export {env_var}=your_key）")
        if signup_url:
            print(f"   申请: {signup_url}")
    return key

HAS_NETWORK = check_network()
FRED_KEY    = get_api_key("FRED_API_KEY",   "FRED",   "https://fred.stlouisfed.org/docs/api/api_key.html")
CENSUS_KEY  = get_api_key("CENSUS_API_KEY", "Census", "https://api.census.gov/data/key_signup.html")
```

---

## 三种使用场景

| 场景 | 条件 | 行为 |
|------|------|------|
| **A（有API Key）** | Key已配置 + 网络可用 | 完整可运行脚本 |
| **B（无Key）** | 网络可用但无Key | 申请链接 + Key占位符模板 |
| **C（无网络/手动）** | 离线/机构数据 | 手动下载指引 + 清洗代码 |

---

## 国际数据源（公开免费）

### FRED API（推荐，代码质量高）

```python
# pip install fredapi
from fredapi import Fred
import pandas as pd, os

fred = Fred(api_key=os.environ.get("FRED_API_KEY"))

FRED_SERIES = {
    "fed_funds_rate":  "FEDFUNDS",    # 联邦基金利率（月）
    "us_gdp":          "GDP",         # 美国GDP（季度）
    "us_cpi":          "CPIAUCSL",    # CPI（月）
    "us_unemployment": "UNRATE",      # 失业率（月）
    "us_10y_treasury": "DGS10",       # 10年期国债收益率（日）
    "vix":             "VIXCLS",      # VIX（日）
    "usd_cny":         "DEXCHUS",     # 美元/人民币（日）
    "wti_oil":         "DCOILWTICO",  # WTI油价（日）
    "us_m2":           "M2SL",        # M2（月）
}

def fetch_fred_series(series_dict: dict, start: str="2000-01-01",
                      end: str=None) -> pd.DataFrame:
    """批量下载FRED序列，返回宽格式DataFrame"""
    dfs = []
    for name, code in series_dict.items():
        try:
            s = fred.get_series(code, observation_start=start, observation_end=end)
            s.name = name
            dfs.append(s)
            print(f"✓ {name} ({code}): {len(s)} 条")
        except Exception as e:
            print(f"✗ {name} ({code}): {e}")
    df = pd.concat(dfs, axis=1)
    df.index.name = "date"
    df.to_parquet("data/raw/fred_macro.parquet")
    return df

df_fred = fetch_fred_series(FRED_SERIES, start="2000-01-01")

# 搜索序列
results = fred.search("China GDP")
print(results[["id","title","frequency","units"]].head(10))
```

---

### World Bank API

**FIX**：`labels=True`是`wb.data.DataFrame`的有效参数（NOT `wb.search`的参数）。

```python
# pip install wbgapi
import wbgapi as wb
import pandas as pd

WB_INDICATORS = {
    "gdp_per_capita":  "NY.GDP.PCAP.CD",
    "gdp_growth":      "NY.GDP.MKTP.KD.ZG",
    "trade_openness":  "NE.TRD.GNFS.ZS",
    "fdi_inflow":      "BX.KLT.DINV.WD.GD.ZS",
    "population":      "SP.POP.TOTL",
    "inflation":       "FP.CPI.TOTL.ZG",
    "internet_users":  "IT.NET.USER.ZS",
}

def fetch_wb_panel(indicators: dict, countries: list=None,
                   start_year: int=2000, end_year: int=2023) -> pd.DataFrame:
    """
    下载世行面板数据，返回长格式DataFrame
    FIX: labels=True是wb.data.DataFrame的参数，不是wb.search的参数
    FIX: numericTimeKeys=True → 时间列为整数年份（非"YR2020"格式字符串）
    """
    if countries is None:
        countries = "all"
    dfs = []
    for name, code in indicators.items():
        try:
            df = wb.data.DataFrame(
                code,
                economy=countries,
                time=range(start_year, end_year+1),
                labels=True,           # 添加economy名称列
                numericTimeKeys=True   # 时间列为整数，非"YR2020"
            )
            df_long = df.stack().reset_index()
            df_long.columns = ["iso3c","country","year",name]
            dfs.append(df_long.set_index(["iso3c","country","year"]))
            print(f"✓ {name} ({code})")
        except Exception as e:
            print(f"✗ {name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=1).reset_index()

df_wb = fetch_wb_panel(WB_INDICATORS,
                       countries=["CHN","IND","BRA","RUS","ZAF"])
df_wb.to_parquet("data/raw/worldbank_panel.parquet", index=False)

# 搜索指标
wb.search("foreign direct investment")
```

---

### OECD（重写！旧API已废弃）

**重要**：`stats.oecd.org/SDMX-JSON/`旧端点已废弃，使用新端点`sdmx.oecd.org`。

```python
import requests, pandas as pd
from io import StringIO

OECD_NEW_BASE = "https://sdmx.oecd.org/public/rest/data"

def fetch_oecd_new(dataset_id: str, filter_expr: str="all",
                   start_period: str="2000", end_period: str="2023") -> pd.DataFrame:
    """
    使用新OECD SDMX REST API（2024年起）
    旧端点 stats.oecd.org/SDMX-JSON 已废弃！

    dataset_id常用值:
    - "GDP"              → 国民账户主要指标
    - "ULC_EEQ"          → 单位劳动力成本
    - "HEALTH_STAT"      → 卫生统计
    - "REVENUE_STATISTICS" → 税收统计
    - "TIVA"             → 增加值贸易
    """
    url = f"{OECD_NEW_BASE}/{dataset_id}/{filter_expr}"
    params = {
        "startPeriod": start_period,
        "endPeriod":   end_period,
        "format":      "csvfilewithlabels"
    }
    headers = {"Accept": "application/vnd.sdmx.data+csv; charset=utf-8"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        print(f"✓ OECD {dataset_id}: {len(df)} 行")
        return df
    except requests.HTTPError as e:
        print(f"✗ OECD API错误: {e}")
        print("  备选：访问 https://data.oecd.org 手动下载CSV")
        return pd.DataFrame()

# 示例
df_oecd = fetch_oecd_new(
    dataset_id   = "GDP",
    filter_expr  = "USA+CHN+DEU+JPN+GBR",
    start_period = "2000"
)
```

---

## 中国数据源

### akshare（首选！比NBS API稳定得多）

```python
# pip install akshare
import akshare as ak

# 宏观数据
df_cpi   = ak.macro_china_cpi_monthly()       # CPI月度
df_ppi   = ak.macro_china_ppi_monthly()        # PPI月度
df_pmi   = ak.macro_china_pmi()                # PMI
df_m2    = ak.macro_china_money_supply()       # M0/M1/M2
df_trade = ak.macro_china_trade_balance()      # 贸易差额（月度）
df_gdp   = ak.macro_china_gdp()               # GDP季度

# 省级GDP
df_province = ak.macro_china_gdp_province()

# A股日行情
df_stock = ak.stock_zh_a_hist(
    symbol="600000", period="daily",
    start_date="20200101", end_date="20231231",
    adjust="qfq"   # 前复权
)

# 利率
df_lpr   = ak.macro_china_lpr()               # LPR贷款基准利率
df_shibor = ak.rate_interbank(
    market="上海银行间同业拆放利率", symbol="Shibor人民币"
)

# 清洗模板
def clean_akshare(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df = df[[date_col, value_col]].copy()
    df[date_col]  = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna().sort_values(date_col).reset_index(drop=True)

df_cpi_clean = clean_akshare(df_cpi, date_col="月份", value_col="今值")
```

---

### NBS API（备选，不稳定）

```python
# ⚠️ 警告: easyquery.htm是非官方接口，随时可能失效
# 频繁请求可能被封IP，优先使用akshare
import requests, time

NBS_BASE = "https://data.stats.gov.cn/easyquery.htm"

def fetch_nbs_data(dataset_code: str, freq: str="M",
                   start: str="201001", end: str="202312") -> dict | None:
    """NBS非官方API（备选）。失败时自动提示akshare替代方案"""
    params = {
        "m": "QueryData", "dbcode": f"h{freq.lower()}",
        "rowcode": "zb", "colcode": "sj",
        "wds":   f'[{{"wdcode":"zb","valuecode":"{dataset_code}"}},'
                 f'{{"wdcode":"sj","valuecode":"{start}-{end}"}}]',
        "dfwds": "[]", "k1": str(int(time.time()*1000))
    }
    try:
        r = requests.get(NBS_BASE, params=params, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        return r.json().get("returndata", {})
    except Exception as e:
        print(f"✗ NBS API失败: {e}")
        print("  → 改用akshare: ak.macro_china_cpi_monthly()")
        return None
```

---

### CSMAR（国泰安，机构账号）

需高校图书馆或购买订阅。

**主要数据表**：

| 数据表 | 合并键 | 核心字段 |
|--------|--------|---------|
| 股票行情 TRD_Mnth | `stkcd` + `trdmnt` | `mretwd`月收益率 |
| 财务报表 FS_Comins | `stkcd` + `accper` | `b001000000`营业收入 |
| 公司信息 TRD_Co | `stkcd` | `indcd`行业代码 |
| 高管信息 TMT | `stkcd` + `year` | `ceo_age`, `board_size` |

**下载 + 读取（场景C）**：
```python
# 登录 https://cn.gtadata.com → 导出CSV
df_csmar = pd.read_csv(
    "data/raw/csmar_financial.csv",
    encoding="gbk",      # Windows下CSMAR默认GBK
    low_memory=False
)
df_csmar["stkcd"] = df_csmar["stkcd"].astype(str).str.zfill(6)  # 6位补零
df_csmar["year"]  = pd.to_datetime(df_csmar["accper"]).dt.year
```

---

### CNRDS（机构账号）

主要数据集：年报文本、新闻舆情情绪得分、专利数据、ESG评级、高管背景。

```python
# ⚠️ 以下为示例/伪代码，实际接口以CNRDS官方文档为准
# CNRDS无标准公开API，实际主要通过网页下载
# https://www.cnrds.com

def query_cnrds_example(dataset: str, token: str) -> None:
    """CNRDS API示例（伪代码，非官方确认接口）"""
    import requests
    headers = {"Authorization": f"Bearer {token}"}
    # 实际端点和参数格式以官方文档为准
    print(f"请访问 https://www.cnrds.com 获取官方API文档")
    print(f"推荐方式：网页下载 → pd.read_excel() 或 pd.read_csv()")
```

---

## 缓存策略

```python
import os, hashlib, json
from datetime import datetime
import pandas as pd

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_key(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]

def load_cached(cache_id: str, max_age_days: int=7) -> pd.DataFrame | None:
    """parquet优先，csv作fallback"""
    for ext in [".parquet",".csv"]:
        data_path = f"{CACHE_DIR}/{cache_id}{ext}"
        meta_path = f"{CACHE_DIR}/{cache_id}_meta.json"
        if os.path.exists(data_path) and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            age = (datetime.now() - datetime.fromisoformat(meta["fetched_at"])).days
            if age <= max_age_days:
                return (pd.read_parquet(data_path) if ext==".parquet"
                        else pd.read_csv(data_path))
            print(f"缓存过期（{age}天）→ 重新获取")
            return None
    return None

def save_cache(df: pd.DataFrame, cache_id: str) -> None:
    """保存缓存（parquet优先，失败时fallback csv）"""
    try:
        df.to_parquet(f"{CACHE_DIR}/{cache_id}.parquet", index=False)
    except Exception:
        df.to_csv(f"{CACHE_DIR}/{cache_id}.csv", index=False)
    with open(f"{CACHE_DIR}/{cache_id}_meta.json","w") as f:
        json.dump({"fetched_at": datetime.now().isoformat(), "rows":len(df)}, f)

def fetch_with_cache(fetch_fn, params: dict, max_age_days: int=7) -> pd.DataFrame:
    cid = cache_key(params)
    cached = load_cached(cid, max_age_days)
    if cached is not None:
        return cached
    df = fetch_fn(**params)
    save_cache(df, cid)
    return df
```

---

## 检验清单 + 常见错误

### 检验清单

- [ ] **环境检测**：已运行Step 0（网络 + API Key）
- [ ] **API Key**：通过环境变量注入，未硬编码
- [ ] **时间范围**：下载覆盖研究期间（含足够前期观测）
- [ ] **频率对齐**：多源合并前已统一频率（见ADVANCED.md）
- [ ] **单位统一**：是否需要CPI deflator平减 / PPP调整
- [ ] **OECD端点**：使用新端点`sdmx.oecd.org`，非废弃的`stats.oecd.org`
- [ ] **wbgapi labels**：`labels=True`在`wb.data.DataFrame`中，非`wb.search`
- [ ] **akshare优先**：中国数据用akshare，NBS API仅作备选
- [ ] **缓存**：数据已缓存到`data/cache/*.parquet`

### 常见错误

1. **OECD旧API**：`stats.oecd.org/SDMX-JSON/`已废弃，返回404/503。用新端点`sdmx.oecd.org`。
2. **wbgapi labels参数**：`wb.data.DataFrame(..., labels=True)`有效；`wb.search(..., labels=True)`无此参数。
3. **NBS频繁请求被封**：`easyquery.htm`是非官方接口，优先akshare。
4. **频率混用**：GDP季度、CPI月度，直接merge产生大量NaN。先对齐频率再合并（见ADVANCED.md）。
5. **CSMAR编码**：Windows默认GBK，服务器上读取需`encoding="gbk"`。
6. **Wind云端不可用**：WindPy需本地终端，无法在服务器/云端使用。
7. **wbgapi年份格式**：不加`numericTimeKeys=True`时年份列为`"YR2020"`字符串，需`.str.replace("YR","")`。

## 输出规范

| 文件 | 路径 |
|------|------|
| 原始下载数据 | `data/raw/{source}_{dataset}_{YYYYMMDD}.parquet` |
| 缓存文件 | `data/cache/*.parquet`（加入.gitignore） |
| 合并后面板 | `data/processed/merged_panel.parquet` |

**论文数据节说明**：变量名、数据库名称、时间范围、频率、处理方式（平减/频率对齐方法）
