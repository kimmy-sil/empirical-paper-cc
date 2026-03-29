# 自动数据获取 (Data Fetcher)

## 概述

本 skill 提供主流经济/财经数据库的 API 接入方法与 Python 代码模板，涵盖 FRED、World Bank、国家统计局、Census、OECD 等公开 API，以及 CNRDS/CSMAR/CEIC/Wind 等需账号平台的字段说明与下载指引。

**适用场景**：
- 宏观经济变量（GDP、利率、汇率、CPI）获取
- 国家/省级截面或时间序列数据下载
- 企业财务面板数据（CSMAR/CNRDS）整合
- 数据自动更新与增量刷新

---

## 前置条件

```bash
pip install fredapi pandas requests wbgapi world_bank_data censusdata
```

| API | 是否需要注册 | 获取密钥地址 |
|-----|------------|------------|
| FRED | 是（免费） | https://fred.stlouisfed.org/docs/api/api_key.html |
| World Bank | 否 | — |
| 国家统计局 | 否（有限制） | https://data.stats.gov.cn/api.htm |
| Census | 是（免费） | https://api.census.gov/data/key_signup.html |
| OECD | 否 | — |
| CNRDS | 是（付费/机构） | https://www.cnrds.com |
| CSMAR | 是（付费/机构） | https://cn.gtadata.com |
| CEIC | 是（付费/机构） | https://www.ceicdata.com |
| Wind | 是（付费，Wind终端） | — |

---

## 分析步骤

### 1. FRED API（美联储经济数据）

FRED 提供超过 800,000 条美国及全球经济时间序列，包括利率、GDP、通胀、就业等。

```python
# 安装: pip install fredapi
from fredapi import Fred
import pandas as pd

# 初始化（API key 存环境变量，不要硬编码）
import os
fred = Fred(api_key=os.environ.get("FRED_API_KEY"))

# 常用序列代码
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",       # 联邦基金利率（月）
    "us_gdp": "GDP",                     # 美国 GDP（季度，十亿美元）
    "us_cpi": "CPIAUCSL",               # CPI（月，1982-84=100）
    "us_unemployment": "UNRATE",         # 失业率（月，%）
    "us_10y_treasury": "DGS10",         # 10年期国债收益率（日）
    "vix": "VIXCLS",                     # VIX 恐慌指数（日）
    "usd_cny": "DEXCHUS",               # 美元/人民币汇率（日）
    "wti_oil": "DCOILWTICO",            # WTI 油价（日）
    "us_m2": "M2SL",                    # M2 货币供应量（月）
}

def fetch_fred_series(series_dict, start="2000-01-01", end=None):
    """批量下载 FRED 序列，返回合并后的 DataFrame"""
    dfs = []
    for name, code in series_dict.items():
        try:
            s = fred.get_series(code, observation_start=start, observation_end=end)
            s.name = name
            dfs.append(s)
            print(f"✓ {name} ({code}): {len(s)} 条记录")
        except Exception as e:
            print(f"✗ {name} ({code}): {e}")
    return pd.concat(dfs, axis=1)

df_fred = fetch_fred_series(FRED_SERIES, start="2000-01-01")
df_fred.index.name = "date"
df_fred.to_csv("data/raw/fred_macro.csv")
print(df_fred.tail())
```

**搜索 FRED 序列**：
```python
# 按关键词搜索
results = fred.search("China GDP")
print(results[["id", "title", "frequency", "units"]].head(10))
```

---

### 2. World Bank API（世界银行）

提供 200+ 国家、1000+ 指标的年度数据，适合跨国面板研究。

```python
import wbgapi as wb
import pandas as pd

# 常用世行指标代码
WB_INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",       # 人均 GDP（现价美元）
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",         # GDP 增长率（%）
    "trade_openness": "NE.TRD.GNFS.ZS",        # 贸易/GDP（%）
    "fdi_inflow": "BX.KLT.DINV.WD.GD.ZS",     # FDI 流入/GDP（%）
    "population": "SP.POP.TOTL",               # 总人口
    "inflation": "FP.CPI.TOTL.ZG",            # 通货膨胀率（%）
    "gov_expenditure": "GC.XPN.TOTL.GD.ZS",   # 政府支出/GDP（%）
    "education": "SE.XPD.TOTL.GD.ZS",         # 教育支出/GDP（%）
    "internet_users": "IT.NET.USER.ZS",        # 互联网用户（%）
}

def fetch_wb_panel(indicators, countries="all", start_year=2000, end_year=2023):
    """下载世行面板数据"""
    dfs = []
    for name, code in indicators.items():
        try:
            df = wb.data.DataFrame(
                code,
                economy=countries,
                time=range(start_year, end_year + 1),
                labels=True      # 带国家名称
            )
            # 转为长格式
            df_long = df.reset_index().melt(
                id_vars=["economy", "Economy"],
                var_name="year",
                value_name=name
            )
            df_long["year"] = df_long["year"].str.replace("YR", "").astype(int)
            df_long = df_long.rename(columns={"economy": "iso3c", "Economy": "country"})
            dfs.append(df_long.set_index(["iso3c", "country", "year"]))
            print(f"✓ {name} ({code})")
        except Exception as e:
            print(f"✗ {name}: {e}")

    df_all = pd.concat(dfs, axis=1).reset_index()
    return df_all

# 下载 BRICS 国家数据
df_wb = fetch_wb_panel(
    WB_INDICATORS,
    countries=["CHN", "IND", "BRA", "RUS", "ZAF"],
    start_year=2000
)
df_wb.to_csv("data/raw/worldbank_panel.csv", index=False)

# 查找指标代码
wb.search("foreign direct investment")
```

---

### 3. 国家统计局（NBS）开放数据

国家统计局提供有限的公开 API（月度/年度宏观数据）。

```python
import requests
import pandas as pd

# 国家统计局数据 API（公开部分，无需密钥）
NBS_BASE = "https://data.stats.gov.cn/easyquery.htm"

# 常用数据集代码（需从官网确认最新代码）
NBS_DATASETS = {
    "cpi_monthly": "A01",        # 居民消费价格指数
    "gdp_quarterly": "A02",      # GDP 季度数据
    "ppi": "A03",                # 工业生产者出厂价格指数
    "fixed_asset_inv": "A04",    # 固定资产投资
}

def fetch_nbs_data(dataset_code, freq="M", start="201001", end="202312"):
    """
    freq: M=月度, Q=季度, A=年度
    注意：NBS API 限制较多，推荐直接从官网下载 Excel
    备选：使用 akshare 库（pip install akshare）
    """
    params = {
        "m": "QueryData",
        "dbcode": f"h{freq.lower()}",
        "rowcode": "zb",
        "colcode": "sj",
        "wds": f'[{{"wdcode":"zb","valuecode":"{dataset_code}"}},{{"wdcode":"sj","valuecode":"{start}-{end}"}}]',
        "dfwds": "[]",
        "k1": "1234567890"  # 时间戳
    }
    try:
        r = requests.get(NBS_BASE, params=params, timeout=10)
        data = r.json()
        # 解析 data["returndata"]["datanodes"]
        return data
    except Exception as e:
        print(f"NBS API 请求失败: {e}")
        return None

# 推荐替代方案：akshare（更稳定）
# pip install akshare
import akshare as ak

# 宏观数据
df_cpi = ak.macro_china_cpi_monthly()        # CPI 月度
df_pmi = ak.macro_china_pmi()                # PMI
df_m2 = ak.macro_china_money_supply()        # 货币供应量
df_trade = ak.macro_china_trade_balance()    # 贸易差额
```

---

### 4. CNRDS / CSMAR 数据说明

这两个数据库需要机构账号（高校图书馆或购买），提供中国上市公司财务、公司治理、分析师、新闻情绪等数据。

#### CSMAR（国泰安数据库）

**主要数据表及核心字段**：

| 数据表 | 字段示例 |
|--------|---------|
| 股票行情（TRD_Mnth） | `stkcd`（股票代码）, `trdmnt`（年月）, `mretwd`（月收益率）, `mclsprc`（月末收盘价） |
| 财务报表（FS_Comins） | `stkcd`, `accper`（会计期间）, `b001000000`（营业收入）, `b002000000`（净利润） |
| 公司基本信息（TRD_Co） | `stkcd`, `stknme`（公司名称）, `indcd`（行业代码）, `listdt`（上市日期） |
| 分析师预测（I/B/E/S） | `stkcd`, `forecastdate`, `eps_forecast`, `analyst_code` |
| 高管信息（TMT） | `stkcd`, `year`, `ceo_age`, `ceo_tenure`, `board_size` |

**下载步骤**：
1. 登录 https://cn.gtadata.com
2. 数据 → 中国上市公司研究数据库 → 选择模块
3. 设置条件筛选（年份、行业、指标）
4. 导出为 CSV/Excel
5. 本地读取：`pd.read_csv("csmar_data.csv", encoding="gbk")`

**合并键**：`stkcd`（6位股票代码）+ `year`/`accper`

#### CNRDS（中国研究数据服务平台）

**主要数据集**：

| 数据集 | 说明 |
|--------|------|
| 年报文本 | 上市公司年报全文，可做文本分析 |
| 新闻舆情 | 上市公司相关新闻、情绪得分 |
| 专利数据 | 企业专利申请/授权数 |
| 社会责任 | ESG 评级、CSR 报告 |
| 高管背景 | CEO/CFO 教育、工作经历 |

**API 接入**（需机构授权）：
```python
# CNRDS Python API（如有授权）
# 具体接口以 CNRDS 官方文档为准
import requests

headers = {
    "Authorization": "Bearer YOUR_TOKEN",  # 替换为实际 token
    "Content-Type": "application/json"
}

response = requests.post(
    "https://api.cnrds.com/data/query",
    json={
        "dataset": "annual_report_sentiment",
        "filters": {"year": [2010, 2022], "industry": "manufacturing"},
        "fields": ["stkcd", "year", "tone_score", "uncertainty_index"]
    },
    headers=headers
)
```

---

### 5. CEIC / Wind 数据说明

#### CEIC（赛迪信息）

提供中国及全球宏观经济、金融市场、行业数据，通常通过机构订阅访问。

```python
# CEIC API（机构订阅）
# pip install ceic-api-client
from ceic_api_client.apis.series_api import SeriesApi
from ceic_api_client.configuration import Configuration

config = Configuration()
config.username = "YOUR_USERNAME"
config.password = "YOUR_PASSWORD"

api = SeriesApi()
# 查询 GDP 序列
result = api.get_series_by_id("CHINA_GDP_QUARTERLY_ID")
```

**常用 CEIC 指标**（中国部分）：
- 中国各省 GDP（季度/年度）
- 房地产价格指数（70城）
- 工业增加值（月度）
- 城镇固定资产投资（分行业）

#### Wind（万得）

Wind 提供 Python API（需 Wind 终端安装）：

```python
# Wind Python API（需本地安装 Wind 终端）
from WindPy import w
w.start()

# 下载股票数据
data = w.wsd(
    "600000.SH,600036.SH",       # 股票代码
    "open,high,low,close,volume", # 字段
    "2020-01-01",                  # 开始日期
    "2023-12-31",                  # 结束日期
    "PriceAdj=B"                   # 复权方式（前复权）
)
df = pd.DataFrame(data.Data, index=data.Fields, columns=data.Times).T

# 下载宏观数据
macro = w.edb("M0001385", "2000-01-01", "2023-12-31")  # M0001385=CPI

w.stop()
```

---

### 6. Census API（美国人口普查）

```python
import requests
import pandas as pd

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")

def fetch_acs5(variables, state="*", year=2022):
    """
    下载 ACS 5年估计数据
    variables: Census 变量代码列表，如 ["B01003_001E"]（总人口）
    """
    var_str = ",".join(["NAME"] + variables)
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": var_str,
        "for": f"state:{state}",
        "key": CENSUS_API_KEY
    }
    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

# 常用变量代码
CENSUS_VARS = {
    "total_pop": "B01003_001E",        # 总人口
    "median_income": "B19013_001E",    # 家庭收入中位数
    "poverty_rate": "B17001_002E",     # 贫困人口数
    "education_ba": "B15003_022E",     # 本科及以上学历人口
}

df_census = fetch_acs5(list(CENSUS_VARS.values()), year=2022)
df_census.columns = ["name"] + list(CENSUS_VARS.keys()) + ["state_fips"]
```

---

### 7. OECD API

```python
import requests
import pandas as pd

def fetch_oecd(dataset, filter_expr, start_period="2000-Q1", end_period="2023-Q4"):
    """
    dataset: 数据集代码，如 "QNA"（季度国民账户）
    filter_expr: 过滤表达式，如 "USA+CHN+DEU.B1_GE.VOBARSA.Q"
    """
    url = f"https://stats.oecd.org/SDMX-JSON/data/{dataset}/{filter_expr}/all"
    params = {
        "startPeriod": start_period,
        "endPeriod": end_period,
        "dimensionAtObservation": "allDimensions",
        "contentType": "csv"
    }
    r = requests.get(url, params=params, timeout=30)
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    return df

# 示例：下载 GDP 季度数据
df_gdp = fetch_oecd(
    dataset="QNA",
    filter_expr="USA+CHN+DEU+JPN.B1_GE.VOBARSA.Q",  # 实际 GDP，季节调整
    start_period="2000-Q1"
)

# OECD 常用数据集代码
OECD_DATASETS = {
    "QNA": "季度国民账户",
    "MEI": "主要经济指标（月度）",
    "REVENUE_STATISTICS": "税收统计",
    "PENSIONS": "养老金统计",
    "HEALTH_STAT": "卫生统计",
    "TIVA": "增加值贸易",
}
```

---

### 8. 数据缓存与增量更新策略

```python
import os
import hashlib
import json
from datetime import datetime, timedelta
import pandas as pd

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_key(params: dict) -> str:
    """根据请求参数生成缓存键"""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]

def load_cached(cache_id: str, max_age_days: int = 7):
    """加载缓存数据，超过 max_age_days 天则返回 None"""
    meta_path = f"{CACHE_DIR}/{cache_id}_meta.json"
    data_path = f"{CACHE_DIR}/{cache_id}.parquet"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    age = (datetime.now() - datetime.fromisoformat(meta["fetched_at"])).days
    if age > max_age_days:
        print(f"缓存已过期（{age} 天），重新获取...")
        return None
    return pd.read_parquet(data_path)

def save_cache(df: pd.DataFrame, cache_id: str):
    """保存数据到缓存"""
    df.to_parquet(f"{CACHE_DIR}/{cache_id}.parquet", index=False)
    meta = {"fetched_at": datetime.now().isoformat(), "rows": len(df)}
    with open(f"{CACHE_DIR}/{cache_id}_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"缓存已保存: {cache_id} ({len(df)} 行)")

def fetch_with_cache(fetch_fn, params: dict, max_age_days=7):
    """带缓存的数据获取装饰器"""
    cid = cache_key(params)
    cached = load_cached(cid, max_age_days)
    if cached is not None:
        print(f"使用缓存数据 (id={cid})")
        return cached
    df = fetch_fn(**params)
    save_cache(df, cid)
    return df

# 增量更新（适合有最新日期概念的数据）
def incremental_update(existing_path: str, fetch_fn, date_col="date"):
    """
    已有历史数据时，只下载最新数据并追加
    """
    if os.path.exists(existing_path):
        df_existing = pd.read_parquet(existing_path)
        last_date = df_existing[date_col].max()
        print(f"已有数据至 {last_date}，获取增量...")
        df_new = fetch_fn(start=last_date)
        df_new = df_new[df_new[date_col] > last_date]  # 去重
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        print("无历史数据，全量下载...")
        df_updated = fetch_fn()

    df_updated.to_parquet(existing_path, index=False)
    print(f"数据已更新至 {df_updated[date_col].max()}，共 {len(df_updated)} 行")
    return df_updated
```

---

## 检验清单

- [ ] API Key 已通过环境变量注入，未硬编码在代码中
- [ ] 下载数据的时间范围与研究期间吻合
- [ ] 缺失值比例已检查（公开 API 数据经常有早期缺失）
- [ ] 频率对齐（月度 → 年度需聚合，使用均值还是期末值需明确）
- [ ] 单位已统一（注意是否需要 GDP deflator 平减）
- [ ] 缓存文件放入 `.gitignore`，原始数据用 DVC 或 Git LFS 管理

---

## 常见错误提醒

1. **频率混用**：GDP 是季度数据，CPI 是月度数据，直接合并会产生大量 NaN。需先统一频率（resampling）。
2. **汇率通胀未调整**：跨国比较时未用 PPP 或 CPI 平减，名义值比较无意义。
3. **API 限速**：FRED 免费版每天 1000 次请求；World Bank API 无硬限制但建议加 `time.sleep(0.5)`。
4. **CSMAR 编码问题**：中文字符在 Windows 下默认 GBK 编码，读取时需 `encoding="gbk"` 或 `encoding="utf-8-sig"`。
5. **Wind 需本地终端**：WindPy 不支持服务器部署，无法在云端/CI 中使用。需在本地下载后上传。
6. **世行数据年份列格式**：wbgapi 返回的年份列为 `"YR2020"` 格式字符串，需去掉 `"YR"` 前缀转为整数。

---

## 输出规范

- 原始下载数据存放：`data/raw/[source]_[dataset]_[date].csv`（含下载日期）
- 缓存文件：`data/cache/*.parquet`（加入 `.gitignore`）
- 处理后合并数据：`data/processed/merged_panel.csv`
- 数据来源说明写入论文数据节：变量名、数据库名称、时间范围、处理方式
