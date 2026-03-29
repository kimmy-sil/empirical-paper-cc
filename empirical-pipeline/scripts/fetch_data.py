#!/usr/bin/env python3
"""
自动数据获取脚本
==============================================================================
支持数据源：
  1. FRED API     — 美联储经济数据（宏观时间序列）
  2. World Bank   — 世界银行跨国面板数据
  3. OECD         — OECD 统计数据
  4. US Census    — 美国人口普查 API
  5. 通用 CSV/Excel 下载

数据缓存：本地存一份，避免重复请求 API。

中国数据说明：CSMAR/CNRDS/Wind 需机构账号（见文件末尾）。

依赖安装：
  pip install fredapi wbgapi requests pandas openpyxl tqdm
  pip install census   # 如需美国 Census API
==============================================================================
"""

import os
import time
import hashlib
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ==============================================================================
# 全局配置
# ==============================================================================

# 缓存目录（本地存储 API 返回数据，避免重复请求）
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 原始数据输出目录
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 缓存有效期（天数），超过后强制刷新
CACHE_TTL_DAYS = 7

# ==============================================================================
# 通用缓存工具
# ==============================================================================

def _cache_key(source: str, params: dict) -> str:
    """根据数据源和参数生成缓存文件名"""
    raw = source + json.dumps(params, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=CACHE_TTL_DAYS)


def save_cache(df: pd.DataFrame, key: str) -> None:
    df.to_parquet(_cache_path(key), index=True)


def load_cache(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if _is_cache_valid(path):
        print(f"  [cache] Loading from cache: {path.name}")
        return pd.read_parquet(path)
    return None


# ==============================================================================
# 1. FRED API（美联储经济数据）
# ==============================================================================
#
# 获取 API Key：https://fred.stlouisfed.org/docs/api/api_key.html（免费）
# pip install fredapi

def fetch_fred(
    series_ids: Union[str, List[str]],
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    从 FRED 获取经济时间序列数据。

    参数：
        series_ids  : 单个序列 ID 或 ID 列表，如 "GDP" 或 ["GDP", "UNRATE"]
        start_date  : 开始日期，格式 "YYYY-MM-DD"
        end_date    : 结束日期，默认今天
        api_key     : FRED API Key（或设环境变量 FRED_API_KEY）
        force_refresh: 忽略缓存强制刷新

    常用序列 ID 示例：
        GDP         — 美国实际 GDP（季度）
        UNRATE      — 美国失业率（月度）
        CPIAUCSL    — 美国 CPI（月度）
        FEDFUNDS    — 联邦基金利率（月度）
        DGS10       — 10年期国债收益率
        DEXCHUS     — 人民币/美元汇率
        INDPRO      — 工业生产指数
    
    更多序列：https://fred.stlouisfed.org/
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("请运行：pip install fredapi")

    if isinstance(series_ids, str):
        series_ids = [series_ids]

    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "需要 FRED API Key。请在 https://fred.stlouisfed.org 注册后，"
            "设置环境变量 FRED_API_KEY='your_key' 或传入 api_key 参数。"
        )

    params = {"series": series_ids, "start": start_date, "end": end_date or "today"}
    key = _cache_key("fred", params)

    if not force_refresh:
        cached = load_cache(key)
        if cached is not None:
            return cached

    print(f"[FRED] Fetching: {series_ids} ({start_date} → {end_date or 'today'})")
    fred = Fred(api_key=api_key)
    frames = {}
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start_date,
                                observation_end=end_date)
            frames[sid] = s
            time.sleep(0.3)  # 礼貌性延迟
        except Exception as e:
            print(f"  [WARN] Failed to fetch {sid}: {e}")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    save_cache(df, key)
    out = RAW_DIR / f"fred_{'_'.join(series_ids[:3])}.csv"
    df.to_csv(out)
    print(f"  => Saved: {out} ({len(df)} rows)")
    return df


# ==============================================================================
# 2. World Bank API（跨国宏观面板数据）
# ==============================================================================
#
# 无需 API Key，免费使用
# pip install wbgapi

def fetch_worldbank(
    indicators: Union[str, List[str]],
    countries: Union[str, List[str]] = "all",
    start_year: int = 2000,
    end_year: Optional[int] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    从 World Bank Open Data 获取跨国指标数据。

    参数：
        indicators  : WB 指标代码（一个或多个）
        countries   : 国家代码列表（ISO 3166-1 alpha-2/3），或 "all"
        start_year  : 起始年份
        end_year    : 截止年份，默认最新

    常用指标代码：
        NY.GDP.MKTP.CD          — GDP (现价美元)
        NY.GDP.MKTP.KD.ZG       — GDP 增长率 (%)
        NY.GDP.PCAP.CD          — 人均 GDP (美元)
        SL.UEM.TOTL.ZS          — 失业率 (% 劳动力)
        NE.TRD.GNFS.ZS          — 贸易开放度 (% GDP)
        SP.POP.TOTL             — 总人口
        SE.ADT.LITR.ZS          — 成人识字率
        FP.CPI.TOTL.ZG          — CPI 通货膨胀率 (%)
        GC.DOD.TOTL.GD.ZS       — 政府债务 (% GDP)
        BX.KLT.DINV.WD.GD.ZS   — 外商直接投资净流入 (% GDP)
        IT.NET.USER.ZS          — 互联网用户比例 (%)
        EG.USE.ELEC.KH.PC       — 人均用电量
    
    更多指标：https://data.worldbank.org/indicator
    """
    try:
        import wbgapi as wb
    except ImportError:
        raise ImportError("请运行：pip install wbgapi")

    if isinstance(indicators, str):
        indicators = [indicators]
    end_year = end_year or datetime.now().year

    params = {"indicators": indicators, "countries": countries,
              "start": start_year, "end": end_year}
    key = _cache_key("wb", params)

    if not force_refresh:
        cached = load_cache(key)
        if cached is not None:
            return cached

    print(f"[World Bank] Fetching: {indicators}")

    frames = []
    for ind in indicators:
        try:
            df_ind = wb.data.DataFrame(
                ind,
                economy=countries,
                time=range(start_year, end_year + 1),
                skipBlanks=True,
                labels=True
            ).reset_index()
            # 宽转长
            df_ind = df_ind.melt(
                id_vars=["economy", "Country"],
                var_name="year",
                value_name=ind
            )
            df_ind["year"] = df_ind["year"].str.replace("YR", "").astype(int)
            frames.append(df_ind.set_index(["economy", "year"]))
            time.sleep(0.5)
        except Exception as e:
            print(f"  [WARN] Failed to fetch {ind}: {e}")

    if not frames:
        raise ValueError("所有指标获取失败，请检查指标代码。")

    df = pd.concat(frames, axis=1).reset_index()
    df = df.loc[:, ~df.columns.duplicated()]  # 去除重复列

    save_cache(df, key)
    out = RAW_DIR / f"wb_{'_'.join(indicators[:2])}.csv"
    df.to_csv(out, index=False)
    print(f"  => Saved: {out} ({len(df)} rows, {df['economy'].nunique()} countries)")
    return df


# ==============================================================================
# 3. OECD API（OECD 统计数据）
# ==============================================================================
#
# 无需 API Key，基于 SDMX REST API
# 官方文档：https://data.oecd.org/api/sdmx-json-documentation/

def fetch_oecd(
    dataset_id: str,
    filter_expr: str = "all",
    start_period: Optional[str] = None,
    end_period: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    从 OECD.Stat 获取统计数据（SDMX-JSON）。

    参数：
        dataset_id  : OECD 数据集代码
        filter_expr : 过滤表达式（OECD SDMX 格式）
        start_period: 开始期（如 "2000"）
        end_period  : 结束期

    常用数据集代码：
        GDP_GROWTH      — GDP 增长率
        MEI_PRICES      — 主要经济指标：价格
        LABOUR_FORCE    — 劳动力统计
        HEALTH_STAT     — 健康统计
        PISA            — PISA 教育评估
        FDI_FLOWS       — 外商直接投资流量
        OECD_ENV_DATA   — 环境数据
    
    数据集浏览：https://stats.oecd.org/

    示例：
        fetch_oecd("GDP_GROWTH", "AUS+CAN+GBR+USA.T.GDP_GROWTH..A")
    """
    base_url = "https://stats.oecd.org/SDMX-JSON/data"
    params_str = f"{dataset_id}/{filter_expr}"
    if start_period:
        params_str += f"?startPeriod={start_period}"
    if end_period:
        sep_char = "&" if "?" in params_str else "?"
        params_str += f"{sep_char}endPeriod={end_period}"

    url = f"{base_url}/{params_str}"
    cache_key = _cache_key("oecd", {"url": url})

    if not force_refresh:
        cached = load_cache(cache_key)
        if cached is not None:
            return cached

    print(f"[OECD] Fetching: {url[:80]}...")
    headers = {"Accept": "application/json"}

    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"OECD API 请求失败: {e}")

    # 解析 SDMX-JSON 格式
    try:
        dataset = data["dataSets"][0]
        structure = data["structure"]

        dims = structure["dimensions"]["observation"]
        dim_names = [d["id"] for d in dims]
        dim_values = {
            d["id"]: {str(i): v["id"] for i, v in enumerate(d["values"])}
            for d in dims
        }

        rows = []
        for key, obs in dataset["observations"].items():
            parts = key.split(":")
            row = {}
            for i, dname in enumerate(dim_names):
                row[dname] = dim_values[dname].get(parts[i], parts[i])
            row["value"] = obs[0] if obs else None
            rows.append(row)

        df = pd.DataFrame(rows)
    except (KeyError, IndexError) as e:
        raise ValueError(f"OECD SDMX-JSON 解析失败: {e}\n返回内容片段: {str(data)[:200]}")

    save_cache(df, cache_key)
    out = RAW_DIR / f"oecd_{dataset_id.lower()}.csv"
    df.to_csv(out, index=False)
    print(f"  => Saved: {out} ({len(df)} rows)")
    return df


# ==============================================================================
# 4. US Census API
# ==============================================================================
#
# 获取 API Key：https://api.census.gov/data/key_signup.html（免费）
# pip install census us

def fetch_census_acs(
    variables: List[str],
    year: int = 2022,
    geo: str = "state",
    api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    从美国人口普查 ACS (American Community Survey) 获取数据。

    参数：
        variables  : Census 变量代码列表
        year       : 数据年份（ACS 支持 2009–2022）
        geo        : 地理层级："state" / "county" / "tract"
        api_key    : Census API Key（或环境变量 CENSUS_API_KEY）

    常用 ACS 5-Year 变量（前缀 B/C 开头）：
        B01003_001E  — 总人口
        B19013_001E  — 家庭中位收入
        B15003_022E  — 学士学位持有者
        B23025_005E  — 失业人口
        B25064_001E  — 中位租金
    
    变量查询：https://api.census.gov/data/{year}/acs/acs5/variables.html
    """
    try:
        from census import Census
    except ImportError:
        raise ImportError("请运行：pip install census us")

    api_key = api_key or os.environ.get("CENSUS_API_KEY")
    if not api_key:
        raise ValueError(
            "需要 Census API Key。注册地址：https://api.census.gov/data/key_signup.html\n"
            "设置环境变量：CENSUS_API_KEY='your_key'"
        )

    params = {"variables": variables, "year": year, "geo": geo}
    key = _cache_key("census", params)

    if not force_refresh:
        cached = load_cache(key)
        if cached is not None:
            return cached

    print(f"[Census] Fetching ACS {year}: {variables}")
    c = Census(api_key, year=year)

    if geo == "state":
        data = c.acs5.get(variables, {"for": "state:*"})
    elif geo == "county":
        data = c.acs5.get(variables, {"for": "county:*", "in": "state:*"})
    else:
        raise ValueError(f"暂不支持 geo='{geo}'，请用 'state' 或 'county'")

    df = pd.DataFrame(data)
    for v in variables:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")

    save_cache(df, key)
    out = RAW_DIR / f"census_acs{year}_{geo}.csv"
    df.to_csv(out, index=False)
    print(f"  => Saved: {out} ({len(df)} rows)")
    return df


# ==============================================================================
# 5. 通用 CSV/Excel 下载函数
# ==============================================================================

def download_file(
    url: str,
    filename: Optional[str] = None,
    sheet_name: Union[int, str] = 0,
    encoding: str = "utf-8",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    通用文件下载函数：支持 CSV / Excel (.xlsx .xls)。
    自动识别格式，本地缓存，避免重复下载。

    参数：
        url        : 文件直链 URL
        filename   : 保存文件名（默认从 URL 提取）
        sheet_name : Excel 工作表（CSV 忽略）
        encoding   : CSV 编码
    
    示例：
        download_file("https://data.worldbank.org/.../GDP.csv")
        download_file("https://example.com/data.xlsx", sheet_name="Sheet1")
    """
    if not filename:
        filename = url.split("/")[-1].split("?")[0] or "downloaded_data.csv"

    local_path = RAW_DIR / filename
    key = _cache_key("file", {"url": url})

    if not force_refresh and _is_cache_valid(_cache_path(key)):
        print(f"  [cache] File already cached: {local_path}")
        if local_path.exists():
            return _read_file(local_path, sheet_name, encoding)

    print(f"[Download] {url}")
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 Chrome/120.0 Safari/537.36")
    }

    try:
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"下载失败: {e}")

    # 从 Content-Type 判断格式
    content_type = resp.headers.get("Content-Type", "")
    if not local_path.suffix:
        if "excel" in content_type or "spreadsheet" in content_type:
            local_path = local_path.with_suffix(".xlsx")
        else:
            local_path = local_path.with_suffix(".csv")

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    # 写缓存标记
    pd.DataFrame().to_parquet(_cache_path(key))

    df = _read_file(local_path, sheet_name, encoding)
    print(f"  => Saved: {local_path} ({len(df)} rows × {len(df.columns)} cols)")
    return df


def _read_file(
    path: Path,
    sheet_name: Union[int, str] = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        # 尝试多种编码
        for enc in [encoding, "utf-8", "gbk", "gb2312", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except (UnicodeDecodeError, ValueError):
                continue
        raise ValueError(f"无法读取文件: {path}")


# ==============================================================================
# 6. 示例：获取各国 GDP、失业率、贸易数据
# ==============================================================================

def example_cross_country() -> pd.DataFrame:
    """
    示例：下载多国宏观面板数据并合并。
    数据源：World Bank Open Data
    """
    print("\n" + "=" * 60)
    print("示例：跨国宏观面板数据（World Bank）")
    print("=" * 60)

    # 目标国家（G20 + 主要发展中国家）
    countries = [
        "CN", "US", "JP", "DE", "GB", "FR", "IN", "BR", "RU", "ZA",
        "KR", "CA", "AU", "MX", "ID", "SA", "TR", "AR", "IT", "ES"
    ]

    # 获取主要宏观指标
    df = fetch_worldbank(
        indicators=[
            "NY.GDP.MKTP.KD.ZG",    # GDP 增长率
            "NY.GDP.PCAP.CD",        # 人均 GDP
            "SL.UEM.TOTL.ZS",        # 失业率
            "NE.TRD.GNFS.ZS",        # 贸易开放度
            "FP.CPI.TOTL.ZG",        # 通货膨胀率
            "BX.KLT.DINV.WD.GD.ZS", # FDI 净流入
            "IT.NET.USER.ZS",        # 互联网用户
        ],
        countries=countries,
        start_year=2000,
        end_year=2023,
    )

    # 基本清洗
    df = df.rename(columns={
        "economy": "iso_code",
        "NY.GDP.MKTP.KD.ZG":    "gdp_growth",
        "NY.GDP.PCAP.CD":        "gdp_per_capita",
        "SL.UEM.TOTL.ZS":        "unemployment_rate",
        "NE.TRD.GNFS.ZS":        "trade_openness",
        "FP.CPI.TOTL.ZG":        "inflation",
        "BX.KLT.DINV.WD.GD.ZS": "fdi_inflow_pct_gdp",
        "IT.NET.USER.ZS":        "internet_users_pct",
    })

    print(f"\n数据形状: {df.shape}")
    print(f"覆盖国家: {df['iso_code'].nunique()}")
    print(f"时间范围: {df['year'].min()} — {df['year'].max()}")
    print("\n前5行预览:")
    print(df.head())

    # 保存清洗后的数据
    out = RAW_DIR / "cross_country_macro.csv"
    df.to_csv(out, index=False)
    print(f"\n=> Saved: {out}")

    return df


def example_fred_us() -> pd.DataFrame:
    """示例：获取美国主要宏观时间序列"""
    print("\n" + "=" * 60)
    print("示例：美国宏观时间序列（FRED）")
    print("=" * 60)

    # 注意：需设置 FRED_API_KEY 环境变量
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("[SKIP] 跳过 FRED 示例：未设置 FRED_API_KEY 环境变量")
        print("  设置方式：export FRED_API_KEY='your_key_here'")
        return pd.DataFrame()

    df = fetch_fred(
        series_ids=["GDPC1", "UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10"],
        start_date="2000-01-01",
        api_key=api_key,
    )

    df.columns = ["real_gdp", "unemployment", "cpi", "fed_funds_rate", "treasury_10y"]
    df = df.resample("Q").mean()  # 统一为季度

    print(f"\n数据形状: {df.shape}")
    print(df.tail())
    return df


# ==============================================================================
# 中国数据说明
# ==============================================================================
"""
中国企业/金融数据主要数据库均需机构账号，无法直接通过 API 自动下载。
以下为主要数据库的字段说明和手动下载指引。

──────────────────────────────────────────────────────────────────────────────
A. CSMAR（中国股票市场与会计研究数据库）
   官网：https://cn.gtadata.com/
   覆盖：A股上市公司财务报表、股价、公司治理、分析师、并购
   
   手动下载步骤：
   1. 登录机构账号（高校 VPN 内访问）
   2. 选择数据集（如"财务报表" → "资产负债表"）
   3. 选择年份范围（如 2000–2023）和字段
   4. 导出 Excel/CSV 到本地
   
   常用表与关键字段：
   - 资产负债表（FS_Combas）：
       Stkcd  — 股票代码
       Accper — 会计期间（YYYY-MM-DD）
       A001000000 — 货币资金
       A002000000 — 应收账款
       A010000000 — 总资产
   - 利润表（FS_Comins）：
       B001000000 — 营业收入
       B002000000 — 营业成本
       B011000000 — 净利润
   - 公司基本信息（TRD_Co）：
       Stkcd — 股票代码, Stknme — 公司名称, Indcd — 行业代码
       Listdt — 上市日期, Ipodt — IPO 日期

──────────────────────────────────────────────────────────────────────────────
B. CNRDS（中国研究数据服务平台）
   官网：https://www.cnrds.com/
   覆盖：上市公司文本分析、专利、企业关联、高管、媒体报道
   
   手动下载步骤：同 CSMAR。
   
   常用数据集：
   - 上市公司年报文本 — 管理层讨论与分析(MD&A)
   - 企业专利数据 — 发明专利/实用新型/外观设计
   - 媒体报道情感分析 — 正/负面新闻词频

──────────────────────────────────────────────────────────────────────────────
C. Wind（万得金融终端）
   官网：https://www.wind.com.cn/
   覆盖：宏观经济、股市、债市、期货、基金、另类数据
   
   Python API（需安装 Wind Python API，限 Wind 终端用户）：
   
   from WindPy import w
   w.start()
   # 获取 A股日收益率
   data = w.wsd("000001.SZ", "close,volume", "2020-01-01", "2023-12-31", "")
   df = pd.DataFrame(data.Data, index=data.Fields, columns=data.Times).T
   
   # 宏观数据示例
   gdp = w.edb("M0001385", "2010-01-01", "2023-12-31")  # 中国 GDP 当季同比
   
   Wind 代码体系：
   - A股：000001.SZ（深市）/ 600000.SH（沪市）
   - 宏观：M 开头（Wind 宏观数据库编码）

──────────────────────────────────────────────────────────────────────────────
D. CNKI（中国知网）
   官网：https://www.cnki.net/
   用途：中文学术文献检索与下载（文献综述）
   
   批量下载注意事项：
   - 每次最多导出 500 条
   - NoteExpress/Endnote 格式可直接导入文献管理工具
   - 避免使用机器人批量下载（违反使用条款）

──────────────────────────────────────────────────────────────────────────────
E. 国家统计局数据
   官网：https://data.stats.gov.cn/
   覆盖：宏观经济年度/季度数据、省级数据、行业数据
   
   # 无官方 Python API，可用 stats_can 思路手动抓取（有风险）
   # 建议：直接在网站手动选择指标下载 Excel
   
   常用指标：
   - 地区生产总值（分省市）
   - 城镇登记失业率
   - 工业增加值
   - 固定资产投资

──────────────────────────────────────────────────────────────────────────────
F. 中国工业企业数据库（Annual Survey of Industrial Firms）
   来源：国家统计局，学界通过 NBER 或合作机构获取
   时间：1998–2013（较新年份访问受限）
   字段说明：
   - orgnum  — 企业唯一标识符
   - year    — 年份
   - output  — 工业总产值
   - employ  — 从业人数年均
   - wage    — 工资总额
   - asset   — 资产总计
   - indcode — 行业代码（GB/T 4754）
   
   注：该数据库存在重复、合并问题，清洗参考：
   Brandt et al. (2012, Journal of Development Economics)
"""


# ==============================================================================
# 缓存管理工具
# ==============================================================================

def list_cache() -> pd.DataFrame:
    """列出所有缓存文件及其大小和修改时间"""
    records = []
    for f in CACHE_DIR.glob("*.parquet"):
        stat = f.stat()
        records.append({
            "file": f.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "expired": not _is_cache_valid(f),
        })
    return pd.DataFrame(records).sort_values("modified", ascending=False)


def clear_cache(expired_only: bool = True) -> None:
    """清除缓存文件"""
    n = 0
    for f in CACHE_DIR.glob("*.parquet"):
        if expired_only and _is_cache_valid(f):
            continue
        f.unlink()
        n += 1
    print(f"Cleared {n} cache file(s).")


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据获取脚本 — fetch_data.py")
    print(f"缓存目录: {CACHE_DIR.resolve()}")
    print(f"数据目录: {RAW_DIR.resolve()}")
    print("=" * 60)

    # ── 运行示例（取消注释以执行）──────────────────────────────────────

    # 示例1: 跨国宏观面板数据（无需 API Key）
    # df_wb = example_cross_country()

    # 示例2: 美国时间序列（需要 FRED API Key）
    # df_fred = example_fred_us()

    # 示例3: 单个 World Bank 指标
    # df_gdp = fetch_worldbank(
    #     indicators=["NY.GDP.PCAP.CD", "SL.UEM.TOTL.ZS"],
    #     countries=["CN", "US", "JP", "DE", "IN"],
    #     start_year=2005, end_year=2023
    # )
    # print(df_gdp.head(10))

    # 示例4: 直接下载 CSV 文件
    # df_csv = download_file(
    #     url="https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv",
    #     filename="world_gdp_raw.csv"
    # )

    # 示例5: 查看缓存状态
    # print(list_cache())

    # 示例6: FRED（需设置 FRED_API_KEY）
    # df_us = fetch_fred(
    #     series_ids=["UNRATE", "GDPC1"],
    #     start_date="2010-01-01",
    # )

    print("\n使用方式：取消注释上方对应示例，或直接调用各 fetch_* 函数。")
    print("参考文档：函数 docstring（help(fetch_worldbank) 等）")
