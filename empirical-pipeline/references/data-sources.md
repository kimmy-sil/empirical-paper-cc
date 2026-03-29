# 常用公开数据源索引

## 中国数据

### 宏观/地区

| 数据源 | 网址 | 内容 | 格式 |
|-------|------|------|------|
| 中国统计年鉴 | stats.gov.cn | 国民经济、社会发展 | Excel/PDF |
| 中国城市统计年鉴 | stats.gov.cn | 地级市面板 | Excel |
| 北大数字普惠金融指数 | idf.pku.edu.cn | 数字金融（2011-） | Excel |
| 中国区域创新能力评价报告 | 各年出版 | 地区创新指标 | PDF→Excel |
| 国家知识产权局 | cnipa.gov.cn | 专利数据 | API |
| 中国数字经济发展指数 | 各研究机构 | 数字经济综合指数 | Excel |
| 电子政务发展指数 | egov.nsa.gov.cn | 政府数字化 | Excel |

### 微观/企业

| 数据源 | 网址 | 内容 | 费用 |
|-------|------|------|------|
| CSMAR | csmar.com | 上市公司财务、治理、专利 | 商业 |
| CNRDS | cnrds.com | 公司研究、ESG、创新 | 商业 |
| Wind | wind.com.cn | 金融终端数据 | 商业 |
| 中国研究数据服务平台 | cnsda.org | 调查数据 | 免费/申请 |
| 中国工业企业数据库 | 国家统计局 | 规上工业企业（1998-2013） | 申请 |
| 中国海关数据库 | customs.gov.cn | 进出口贸易 | 申请 |
| 天眼查/企查查 | tianyancha.com | 企业工商信息 | 商业 |

### 调查/个体

| 数据源 | 网址 | 内容 | 费用 |
|-------|------|------|------|
| CFPS (中国家庭追踪) | isss.pku.edu.cn/cfps | 家庭、教育、收入 | 免费申请 |
| CHNS (中国健康营养) | cpc.unc.edu/projects/china | 健康、营养、劳动 | 免费 |
| CGSS (中国综合社会调查) | cnsda.org | 社会态度、行为 | 免费 |
| CHARLS (中国健康与养老) | charls.pku.edu.cn | 中老年健康 | 免费 |
| CHIP (中国收入分配) | ciid.bnu.edu.cn | 收入分配 | 免费申请 |

### 政策/制度

| 数据源 | 类型 | 说明 |
|-------|------|------|
| 北大法宝 | 法律法规 | 各级政府政策文本 |
| 中国政府网 | 政策文件 | 国务院发文 |
| 各省市政府公报 | 地方政策 | 条例实施时间（DID所需） |
| 国家企业信用信息公示 | 行政处罚 | 监管执行数据 |

---

## 国际数据

### 宏观

| 数据源 | 网址 | 内容 |
|-------|------|------|
| World Bank Open Data | data.worldbank.org | 发展指标 |
| FRED | fred.stlouisfed.org | 美国宏观经济 |
| OECD iLibrary | oecd-ilibrary.org | OECD国家比较 |
| Penn World Table | rug.nl/ggdc/productivity/pwt | 国际GDP比较 |
| IMF Data | data.imf.org | 国际金融统计 |

### 微观/调查

| 数据源 | 网址 | 内容 |
|-------|------|------|
| Census ACS PUMS | census.gov/programs-surveys/acs | 美国微观 |
| EU-SILC | ec.europa.eu/eurostat | 欧洲收入调查 |
| DHS Program | dhsprogram.com | 发展中国家健康 |
| LSMS | worldbank.org/lsms | 生活水平调查 |

### 学术/文本

| 数据源 | 网址 | 内容 |
|-------|------|------|
| OpenAlex | openalex.org | 学术文献元数据 |
| Semantic Scholar | semanticscholar.org | 论文引用网络 |
| UCDP | ucdp.uu.se | 冲突事件 |
| GDELT | gdeltproject.org | 全球事件数据 |

---

## 数字经济/数据治理相关

以下数据源特别适合数据治理、AI 政策、数字经济研究：

| 主题 | 可用数据 | 研究设计思路 |
|------|---------|------------|
| 数据安全法(2021.9) | 上市公司信息披露+CSMAR | DID（法律实施前后） |
| 个人信息保护法(2021.11) | 企业隐私投入+天眼查处罚 | DID/事件研究 |
| 数据要素市场化 | 各地数据交易所设立时间 | 交错DID |
| 智慧城市试点 | 住建部试点名单+城市面板 | DID（批次处理） |
| "东数西算" | 枢纽节点城市+经济指标 | DID/合成控制 |
| AI 监管 | 算法备案+深度合成管理办法 | 事件研究 |
| 平台经济反垄断 | 处罚事件+平台市值/行为 | 事件研究 |
| 数字人民币试点 | 试点城市扩展+消费数据 | 交错DID |
