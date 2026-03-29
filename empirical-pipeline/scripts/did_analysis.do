/*==============================================================================
  DID 分析模板脚本 (Stata 版)
  核心命令：reghdfe (TWFE) + csdid/did_multiplegt (交错DID) + coefplot (图)
  对标：did_analysis.py (Python) 和 did_analysis.R (R)

  使用方式：
    1. 修改下方"配置区"中的 global 宏
    2. 在 Stata 中运行：do "did_analysis.do"
  
  依赖：
    ssc install reghdfe
    ssc install ftools
    ssc install coefplot
    ssc install estout
    ssc install csdid          (交错 DID: Callaway-Sant'Anna)
    ssc install drdid           (csdid 依赖)
    ssc install did_multiplegt  (交错 DID 备选)
    ssc install event_study_stag (可选)
==============================================================================*/

clear all
set more off
set linesize 120
version 16                          // 最低版本要求

********************************************************************************
* 配置区 — 根据实际研究修改以下 global 宏定义
********************************************************************************

* 路径
global DATA_PATH   "data.dta"           // 数据文件路径（.dta 或 .csv）
global OUTPUT_DIR  "output"             // 输出根目录

* 变量名
global Y_VAR       "outcome"            // 因变量
global TREAT_VAR   "treated"            // 处理组虚拟变量（0/1，时不变）
global POST_VAR    "post"               // 政策后虚拟变量（0/1）
global ENTITY_VAR  "entity_id"          // 个体标识
global TIME_VAR    "year"               // 时间变量（数值型）
global EVENT_VAR   "event_time"         // 相对事件时间 (year - treatment_year)
global TREAT_YEAR  "treat_year"         // 首次处理年份（未处理=0，用于交错DID）
global CLUSTER_VAR "entity_id"          // 聚类层级

* 控制变量列表（空格分隔）
global CONTROLS    "control1 control2 control3"

* 事件研究窗口期
global LEADS       4                    // 政策前期数
global LAGS        4                    // 政策后期数
global REF_PERIOD  -1                   // 基准期（事件研究归零的时期）

* 真实政策年份（安慰剂检验使用）
global TRUE_YEAR   2015

********************************************************************************
* 0. 初始化目录
********************************************************************************

cap mkdir "${OUTPUT_DIR}"
cap mkdir "${OUTPUT_DIR}/tables"
cap mkdir "${OUTPUT_DIR}/figures"

* 开启日志
cap log close
log using "${OUTPUT_DIR}/analysis_log.txt", replace text

********************************************************************************
* 1. 数据加载与准备
********************************************************************************

di as text _n "==============================="
di as text   " 1. 数据加载"
di as text   "==============================="

* 根据扩展名选择加载命令
if regexm("${DATA_PATH}", "\.csv$") {
    import delimited "${DATA_PATH}", clear
} 
else {
    use "${DATA_PATH}", clear
}

* 构建交乘项
gen treat_post = ${TREAT_VAR} * ${POST_VAR}
label var treat_post "Treat × Post"

* 基本信息
count
di "样本量: `r(N)'"
sum ${TREAT_VAR}
di "处理组比例: " r(mean)
di "时间范围: " : ${TIME_VAR}

* 设置面板结构
encode ${ENTITY_VAR}, gen(entity_num)    // 若 entity_id 为字符串
xtset entity_num ${TIME_VAR}
xtdescribe

********************************************************************************
* 2. Table 1: 描述性统计
********************************************************************************

di as text _n "==============================="
di as text   " 2. 描述性统计"
di as text   "==============================="

* estpost sum 生成可导出的描述性统计
* 全样本
estpost sum ${Y_VAR} ${CONTROLS}, listwise

* 分组描述性统计（处理组 vs. 控制组）
eststo clear
eststo descr_treat: quietly estpost sum ${Y_VAR} ${CONTROLS} if ${TREAT_VAR} == 1
eststo descr_ctrl:  quietly estpost sum ${Y_VAR} ${CONTROLS} if ${TREAT_VAR} == 0

* esttab 输出（控制台预览）
esttab descr_treat descr_ctrl, ///
    cells("mean(fmt(3)) sd(fmt(3)) count(fmt(0))") ///
    mtitles("Treated" "Control") ///
    title("Table 1: Descriptive Statistics") ///
    noobs

* 均值差 t 检验
foreach var of global CONTROLS {
    ttest `var', by(${TREAT_VAR})
    di "均值差检验 `var': t = " r(t) ", p = " r(p)
}
ttest ${Y_VAR}, by(${TREAT_VAR})

* 保存到 LaTeX
esttab descr_treat descr_ctrl using ///
    "${OUTPUT_DIR}/tables/table1_descriptive.tex", ///
    cells("mean(fmt(3)) sd(fmt(3)) count(fmt(0))") ///
    mtitles("Treated" "Control") ///
    title("Table 1: Descriptive Statistics") ///
    booktabs replace noobs

di "=> Saved: ${OUTPUT_DIR}/tables/table1_descriptive.tex"

********************************************************************************
* 3. 主回归: TWFE DID（reghdfe）
********************************************************************************

di as text _n "==============================="
di as text   " 3. 主回归 TWFE DID"
di as text   "==============================="

* reghdfe 语法：
*   absorb(entity_id year)   — 双向固定效应
*   cluster(entity_id)       — 聚类标准误

eststo clear

* Model (1): 基础 DID（无控制变量）
eststo m1: reghdfe ${Y_VAR} treat_post, ///
    absorb(entity_num ${TIME_VAR}) ///
    cluster(${CLUSTER_VAR}) ///
    nocons

* Model (2): + 控制变量
eststo m2: reghdfe ${Y_VAR} treat_post ${CONTROLS}, ///
    absorb(entity_num ${TIME_VAR}) ///
    cluster(${CLUSTER_VAR}) ///
    nocons

* Model (3): + 个体时间趋势（更严格）
* eststo m3: reghdfe ${Y_VAR} treat_post ${CONTROLS}, ///
*     absorb(entity_num##c.${TIME_VAR} ${TIME_VAR}) ///
*     cluster(${CLUSTER_VAR}) nocons

* 控制台输出
esttab m1 m2, ///
    keep(treat_post) ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    stats(r2 N, labels("R-squared" "Observations")) ///
    mtitles("(1) Baseline" "(2) + Controls") ///
    title("Table 2: Main DID Results") ///
    addnotes("Standard errors clustered at entity level." ///
             "Both entity and year fixed effects included.")

* 保存 LaTeX
esttab m1 m2 using ///
    "${OUTPUT_DIR}/tables/table2_main_results.tex", ///
    keep(treat_post) ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    booktabs ///
    stats(r2 N, labels("\$R^2\$" "Observations")) ///
    mtitles("(1) Baseline" "(2) + Controls") ///
    title("Table 2: Main DID Results") ///
    addnotes("Standard errors clustered at entity level.") ///
    replace

di "=> Saved: ${OUTPUT_DIR}/tables/table2_main_results.tex"

********************************************************************************
* 4. 事件研究（平行趋势检验）
********************************************************************************

di as text _n "==============================="
di as text   " 4. 事件研究"
di as text   "==============================="

* 4a. 使用 reghdfe + forvalues 手动生成事件虚拟变量
* （兼容性最好，Stata 14+ 均可用）

* 生成 lead/lag 虚拟变量
local leads = ${LEADS}
local lags  = ${LAGS}
local ref   = abs(${REF_PERIOD})

* Lead 变量（pre-treatment，负数）
forvalues k = 2/`leads' {
    gen byte D_pre_`k' = (${EVENT_VAR} == -`k') & (${TREAT_VAR} == 1)
    label var D_pre_`k' "t = -`k'"
}

* Lag 变量（post-treatment，非负）
forvalues k = 0/`lags' {
    gen byte D_post_`k' = (${EVENT_VAR} == `k') & (${TREAT_VAR} == 1)
    label var D_post_`k' "t = +`k'"
}
* 基准期 D_pre_1 对应 event_time == -1（REF = -1），不生成（作为参考期）

* 列举所有事件虚拟变量
local evars ""
forvalues k = `leads'(-1)2 {
    local evars "`evars' D_pre_`k'"
}
forvalues k = 0/`lags' {
    local evars "`evars' D_post_`k'"
}

* 回归
reghdfe ${Y_VAR} `evars' ${CONTROLS}, ///
    absorb(entity_num ${TIME_VAR}) ///
    cluster(${CLUSTER_VAR}) ///
    nocons

* 4b. coefplot 绘制事件研究图
coefplot, ///
    keep(`evars') ///
    vertical ///
    order(`evars') ///
    yline(0, lcolor(gray) lpattern(dash)) ///
    xline(`leads', lcolor(red) lpattern(dash))  /* 政策时点竖线 */ ///
    ciopts(recast(rcap) color(%50)) ///
    msymbol(circle) msize(small) mcolor(navy) ///
    lcolor(navy) ///
    xtitle("Relative Time to Policy") ///
    ytitle("Coefficient (95% CI)") ///
    title("Event Study: Parallel Trends Test") ///
    note("Reference period: t = -1. Standard errors clustered at entity level.", ///
         size(vsmall))

graph export "${OUTPUT_DIR}/figures/fig2_event_study.pdf", replace
graph export "${OUTPUT_DIR}/figures/fig2_event_study.png", ///
    width(2700) height(1650) replace

di "=> Saved: ${OUTPUT_DIR}/figures/fig2_event_study.png"

* 4c. 预处理期系数联合显著性检验
local pre_vars ""
forvalues k = 2/`leads' {
    local pre_vars "`pre_vars' D_pre_`k'"
}
testparm `pre_vars'
di "Pre-trend F-test: F(" r(df) "," r(df_r) ") = " r(F) ", p = " r(p)
di "（p > 0.10 表明平行趋势检验通过）"

* 清理临时变量
drop D_pre_* D_post_*

********************************************************************************
* 5. 交错 DID
********************************************************************************

di as text _n "==============================="
di as text   " 5. 交错 DID"
di as text   "==============================="

* ── 方法1：csdid（Callaway & Sant'Anna 2021）──────────────────────────────
* 需要：ssc install csdid drdid
* TREAT_YEAR 变量：首次处理年份（未处理=0）

cap confirm variable ${TREAT_YEAR}
if !_rc {
    di "--- 5a. Callaway & Sant'Anna (2021) via csdid ---"

    * 基本 CS 估计
    csdid ${Y_VAR} ${CONTROLS}, ///
        ivar(entity_num) ///
        time(${TIME_VAR}) ///
        gvar(${TREAT_YEAR}) ///
        method(dripw) ///          // dripw = 双重稳健 IPW；drimp/reg 也可选
        cluster(entity_num)

    * 聚合 ATT（overall）
    estat all                       // 输出 simple/group/calendar/dynamic

    * 事件研究图
    estat event
    csdid_plot, ///
        title("Callaway-Sant'Anna: Dynamic ATT") ///
        ytitle("ATT") xtitle("Relative Time")
    graph export "${OUTPUT_DIR}/figures/fig2b_cs_event_study.png", ///
        width(2700) height(1650) replace

    di "=> Saved: ${OUTPUT_DIR}/figures/fig2b_cs_event_study.png"
}
else {
    di "(跳过 csdid：数据中未找到 ${TREAT_YEAR})"
}

* ── 方法2：did_multiplegt（de Chaisemartin & D'Haultfoeuille 2020）──────
* 适用于连续处理变量或更复杂的交错结构
* cap {
*     did_multiplegt ${Y_VAR} entity_num ${TIME_VAR} ${TREAT_VAR}, ///
*         robust_dynamic dynamic(`lags') placebo(`leads') ///
*         cluster(${CLUSTER_VAR}) graph_effect
*     graph export "${OUTPUT_DIR}/figures/fig2c_dCdH_event.png", ///
*         width(2700) height(1650) replace
* }

********************************************************************************
* 6. 安慰剂检验
********************************************************************************

di as text _n "==============================="
di as text   " 6. 安慰剂检验"
di as text   "==============================="

* 仅使用政策实施前的数据
preserve
keep if ${TIME_VAR} < ${TRUE_YEAR}

local placebo_years "2012 2013"          // 修改为实际安慰剂年份
eststo clear

foreach py of local placebo_years {
    gen byte plac_post = (${TIME_VAR} >= `py')
    gen plac_tp = ${TREAT_VAR} * plac_post

    eststo plac_`py': reghdfe ${Y_VAR} plac_tp ${CONTROLS}, ///
        absorb(entity_num ${TIME_VAR}) ///
        cluster(${CLUSTER_VAR}) nocons

    drop plac_post plac_tp
}

esttab plac_*, ///
    keep(plac_tp) ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles(`placebo_years') ///
    title("Placebo Test: Fake Policy Timing") ///
    addnotes("Pre-policy data only. No coefficient should be significant.")
restore

********************************************************************************
* 7. 稳健性检验
********************************************************************************

di as text _n "==============================="
di as text   " 7. 稳健性检验"
di as text   "==============================="

eststo clear

* (1) 基准（重新估计为参照）
eststo rob0: reghdfe ${Y_VAR} treat_post ${CONTROLS}, ///
    absorb(entity_num ${TIME_VAR}) cluster(${CLUSTER_VAR}) nocons

* (2) 改变聚类层级（如改为省级）
* eststo rob1: reghdfe ${Y_VAR} treat_post ${CONTROLS}, ///
*     absorb(entity_num ${TIME_VAR}) cluster(province_id) nocons

* (3) 剔除政策实施当年
eststo rob2: reghdfe ${Y_VAR} treat_post ${CONTROLS} ///
    if ${TIME_VAR} != ${TRUE_YEAR}, ///
    absorb(entity_num ${TIME_VAR}) cluster(${CLUSTER_VAR}) nocons

* (4) 对数因变量（若原始为水平值）
* gen ln_y = ln(${Y_VAR} + 1)
* eststo rob3: reghdfe ln_y treat_post ${CONTROLS}, ///
*     absorb(entity_num ${TIME_VAR}) cluster(${CLUSTER_VAR}) nocons

* (5) Winsorize 因变量（排除极值影响）
winsor2 ${Y_VAR}, cuts(1 99) gen(y_wins)
eststo rob4: reghdfe y_wins treat_post ${CONTROLS}, ///
    absorb(entity_num ${TIME_VAR}) cluster(${CLUSTER_VAR}) nocons
drop y_wins

esttab rob0 rob2 rob4, ///
    keep(treat_post) ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles("Baseline" "Drop event year" "Winsorized") ///
    title("Table 3: Robustness Checks") ///
    stats(r2 N, labels("R-squared" "Obs"))

esttab rob0 rob2 rob4 using ///
    "${OUTPUT_DIR}/tables/table3_robustness.tex", ///
    keep(treat_post) b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles("Baseline" "Drop event year" "Winsorized") ///
    title("Table 3: Robustness Checks") ///
    booktabs replace

di "=> Saved: ${OUTPUT_DIR}/tables/table3_robustness.tex"

********************************************************************************
* 8. 异质性分析
********************************************************************************

di as text _n "==============================="
di as text   " 8. 异质性分析"
di as text   "==============================="

* 修改 het_var 和分组标签
local het_var  "size_group"           // 异质性变量（0/1 二值）
local label0   "Small"
local label1   "Large"

cap confirm variable `het_var'
if !_rc {
    eststo clear
    eststo het0: reghdfe ${Y_VAR} treat_post ${CONTROLS} if `het_var' == 0, ///
        absorb(entity_num ${TIME_VAR}) cluster(${CLUSTER_VAR}) nocons
    eststo het1: reghdfe ${Y_VAR} treat_post ${CONTROLS} if `het_var' == 1, ///
        absorb(entity_num ${TIME_VAR}) cluster(${CLUSTER_VAR}) nocons

    * 系数图
    coefplot (het0, label("`label0'") mcolor(navy) ciopts(color(navy%40))) ///
             (het1, label("`label1'") mcolor(maroon) ciopts(color(maroon%40))), ///
        keep(treat_post) vertical ///
        yline(0, lcolor(gray) lpattern(dash)) ///
        title("Heterogeneity: by `het_var'") ///
        ytitle("Treat × Post Coefficient") ///
        xtitle("")
    graph export "${OUTPUT_DIR}/figures/fig3_heterogeneity.png", ///
        width(2700) height(1650) replace

    esttab het0 het1, ///
        keep(treat_post) b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
        mtitles("`label0'" "`label1'") ///
        title("Table 4: Heterogeneity Analysis")
}
else {
    di "(跳过异质性分析：变量 `het_var' 不存在)"
}

********************************************************************************
* 结束
********************************************************************************

di as text _n "==============================="
di as text   " Analysis Complete"
di as text   "==============================="
di "输出文件位于: ${OUTPUT_DIR}/"

log close
