"""
Valuation analysis tools for MCP server.
Provides comprehensive valuation metrics including P/E, P/B, P/S, PEG, and DCF analysis.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource, NoDataFoundError, LoginError, DataSourceError
from src.formatting.markdown_formatter import format_df_to_markdown

logger = logging.getLogger(__name__)


def _calculate_dcf_value(cash_flows: List[float], terminal_growth_rate: float = 0.025, 
                        discount_rate: float = 0.10, forecast_years: int = 5) -> Dict[str, float]:
    """
    Calculate DCF (Discounted Cash Flow) valuation.
    
    Args:
        cash_flows: Historical cash flows for extrapolation
        terminal_growth_rate: Long-term growth rate assumption (default 2.5%)
        discount_rate: Discount rate/WACC (default 10%)
        forecast_years: Forecast period in years (default 5)
    
    Returns:
        Dictionary with DCF components and results
    """
    if len(cash_flows) < 2:
        return {"error": "Insufficient cash flow data for DCF calculation"}
    
    # Calculate average growth rate from historical data
    cash_flows = [cf for cf in cash_flows if cf > 0]  # Filter positive cash flows
    if len(cash_flows) < 2:
        return {"error": "Insufficient positive cash flow data"}
    
    # Calculate compound annual growth rate (CAGR)
    historical_growth = (cash_flows[-1] / cash_flows[0]) ** (1 / (len(cash_flows) - 1)) - 1
    
    # Use conservative growth rate
    forecast_growth_rate = min(historical_growth, 0.15)  # Cap at 15%
    
    # Project future cash flows
    projected_cash_flows = []
    last_cf = cash_flows[-1]
    
    for year in range(1, forecast_years + 1):
        next_cf = last_cf * (1 + forecast_growth_rate) ** year
        projected_cash_flows.append(next_cf)
    
    # Calculate terminal value
    terminal_cf = projected_cash_flows[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_cf / (discount_rate - terminal_growth_rate)
    
    # Discount all cash flows to present value
    pv_cash_flows = []
    for i, cf in enumerate(projected_cash_flows, 1):
        pv = cf / (1 + discount_rate) ** i
        pv_cash_flows.append(pv)
    
    pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years
    
    enterprise_value = sum(pv_cash_flows) + pv_terminal
    
    return {
        "enterprise_value": enterprise_value,
        "pv_cash_flows": sum(pv_cash_flows),
        "pv_terminal_value": pv_terminal,
        "terminal_value": terminal_value,
        "forecast_growth_rate": forecast_growth_rate,
        "historical_growth": historical_growth,
        "projected_cash_flows": projected_cash_flows
    }


def register_valuation_analysis_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    Register valuation analysis tools with the MCP app.
    
    Args:
        app: The FastMCP app instance
        active_data_source: The active financial data source
    """

    @app.tool()
    def get_valuation_metrics(
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        """
        获取股票的估值指标数据，包括市盈率(P/E)、市净率(P/B)、市销率(P/S)等的实时数据和历史趋势。

        Args:
            code: 股票代码，如'sh.600000'
            start_date: 开始日期，格式'YYYY-MM-DD'，默认为最近1年
            end_date: 结束日期，格式'YYYY-MM-DD'，默认为当前日期

        Returns:
            包含各种估值指标的Markdown表格和趋势分析
        """
        logger.info(f"Tool 'get_valuation_metrics' called for {code}")
        
        try:
            # 设置默认日期范围
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            # 获取包含估值指标的历史数据
            df = active_data_source.get_historical_k_data(
                code=code,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjust_flag="3",
                fields=["date", "code", "close", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
            )
            
            if df.empty:
                return f"Error: No valuation data found for {code}"
            
            # 数据预处理
            df['date'] = pd.to_datetime(df['date'])
            numeric_cols = ['close', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 过滤掉无效数据
            df = df.dropna(subset=['close'])
            
            # 获取股票基本信息
            basic_info = active_data_source.get_stock_basic_info(code=code)
            stock_name = basic_info['code_name'].values[0] if not basic_info.empty else code
            
            # 生成分析报告
            report = f"# {stock_name} ({code}) 估值指标分析\n\n"
            report += f"**分析期间**: {start_date} 至 {end_date}\n"
            report += f"**数据点数**: {len(df)} 个交易日\n\n"
            
            # 当前估值指标
            latest_data = df.iloc[-1]
            report += "## 最新估值指标\n"
            report += f"- **收盘价**: {latest_data['close']:.2f}\n"
            
            if pd.notna(latest_data.get('peTTM')):
                report += f"- **市盈率TTM**: {latest_data['peTTM']:.2f}\n"
            if pd.notna(latest_data.get('pbMRQ')):
                report += f"- **市净率MRQ**: {latest_data['pbMRQ']:.2f}\n"
            if pd.notna(latest_data.get('psTTM')):
                report += f"- **市销率TTM**: {latest_data['psTTM']:.2f}\n"
            if pd.notna(latest_data.get('pcfNcfTTM')):
                report += f"- **市现率TTM**: {latest_data['pcfNcfTTM']:.2f}\n"
            
            # 历史趋势分析
            report += "\n## 估值指标趋势分析\n"
            
            for metric in ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        current_val = values.iloc[-1]
                        avg_val = values.mean()
                        min_val = values.min()
                        max_val = values.max()
                        
                        metric_name = {
                            'peTTM': '市盈率TTM',
                            'pbMRQ': '市净率MRQ', 
                            'psTTM': '市销率TTM',
                            'pcfNcfTTM': '市现率TTM'
                        }[metric]
                        
                        deviation = ((current_val / avg_val) - 1) * 100 if avg_val != 0 else 0
                        percentile = (values <= current_val).mean() * 100
                        
                        report += f"\n### {metric_name}\n"
                        report += f"- 当前值: {current_val:.2f}\n"
                        report += f"- 历史均值: {avg_val:.2f}\n"
                        report += f"- 历史区间: {min_val:.2f} - {max_val:.2f}\n"
                        report += f"- 相对均值: {deviation:+.1f}%\n"
                        report += f"- 历史分位: {percentile:.1f}%\n"
            
            # 最近30天数据表格
            recent_df = df.tail(30)[['date', 'close', 'peTTM', 'pbMRQ', 'psTTM']].copy()
            recent_df = recent_df.round(4)
            
            report += "\n## 最近30个交易日估值数据\n"
            report += format_df_to_markdown(recent_df)
            
            logger.info(f"Successfully generated valuation metrics for {code}")
            return report
            
        except Exception as e:
            logger.exception(f"Error generating valuation metrics for {code}: {e}")
            return f"Error: Failed to generate valuation metrics: {e}"

    @app.tool()
    def calculate_peg_ratio(
        code: str,
        year: str,
        quarter: int
    ) -> str:
        """
        计算PEG比率（市盈率相对盈利增长比率），PEG = PE / 净利润增长率。

        Args:
            code: 股票代码，如'sh.600000'
            year: 4位数字年份，如'2024'
            quarter: 季度，1、2、3或4

        Returns:
            包含PEG比率计算和分析的详细报告
        """
        logger.info(f"Tool 'calculate_peg_ratio' called for {code}, {year}Q{quarter}")
        
        try:
            # 获取当前估值数据
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            valuation_df = active_data_source.get_historical_k_data(
                code=code,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                fields=["date", "close", "peTTM"]
            )
            
            # 获取成长能力数据
            growth_data = active_data_source.get_growth_data(
                code=code, year=year, quarter=quarter
            )
            
            if valuation_df.empty or growth_data.empty:
                return f"Error: Unable to fetch required data for PEG calculation"
            
            # 获取股票基本信息
            basic_info = active_data_source.get_stock_basic_info(code=code)
            stock_name = basic_info['code_name'].values[0] if not basic_info.empty else code
            
            # 获取最新PE
            valuation_df['peTTM'] = pd.to_numeric(valuation_df['peTTM'], errors='coerce')
            latest_pe = valuation_df['peTTM'].dropna().iloc[-1] if not valuation_df['peTTM'].dropna().empty else None
            
            # 获取净利润增长率
            growth_columns = ['YOYNI', 'YOYProfit', 'YOYEPSBasic']  # 净利润增长率相关字段
            growth_rate = None
            growth_field = None
            
            for col in growth_columns:
                if col in growth_data.columns:
                    rate = pd.to_numeric(growth_data[col].iloc[0], errors='coerce')
                    if pd.notna(rate) and rate != 0:
                        growth_rate = rate
                        growth_field = col
                        break
            
            # 生成报告
            report = f"# {stock_name} ({code}) PEG比率分析\n\n"
            report += f"**分析时点**: {year}年第{quarter}季度\n\n"
            
            if latest_pe is None:
                report += "❌ **无法计算PEG**: 缺少有效的市盈率数据\n"
                return report
            
            if growth_rate is None:
                report += "❌ **无法计算PEG**: 缺少有效的净利润增长率数据\n"
                report += f"- 当前市盈率TTM: {latest_pe:.2f}\n"
                return report
            
            # 计算PEG比率
            peg_ratio = latest_pe / growth_rate if growth_rate != 0 else float('inf')
            
            report += "## PEG比率计算结果\n"
            report += f"- **市盈率TTM**: {latest_pe:.2f}\n"
            report += f"- **净利润增长率**: {growth_rate:.2f}%\n"
            report += f"- **PEG比率**: {peg_ratio:.3f}\n\n"
            
            # PEG比率解读
            report += "## PEG比率解读\n"
            if peg_ratio < 0:
                report += "⚠️ **负增长**: 公司净利润出现负增长，PEG比率失去参考意义\n"
            elif peg_ratio < 0.5:
                report += "🟢 **低估**: PEG < 0.5，股票可能被严重低估\n"
            elif peg_ratio <= 1.0:
                report += "🟡 **合理**: 0.5 ≤ PEG ≤ 1.0，估值相对合理\n"
            elif peg_ratio <= 1.5:
                report += "🟠 **偏高**: 1.0 < PEG ≤ 1.5，估值偏高但可接受\n"
            elif peg_ratio <= 2.0:
                report += "🔴 **高估**: 1.5 < PEG ≤ 2.0，股票可能被高估\n"
            else:
                report += "🔴 **严重高估**: PEG > 2.0，股票可能被严重高估\n"
            
            report += "\n## 说明\n"
            report += "- PEG比率结合了估值和成长性，比单纯的PE更全面\n"
            report += "- 一般认为PEG=1为合理估值的分水岭\n"
            report += f"- 本次计算基于{growth_field}字段的增长率数据\n"
            report += "- PEG分析应结合行业特点和市场环境综合判断\n"
            
            logger.info(f"Successfully calculated PEG ratio for {code}")
            return report
            
        except Exception as e:
            logger.exception(f"Error calculating PEG ratio for {code}: {e}")
            return f"Error: Failed to calculate PEG ratio: {e}"

    @app.tool()
    def calculate_dcf_valuation(
        code: str,
        years_back: int = 5,
        discount_rate: float = 0.10,
        terminal_growth_rate: float = 0.025
    ) -> str:
        """
        计算DCF（现金流贴现）估值，基于历史现金流数据进行未来现金流预测和贴现。

        Args:
            code: 股票代码，如'sh.600000'
            years_back: 用于分析的历史年份数，默认5年
            discount_rate: 折现率/WACC，默认10%
            terminal_growth_rate: 永续增长率，默认2.5%

        Returns:
            包含DCF估值计算过程和结果的详细报告
        """
        logger.info(f"Tool 'calculate_dcf_valuation' called for {code}")
        
        try:
            # 获取股票基本信息
            basic_info = active_data_source.get_stock_basic_info(code=code)
            stock_name = basic_info['code_name'].values[0] if not basic_info.empty else code
            
            # 收集多年现金流数据
            current_year = datetime.now().year
            cash_flows = []
            years_data = []
            
            for i in range(years_back):
                year = str(current_year - i - 1)
                try:
                    # 获取年度现金流数据（第4季度数据代表全年）
                    cf_data = active_data_source.get_cash_flow_data(
                        code=code, year=year, quarter=4
                    )
                    
                    if not cf_data.empty:
                        # 查找经营现金流相关字段
                        cf_fields = ['manageCashFlow', 'operatingCashFlow', 'NCFFromOA']
                        annual_cf = None
                        
                        for field in cf_fields:
                            if field in cf_data.columns:
                                cf_value = pd.to_numeric(cf_data[field].iloc[0], errors='coerce')
                                if pd.notna(cf_value):
                                    annual_cf = cf_value
                                    break
                        
                        if annual_cf is not None:
                            cash_flows.append(annual_cf)
                            years_data.append((year, annual_cf))
                except:
                    continue
            
            if len(cash_flows) < 2:
                return f"Error: Insufficient cash flow data for DCF calculation (need at least 2 years)"
            
            # 反转数组，使其按时间顺序排列
            cash_flows.reverse()
            years_data.reverse()
            
            # 计算DCF估值
            dcf_result = _calculate_dcf_value(
                cash_flows=cash_flows,
                terminal_growth_rate=terminal_growth_rate,
                discount_rate=discount_rate
            )
            
            if "error" in dcf_result:
                return f"Error: {dcf_result['error']}"
            
            # 获取当前股价用于比较
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            price_data = active_data_source.get_historical_k_data(
                code=code, start_date=start_date, end_date=end_date
            )
            
            current_price = None
            if not price_data.empty:
                current_price = pd.to_numeric(price_data['close'].iloc[-1], errors='coerce')
            
            # 生成DCF估值报告
            report = f"# {stock_name} ({code}) DCF估值分析\n\n"
            
            report += "## 模型参数\n"
            report += f"- **折现率 (WACC)**: {discount_rate:.1%}\n"
            report += f"- **永续增长率**: {terminal_growth_rate:.1%}\n"
            report += f"- **预测期**: 5年\n"
            report += f"- **历史数据期**: {len(cash_flows)}年\n\n"
            
            report += "## 历史现金流数据\n"
            for year, cf in years_data:
                report += f"- {year}年: {cf:,.0f} 万元\n"
            
            # 显示增长率计算
            historical_growth = dcf_result['historical_growth']
            forecast_growth = dcf_result['forecast_growth_rate']
            
            report += f"\n## 增长率分析\n"
            report += f"- **历史复合增长率**: {historical_growth:.1%}\n"
            report += f"- **预测增长率**: {forecast_growth:.1%} (保守取值)\n\n"
            
            # DCF估值结果
            enterprise_value = dcf_result['enterprise_value']
            pv_cash_flows = dcf_result['pv_cash_flows']
            pv_terminal = dcf_result['pv_terminal_value']
            
            report += "## DCF估值结果\n"
            report += f"- **预测期现金流现值**: {pv_cash_flows:,.0f} 万元\n"
            report += f"- **终值现值**: {pv_terminal:,.0f} 万元\n"
            report += f"- **企业价值**: {enterprise_value:,.0f} 万元\n\n"
            
            # 与当前股价比较
            if current_price is not None:
                report += "## 估值比较\n"
                report += f"- **当前股价**: {current_price:.2f} 元\n"
                report += f"- **DCF理论价值**: 需要股本数据计算每股价值\n"
                report += "- **说明**: DCF计算得出的是企业整体价值，需要除以总股本得到每股价值\n\n"
            
            report += "## 重要假设与局限性\n"
            report += "1. **现金流预测**: 基于历史数据的外推，实际业务发展可能偏离预测\n"
            report += "2. **折现率假设**: 使用固定折现率，实际WACC可能随市场变化\n"
            report += "3. **永续增长率**: 假设企业能够永续经营并保持稳定增长\n"
            report += "4. **不包含债务**: 当前计算为企业价值，未扣除净债务得出股权价值\n\n"
            
            report += "**免责声明**: DCF估值高度依赖假设条件，仅供参考，不构成投资建议。"
            
            logger.info(f"Successfully calculated DCF valuation for {code}")
            return report
            
        except Exception as e:
            logger.exception(f"Error calculating DCF valuation for {code}: {e}")
            return f"Error: Failed to calculate DCF valuation: {e}"

    @app.tool()
    def compare_industry_valuation(
        code: str,
        date: Optional[str] = None
    ) -> str:
        """
        进行同行业估值比较分析，对比目标股票与同行业其他公司的估值水平。

        Args:
            code: 目标股票代码，如'sh.600000'
            date: 比较基准日期，格式'YYYY-MM-DD'，默认为最新交易日

        Returns:
            包含同行业估值比较的详细分析报告
        """
        logger.info(f"Tool 'compare_industry_valuation' called for {code}")
        
        try:
            # 获取目标股票的行业信息
            industry_data = active_data_source.get_stock_industry(code=code, date=date)
            
            if industry_data.empty:
                return f"Error: Unable to fetch industry information for {code}"
            
            target_industry = industry_data['industry'].iloc[0]
            
            # 获取同行业所有股票
            all_industry_stocks = active_data_source.get_stock_industry(date=date)
            same_industry = all_industry_stocks[
                all_industry_stocks['industry'] == target_industry
            ].copy()
            
            if len(same_industry) < 2:
                return f"Error: Insufficient companies in industry '{target_industry}' for comparison"
            
            # 设置日期范围
            if date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            else:
                end_date = date
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # 收集同行业公司估值数据
            industry_valuations = []
            
            for _, stock in same_industry.iterrows():
                stock_code = stock['code']
                try:
                    valuation_df = active_data_source.get_historical_k_data(
                        code=stock_code,
                        start_date=start_date,
                        end_date=end_date,
                        frequency="d",
                        fields=["date", "code", "close", "peTTM", "pbMRQ", "psTTM"]
                    )
                    
                    if not valuation_df.empty:
                        latest_data = valuation_df.iloc[-1]
                        
                        # 转换数值
                        pe = pd.to_numeric(latest_data.get('peTTM'), errors='coerce')
                        pb = pd.to_numeric(latest_data.get('pbMRQ'), errors='coerce')
                        ps = pd.to_numeric(latest_data.get('psTTM'), errors='coerce')
                        price = pd.to_numeric(latest_data.get('close'), errors='coerce')
                        
                        industry_valuations.append({
                            'code': stock_code,
                            'code_name': stock.get('code_name', stock_code),
                            'pe_ttm': pe,
                            'pb_mrq': pb,
                            'ps_ttm': ps,
                            'price': price,
                            'is_target': stock_code == code
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {stock_code}: {e}")
                    continue
            
            if len(industry_valuations) < 2:
                return f"Error: Unable to fetch sufficient valuation data for industry comparison"
            
            # 转换为DataFrame
            valuation_df = pd.DataFrame(industry_valuations)
            
            # 计算行业统计
            metrics = ['pe_ttm', 'pb_mrq', 'ps_ttm']
            industry_stats = {}
            
            for metric in metrics:
                valid_data = valuation_df[metric].dropna()
                if len(valid_data) > 0:
                    industry_stats[metric] = {
                        'mean': valid_data.mean(),
                        'median': valid_data.median(),
                        'min': valid_data.min(),
                        'max': valid_data.max(),
                        'std': valid_data.std()
                    }
            
            # 获取目标公司数据
            target_data = valuation_df[valuation_df['is_target'] == True]
            if target_data.empty:
                return f"Error: Target company {code} not found in industry data"
            
            target_row = target_data.iloc[0]
            
            # 生成比较报告
            report = f"# {target_row['code_name']} ({code}) 行业估值比较\n\n"
            report += f"**所属行业**: {target_industry}\n"
            report += f"**同行业公司数量**: {len(industry_valuations)} 家\n"
            report += f"**比较基准日**: {end_date}\n\n"
            
            # 目标公司估值
            report += "## 目标公司当前估值\n"
            if pd.notna(target_row['pe_ttm']):
                report += f"- **市盈率TTM**: {target_row['pe_ttm']:.2f}\n"
            if pd.notna(target_row['pb_mrq']):
                report += f"- **市净率MRQ**: {target_row['pb_mrq']:.2f}\n"
            if pd.notna(target_row['ps_ttm']):
                report += f"- **市销率TTM**: {target_row['ps_ttm']:.2f}\n"
            
            # 行业估值统计
            report += f"\n## {target_industry}行业估值统计\n"
            
            for metric in metrics:
                if metric in industry_stats:
                    stats = industry_stats[metric]
                    target_value = target_row[metric]
                    
                    metric_name = {
                        'pe_ttm': '市盈率TTM',
                        'pb_mrq': '市净率MRQ',
                        'ps_ttm': '市销率TTM'
                    }[metric]
                    
                    report += f"\n### {metric_name}\n"
                    report += f"- 行业均值: {stats['mean']:.2f}\n"
                    report += f"- 行业中位数: {stats['median']:.2f}\n"
                    report += f"- 行业区间: {stats['min']:.2f} - {stats['max']:.2f}\n"
                    
                    if pd.notna(target_value):
                        deviation_from_mean = ((target_value / stats['mean']) - 1) * 100
                        percentile = (valuation_df[metric] <= target_value).mean() * 100
                        
                        report += f"- **目标公司**: {target_value:.2f}\n"
                        report += f"- **相对均值**: {deviation_from_mean:+.1f}%\n"
                        report += f"- **行业排名**: 第{percentile:.0f}分位\n"
            
            # 估值水平评价
            report += "\n## 估值水平评价\n"
            
            for metric in metrics:
                if metric in industry_stats and pd.notna(target_row[metric]):
                    target_value = target_row[metric]
                    mean_value = industry_stats[metric]['mean']
                    
                    metric_name = {
                        'pe_ttm': '市盈率',
                        'pb_mrq': '市净率',
                        'ps_ttm': '市销率'
                    }[metric]
                    
                    if target_value < mean_value * 0.8:
                        level = "明显低估"
                    elif target_value < mean_value * 0.95:
                        level = "轻微低估"
                    elif target_value <= mean_value * 1.05:
                        level = "估值合理"
                    elif target_value <= mean_value * 1.2:
                        level = "轻微高估"
                    else:
                        level = "明显高估"
                    
                    report += f"- **{metric_name}**: {level}（相对行业均值）\n"
            
            # 行业估值数据表格（前10家公司）
            display_df = valuation_df.head(10)[['code', 'code_name', 'pe_ttm', 'pb_mrq', 'ps_ttm']].copy()
            display_df = display_df.round(2)
            
            report += f"\n## 行业主要公司估值对比（前10家）\n"
            report += format_df_to_markdown(display_df)
            
            report += "\n**说明**: 以上比较基于公开市场数据，实际投资决策还需考虑公司基本面、成长性等因素。"
            
            logger.info(f"Successfully completed industry valuation comparison for {code}")
            return report
            
        except Exception as e:
            logger.exception(f"Error in industry valuation comparison for {code}: {e}")
            return f"Error: Failed to complete industry valuation comparison: {e}" 