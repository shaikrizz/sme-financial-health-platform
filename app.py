import streamlit as st
import pdfplumber
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import requests

def ollama_insights(metrics: dict, model: str = "llama3.2") -> str:
    """
    Calls local Ollama running at http://localhost:11434
    Returns AI insights text. Falls back gracefully if Ollama isn't running.
    """
    rev = metrics.get("revenue", 0)
    exp = metrics.get("expense", 0)
    pl = metrics.get("profit_loss", 0)
    ratio = metrics.get("expense_ratio", 0)
    risk = metrics.get("risk_level", "Unknown")
    credit = metrics.get("credit_level", "Unknown")

    prompt = f"""
    You are a finance assistant for Indian SMEs.

    IMPORTANT:
    - Use the numbers exactly as provided.
    - Do NOT recalculate or modify them.
    - When you show any amount, copy the exact digits.

    OFFICIAL NUMBERS (copy exactly):
    REVENUE = {rev}
    EXPENSE = {exp}
    NET = {pl}
    EXPENSE_RATIO = {ratio}
    RISK_LEVEL = {risk}
    CREDIT = {credit}

    Write:
    1) Summary (must display REVENUE, EXPENSE, NET exactly)
    2) 5 bullet insights
    3) 5 actionable recommendations
    4) 2 suitable bank/NBFC product types (generic, no brand names)

    Keep it simple for non-finance owners.
    """


    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip() or "No response from local AI."
    except Exception as e:
        return f"Local AI not available (Ollama). Using fallback instead.\n\nReason: {e}"


def H(text):
    st.header(text.upper())

def forecast_next_months(df: pd.DataFrame, months: int = 3) -> pd.DataFrame:
    """
    Beginner-friendly monthly forecasting using Linear Regression on monthly totals.
    Assumes df has columns: Date, Category, Amount, Description
    Category: Revenue/Expense (case-insensitive)
    Returns a DataFrame with next `months` rows: Month, Forecast_Revenue, Forecast_Expense, Forecast_Net
    """

    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Category", "Amount"])

    data["Category"] = data["Category"].astype(str).str.strip().str.lower()
    data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce")
    data = data.dropna(subset=["Amount"])

    # Keep only revenue/expense
    data = data[data["Category"].isin(["revenue", "expense"])]
    if data.empty:
        return pd.DataFrame()

    # Monthly totals per category
    data["Month"] = data["Date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        data.groupby(["Month", "Category"])["Amount"]
        .sum()
        .reset_index()
        .pivot(index="Month", columns="Category", values="Amount")
        .fillna(0.0)
        .reset_index()
    )

    # Ensure both columns exist
    if "revenue" not in monthly.columns:
        monthly["revenue"] = 0.0
    if "expense" not in monthly.columns:
        monthly["expense"] = 0.0

    # Need at least 2 months for regression
    monthly = monthly.sort_values("Month").reset_index(drop=True)
    if len(monthly) < 2:
        return pd.DataFrame()

    # Build a simple time index
    monthly["t"] = np.arange(len(monthly))

    def _predict(series: pd.Series) -> np.ndarray:
        X = monthly[["t"]].values
        y = series.values
        model = LinearRegression()
        model.fit(X, y)

        future_t = np.arange(len(monthly), len(monthly) + months).reshape(-1, 1)
        pred = model.predict(future_t)

        # Avoid negative forecasts (common beginner safeguard)
        pred = np.maximum(pred, 0.0)
        return pred

    rev_pred = _predict(monthly["revenue"])
    exp_pred = _predict(monthly["expense"])

    last_month = monthly["Month"].iloc[-1]
    future_months = pd.date_range(
        start=(last_month + pd.offsets.MonthBegin(1)),
        periods=months,
        freq="MS"
    )

    out = pd.DataFrame({
        "Month": future_months,
        "Forecast_Revenue": rev_pred,
        "Forecast_Expense": exp_pred
    })
    out["Forecast_Net"] = out["Forecast_Revenue"] - out["Forecast_Expense"]

    return out


def t(key: str, lang: str, translations: dict) -> str:
    # Fallback to English if key missing
    return translations.get(lang, {}).get(key) or translations["English"].get(key, key)


def build_report_text(company_name, industry, metrics, monthly_df=None, forecast_df=None, benchmark_df=None, insights_text=None):
    lines = []
    lines.append(f"SME Financial Health Report")
    lines.append(f"Company: {company_name}")
    lines.append(f"Industry: {industry}")
    lines.append("-" * 50)

    # Core KPIs
    rev = float(metrics.get("revenue", 0))
    exp = float(metrics.get("expense", 0))
    pl = float(metrics.get("profit_loss", 0))
    risk = str(metrics.get("risk_level", "Unknown"))
    credit = str(metrics.get("credit_level", "Unknown"))

    lines.append("Key Metrics")
    lines.append(f"Total Revenue: {rev:,.0f}")
    lines.append(f"Total Expense: {exp:,.0f}")
    lines.append(f"Net (Revenue - Expense): {pl:,.0f}")
    if rev > 0:
        lines.append(f"Expense Ratio: {(exp/rev):.2f}")
        lines.append(f"Profit Margin: {(pl/rev):.2f}")
    lines.append(f"Risk Level: {risk}")
    lines.append(f"Creditworthiness: {credit}")
    lines.append("-" * 50)

    # Monthly summary (top few rows)
    if monthly_df is not None and not monthly_df.empty:
        lines.append("Monthly Summary (latest months)")
        preview = monthly_df.tail(6)
        for _, r in preview.iterrows():
            lines.append(f"{r['Month']}: Revenue={float(r['Revenue']):,.0f}, Expense={float(r['Expense']):,.0f}, Net={float(r['Net']):,.0f}")
        lines.append("-" * 50)

    # Forecast
    if forecast_df is not None and not forecast_df.empty:
        lines.append("Forecast (next months)")
        for _, r in forecast_df.iterrows():
            m = r["Month"]
            if hasattr(m, "strftime"):
                m = m.strftime("%b %Y")
            lines.append(
                f"{m}: Revenue={float(r['Forecast_Revenue']):,.0f}, "
                f"Expense={float(r['Forecast_Expense']):,.0f}, "
                f"Net={float(r['Forecast_Net']):,.0f}"
            )
        lines.append("-" * 50)

    # Benchmarking
    if benchmark_df is not None and not benchmark_df.empty:
        lines.append("Industry Benchmark Comparison")
        for _, r in benchmark_df.iterrows():
            lines.append(f"{r['Metric']}: Your={r['Your Business']} | Industry={r['Industry Avg']} | Status={r['Status']}")
        lines.append("-" * 50)

    # Insights
    if insights_text:
        lines.append("AI Insights & Recommendations")
        lines.append(insights_text.strip())
        lines.append("-" * 50)

    lines.append("Disclaimer: This report is generated for demo purposes and should not be treated as financial advice.")
    return "\n".join(lines)


def report_text_to_pdf_bytes(report_text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left = 40
    top = height - 50
    line_height = 14

    y = top
    for line in report_text.split("\n"):
        if y < 50:
            c.showPage()
            y = top
        c.drawString(left, y, str(line)[:120])  # simple wrap-safety
        y -= line_height

    c.save()
    buffer.seek(0)
    return buffer.read()


# Industrial benchmarking
def industry_benchmark_table(metrics, industry):
    """
    Compare business metrics with simple industry benchmarks
    """

    benchmarks = {
        "Retail": {
            "expense_ratio": 0.75,
            "profit_margin": 0.20
        },
        "Restaurant": {
            "expense_ratio": 0.85,
            "profit_margin": 0.12
        },
        "IT Services": {
            "expense_ratio": 0.60,
            "profit_margin": 0.35
        },
        "Manufacturing": {
            "expense_ratio": 0.70,
            "profit_margin": 0.25
        }
    }

    bench = benchmarks[industry]

    exp_ratio = metrics["expense"] / metrics["revenue"] if metrics["revenue"] else 0
    profit_margin = metrics["profit_loss"] / metrics["revenue"] if metrics["revenue"] else 0

    data = []

    def status_good_low(val, bench):
        if val <= bench:
            return "✅ Good"
        elif val <= bench * 1.1:
            return "⚠ Average"
        return "❌ High"

    def status_good_high(val, bench):
        if val >= bench:
            return "✅ Good"
        elif val >= bench * 0.9:
            return "⚠ Average"
        return "❌ Low"

    data.append(["Expense Ratio", f"{exp_ratio:.2f}", f"{bench['expense_ratio']:.2f}", status_good_low(exp_ratio, bench['expense_ratio'])])
    data.append(["Profit Margin", f"{profit_margin:.2f}", f"{bench['profit_margin']:.2f}", status_good_high(profit_margin, bench['profit_margin'])])

    return pd.DataFrame(data, columns=["Metric", "Your Business", "Industry Avg", "Status"])


# Monthly overview
def monthly_summary_table(df):
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Category", "Amount"])

    data["Category"] = data["Category"].astype(str).str.strip().str.lower()
    data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce")
    data = data.dropna(subset=["Amount"])

    # Month column
    data["Month"] = data["Date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        data.groupby(["Month", "Category"])["Amount"]
        .sum()
        .reset_index()
        .pivot(index="Month", columns="Category", values="Amount")
        .fillna(0.0)
        .reset_index()
    )

    # Ensure both exist even if missing in data
    if "revenue" not in monthly.columns:
        monthly["revenue"] = 0.0
    if "expense" not in monthly.columns:
        monthly["expense"] = 0.0

    monthly["Net"] = monthly["revenue"] - monthly["expense"]

    # Nice column names
    monthly = monthly.rename(columns={"revenue": "Revenue", "expense": "Expense"})
    monthly["Month"] = monthly["Month"].dt.strftime("%b %Y")

    return monthly[["Month", "Revenue", "Expense", "Net"]]


def smart_fallback_insights(metrics: dict, language: str) -> str:
    rev = float(metrics.get("revenue", 0))
    exp = float(metrics.get("expense", 0))
    pl = float(metrics.get("profit_loss", 0))
    ratio = float(metrics.get("expense_ratio", 1))
    risk = str(metrics.get("risk_level", "Unknown"))
    credit = str(metrics.get("credit_level", "Unknown"))

    # Derived KPIs
    profit_margin = (pl / rev) if rev > 0 else 0.0
    burn_rate = exp  # simplistic monthly burn if rows represent a month; still useful wording
    runway_months = (pl / exp) if (pl > 0 and exp > 0) else 0.0  # rough; used carefully

    # Insight flags
    loss = pl < 0
    high_cost = ratio >= 0.85
    moderate_cost = 0.70 <= ratio < 0.85
    healthy = (pl > 0 and ratio < 0.70)

    # Product recommendations (types, hackathon-safe)
    products = []
    if loss or risk.lower().startswith("high"):
        products.append(("Working Capital / Overdraft", "Helps manage cash gaps while you stabilize expenses and collections."))
        products.append(("Invoice Financing", "Useful if sales exist but customer payments are delayed."))
    elif moderate_cost or credit.lower() == "fair":
        products.append(("Short-term Working Capital Loan", "Good for managing seasonal cash needs and vendor payments."))
        products.append(("Business Credit Card / Line of Credit", "Flexible for small operational spends with short repayment cycles."))
    else:
        products.append(("Bank Term Loan (MSME)", "Suitable for growth/expansion with better interest rates when risk is low."))
        products.append(("Equipment / Machinery Finance", "Best if you plan capex; keeps cash free for operations."))

    # Recommendations (rule-based, customized)
    actions = []
    if high_cost:
        actions.append("Cut non-essential operating costs (target 5–10% reduction) and renegotiate fixed expenses (rent, subscriptions, logistics).")
    if moderate_cost:
        actions.append("Track top expense heads weekly; set category budgets to prevent cost drift.")
    if loss:
        actions.append("Prioritize profitability: pause low-margin offerings and focus on high-margin revenue lines.")
    if profit_margin < 0.10 and not loss:
        actions.append("Improve margin: review pricing, vendor rates, and discounts; push higher-margin products/services.")
    if rev > 0 and exp > 0:
        actions.append("Improve working capital: shorten receivables cycle (follow-ups, early-pay discounts) and extend payables where possible.")
    actions.append("Maintain a cash buffer of 30–60 days of expenses to reduce financial stress during low-sales periods.")

    # Risk explanation
    risk_reason = []
    if loss:
        risk_reason.append("Net loss indicates higher financial risk and weaker repayment capacity.")
    if high_cost:
        risk_reason.append("High expense ratio suggests cost pressure and lower resilience.")
    if healthy:
        risk_reason.append("Healthy surplus and controlled expense ratio indicate stability.")
    if not risk_reason:
        risk_reason.append("Risk level is driven by your profit and expense ratio trend.")

    # Format output (English/Hindi)
    if language == "Hindi":
        return f"""
## ✅ AI Insights & Recommendations (Fallback)

### 📊 सारांश (Summary)
- कुल राजस्व: ₹{rev:,.0f}
- कुल खर्च: ₹{exp:,.0f}
- नेट स्थिति (Revenue-Expense): ₹{pl:,.0f}
- खर्च अनुपात (Expense Ratio): {ratio:.2f}
- जोखिम स्तर: {risk} | क्रेडिट संकेत: {credit}

### 🔎 मुख्य इनसाइट्स
- लाभ मार्जिन (approx): {profit_margin*100:.1f}% {"(नुकसान)" if loss else ""}
- {"खर्च बहुत अधिक है — लागत नियंत्रण ज़रूरी है।" if high_cost else "लागत नियंत्रण ठीक है, लेकिन निगरानी जरूरी है।" if moderate_cost else "लागत नियंत्रण अच्छा है।"}
- कारण: {"; ".join(risk_reason)}

### 🎯 एक्शन सुझाव (Actionable)
{chr(10).join([f"- {a}" for a in actions[:5]])}

### 🏦 उपयुक्त बैंक/NBFC प्रोडक्ट्स
- {products[0][0]} — {products[0][1]}
- {products[1][0]} — {products[1][1]}

### 📌 Investor-ready Summary
बिज़नेस की नेट स्थिति ₹{pl:,.0f} है और खर्च अनुपात {ratio:.2f} है। {("लागत घटाकर और कलेक्शन बेहतर करके" if high_cost or loss else "कलेक्शन और लागत अनुशासन बनाए रखकर")} आगे का ग्रोथ मजबूत किया जा सकता है।
"""
    else:
        return f"""
## ✅ AI Insights & Recommendations (Fallback)

### 📊 Summary
- Total Revenue: ₹{rev:,.0f}
- Total Expenses: ₹{exp:,.0f}
- Net Position (Revenue-Expense): ₹{pl:,.0f}
- Expense Ratio: {ratio:.2f}
- Risk Level: {risk} | Credit Indicator: {credit}

### 🔎 Key Insights
- Approx profit margin: {profit_margin*100:.1f}% {"(Loss)" if loss else ""}
- {"Costs are very high — strong cost control is needed." if high_cost else "Costs are moderate — monitor weekly." if moderate_cost else "Costs are well controlled."}
- Reasoning: {"; ".join(risk_reason)}

### 🎯 Actionable Recommendations
{chr(10).join([f"- {a}" for a in actions[:5]])}

### 🏦 Suitable Bank/NBFC Products
- {products[0][0]} — {products[0][1]}
- {products[1][0]} — {products[1][1]}

### 📌 Investor-ready Summary
The business shows a net position of ₹{pl:,.0f} with an expense ratio of {ratio:.2f}. {("Improving cost discipline and collections" if high_cost or loss else "Maintaining cost discipline and cash-cycle control")} can strengthen growth readiness.
"""


st.title("AI Financial Health Assessment for SMEs")

uploaded_file = st.file_uploader(
    "Upload Financial File (CSV, XLSX, PDF)",
    type=["csv", "xlsx", "pdf"]
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully")
            st.dataframe(df)

        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file)
            st.success("Excel file loaded successfully")
            st.dataframe(df)

        elif file_type == "pdf":
            st.success("PDF file uploaded")
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            st.text_area("Extracted PDF Text (Preview)", text[:3000])

    except Exception as e:
        st.error(f"Error reading file: {e}")

# ---------------- STEP 2 + STEP 3: Financial Analysis + Risk/Credit ----------------

if uploaded_file is not None and file_type in ["csv", "xlsx"]:
    st.markdown("---")
    st.header("Financial Overview")
    
   
    # Normalize column names
    df.columns = df.columns.str.strip()
    

    if "Category" in df.columns and "Amount" in df.columns:

        # Clean values
        df["Category"] = df["Category"].astype(str).str.strip().str.lower()
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

        # Step 2 calculations
        revenue = df[df["Category"] == "revenue"]["Amount"].sum()
        expense = df[df["Category"] == "expense"]["Amount"].sum()
        profit_loss = revenue - expense

        # ---- Step 2 Metrics ----
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"₹ {revenue:,.2f}")
        col2.metric("Total Expenses", f"₹ {expense:,.2f}")
        col3.metric("Net Amount", f"₹ {profit_loss:,.2f}")
        
        # Monthly overview UI
        st.subheader("📅 Monthly Financial Summary")
        monthly_df = monthly_summary_table(df)
        st.dataframe(monthly_df, use_container_width=True)


        # ---- Step 2 Numeric Surplus/Deficit ----
        if profit_loss >= 0:
            st.success(f"📈 Amount Surplus: ₹ {profit_loss:,.2f}")
        else:
            st.error(f"📉 Amount Deficit: ₹ {abs(profit_loss):,.2f}")

        # ---- Step 2 Identifier ----
        st.subheader("Cash Flow Status")
        if profit_loss > 0:
            st.success("Positive cash flow detected")
        elif profit_loss < 0:
            st.error("Negative cash flow detected")
        else:
            st.warning("Break-even cash flow")

        # ---------------- STEP 3: Risk & Creditworthiness ----------------
        st.markdown("---")
        st.header("Financial Risk & Credit Assessment")

        expense_ratio = (expense / revenue) if revenue > 0 else 1.0

        # Risk assessment
        if profit_loss > 0 and expense_ratio < 0.7:
            risk_level = "Low Risk"
            st.success("🟢 Low Financial Risk")
        elif profit_loss > 0 and expense_ratio <= 0.9:
            risk_level = "Moderate Risk"
            st.warning("🟡 Moderate Financial Risk")
        else:
            risk_level = "High Risk"
            st.error("🔴 High Financial Risk")

        # Creditworthiness indicator
        st.subheader("Creditworthiness Indicator")

        if profit_loss > 0 and expense_ratio < 0.8:
            credit_level = "Good"
            st.success("✅ Good creditworthiness — Eligible for bank/NBFC products")

        elif profit_loss > 0:
            credit_level = "Fair"
            st.warning("⚠️ Fair creditworthiness — Limited loan eligibility")

        else:
            credit_level = "Poor"
            st.error("❌ Weak creditworthiness — High default risk")


        # Display supporting metrics
        c1, c2 = st.columns(2)
        c1.metric("Expense Ratio", f"{expense_ratio:.2f}")
        c2.metric("Risk Level", risk_level)
        
        # ✅ Build metrics dict for Step 4
        metrics = {
            "revenue": revenue,
            "expense": expense,
            "profit_loss": profit_loss,
            "expense_ratio": expense_ratio,
            "risk_level": risk_level,
            "credit_level": credit_level,
        }
        


   
        # ---------------- STEP 4: AI Insights (Offline Smart Fallback) ----------------

        if st.button("Generate AI Report"):
            ai_text = ollama_insights(metrics, model="llama3.2")

            # automatic fallback if Ollama fails
            if not ai_text or "not available" in ai_text.lower():
                ai_text = smart_fallback_insights(metrics, "English")

                st.session_state["insights_text"] = ai_text
            st.markdown(ai_text)


        
        # ---------------- STEP 5: Cost Optimization & Expense Breakdown ----------------
        st.markdown("---")
        st.header("Cost Optimization & Expense Breakdown")

        # If Description exists, we can break down expenses meaningfully
        has_desc = "Description" in df.columns

        # Build an expense-only dataframe
        exp_df = df[df["Category"] == "expense"].copy()

        if exp_df.empty:
            st.info("No expense rows found to analyze cost optimization.")
        else:
            # Choose grouping column
            group_col = "Description" if has_desc else "Category"

            # Summarize expenses
            exp_summary = (
                exp_df.groupby(group_col)["Amount"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={group_col: "Expense Head", "Amount": "Total Amount"})
            )

            st.write("### Top Expense Heads")
            st.dataframe(exp_summary, use_container_width=True)

            # Charts
            st.write("### Expense Distribution (Top 8)")
            top8 = exp_summary.head(8).set_index("Expense Head")
            st.bar_chart(top8["Total Amount"])

            # Key ratios for recommendations
            expense_pct = (expense / revenue) * 100 if revenue > 0 else 0

            st.write("### Optimization Suggestions")
            suggestions = []

            if expense_pct >= 85:
                suggestions.append("🔴 Expenses are very high vs revenue. Target an immediate 10–15% cut in non-essential spending.")
            elif expense_pct >= 70:
                suggestions.append("🟡 Expenses are moderate vs revenue. Aim for 5–10% optimization and track weekly budgets.")
            else:
                suggestions.append("🟢 Expenses are under control. Maintain discipline and monitor major fixed costs monthly.")

            # Suggest cutting top head
            top_head = exp_summary.iloc[0]["Expense Head"]
            top_amt = exp_summary.iloc[0]["Total Amount"]
            suggestions.append(f"✅ Biggest expense head is **{top_head}** (₹ {top_amt:,.0f}). Review vendor rates / usage for quick savings.")

            # Basic working-capital style suggestion
            suggestions.append("💡 Set category budgets (Rent, Salaries, Marketing, Utilities) and review variance every week.")

            for s in suggestions:
                st.write("-", s)

            # Optional: monthly trend if Date exists
            if "Date" in df.columns:
                try:
                    tmp = df.copy()
                    tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
                    tmp = tmp.dropna(subset=["Date"])
                    tmp["Month"] = tmp["Date"].dt.to_period("M").astype(str)

                    monthly = tmp.pivot_table(
                        index="Month",
                        columns="Category",
                        values="Amount",
                        aggfunc="sum",
                        fill_value=0
                    ).reset_index()

                    st.write("### Monthly Trend (Revenue vs Expense)")
                    st.dataframe(monthly, use_container_width=True)

                    monthly_chart = monthly.set_index("Month")
                    chart_type = st.selectbox(
                        "Select Chart Type",
                        ["Line", "Bar", "Area"]
                    )

                    if chart_type == "Line":
                        st.line_chart(monthly_chart)
                    elif chart_type == "Bar":
                        st.bar_chart(monthly_chart)
                    else:
                        st.area_chart(monthly_chart)

                except Exception:
                    st.info("Date column exists but monthly trend could not be generated (check date format).")
            
            
            
            # Step-6 UI
            st.markdown("---")
            st.header("📈Financial Forecasting")

            # Let user choose forecast horizon (default 3)
            months_to_forecast = st.selectbox("Forecast horizon (months)", [3, 4, 6], index=0)

            forecast_df = forecast_next_months(df, months=months_to_forecast)

            if forecast_df.empty:
                st.info("Not enough data to forecast. Add at least 2 months of revenue/expense records.")
            else:
                st.subheader("Forecast Table")
                show_df = forecast_df.copy()
                show_df["Month"] = show_df["Month"].dt.strftime("%b %Y")
                st.dataframe(show_df, use_container_width=True, hide_index=True)

                # Chart: Revenue vs Expense forecast
                st.subheader("Forecast Chart: Revenue vs Expense")
                chart_df = forecast_df.set_index("Month")[["Forecast_Revenue", "Forecast_Expense"]]
                st.line_chart(chart_df)

                # Chart: Net forecast
                st.subheader("Forecast Chart: Net (Revenue - Expense)")
                st.line_chart(forecast_df.set_index("Month")[["Forecast_Net"]])

                # Simple projection insights
                avg_net = float(forecast_df["Forecast_Net"].mean())
                total_net = float(forecast_df["Forecast_Net"].sum())

                c1, c2 = st.columns(2)
                c1.metric("Avg Forecast Net / Month", f"{avg_net:,.0f}")
                c2.metric(f"Total Forecast Net ({months_to_forecast} mo)", f"{total_net:,.0f}")

                if avg_net < 0:
                    st.error("⚠️ Forecast shows a deficit trend. Consider reducing expenses or increasing revenue.")
                elif avg_net < (0.1 * float(forecast_df["Forecast_Revenue"].mean())):  # small margin
                    st.warning("⚠️ Forecast margin is low. Small shocks may cause deficit.")
                else:
                    st.success("✅ Forecast looks healthy based on current trend.")
                    
                    
                #----------------------------------Step 7 - Industrial Benchmarking ----------------------------------
                    
            st.markdown("---")
            st.header("🏭Industry Benchmarking")

            industry = st.selectbox(
                "Select Industry",
                ["Retail", "Restaurant", "IT Services", "Manufacturing"]
            )

            benchmark_df = industry_benchmark_table(metrics, industry)

            st.subheader("Benchmark Comparison")
            st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.header("📄 Export Report")

            company_name = st.text_input("Company / Business Name", value="My SME")
            # reuse your selected industry variable (industry) from benchmarking
            # if you want, keep a default if benchmarking not selected
            try:
                selected_industry = industry
            except NameError:
                selected_industry = "Not Selected"

            st.caption("Download an investor-ready summary of KPIs, monthly trends, forecast, benchmarks, and insights.")

            # Prepare data for report
            # monthly_df should exist from your monthly summary table section; if not, compute it here safely
            try:
                monthly_for_report = monthly_df
            except NameError:
                monthly_for_report = pd.DataFrame()

            # forecast_df should exist from forecasting section; if not, set empty
            try:
                forecast_for_report = forecast_df
            except NameError:
                forecast_for_report = pd.DataFrame()

            # benchmark_df should exist from benchmarking section; if not, set empty
            try:
                benchmark_for_report = benchmark_df
            except NameError:
                benchmark_for_report = pd.DataFrame()

            # insights text: use what you already generate in Step 4 (fallback). If you stored it, use that variable.
            # If you don't have it stored, keep empty for now.
            try:
                insights_for_report = insights_text  # rename this to your actual insights variable if different
            except NameError:
                insights_for_report = ""

            report_text = build_report_text(
                company_name=company_name,
                industry=selected_industry,
                metrics=metrics,
                monthly_df=monthly_for_report,
                forecast_df=forecast_for_report,
                benchmark_df=benchmark_for_report,
                insights_text=insights_for_report
            )

            colA, colB = st.columns(2)

            with colA:
                st.download_button(
                    label="⬇️ Download TXT Report",
                    data=report_text.encode("utf-8"),
                    file_name=f"{company_name}_report.txt",
                    mime="text/plain"
                )

            with colB:
                pdf_bytes = report_text_to_pdf_bytes(report_text)
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{company_name}_report.pdf",
                    mime="application/pdf"
                )

            with st.expander("Preview Report Text"):
                st.text(report_text)
                
            #-------------------------------------step-8-----------------------------------------------
            
            st.markdown("---")
            st.header("🔒 Security & Compliance")

            st.info("""
            • Financial data is processed locally in memory only  
            • No permanent storage of uploaded files  
            • No third-party data sharing  
            • No external API transmission during analysis  
            • Session-based processing (auto-cleared on refresh)  
            • Designed with privacy-first architecture for SMEs  
            """)

    else:
        st.error("CSV must contain 'Category' and 'Amount' columns")

    