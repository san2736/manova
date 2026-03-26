import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="MANOVA & ANOVA Analysis", layout="wide")

st.title("📊 MANOVA & ANOVA Analysis")
st.markdown("Perform Multivariate Analysis of Variance on your dataset")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    with st.expander("📋 Data Preview"):
        st.write(df.head(10))
        st.write(f"Shape: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    st.sidebar.header("Select Variables")
    dependent_vars = st.sidebar.multiselect(
        "Select Dependent Variables (Numeric)",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
    )
    
    independent_var = st.sidebar.selectbox(
        "Select Independent Variable (Categorical)",
        categorical_cols
    )
    
    if dependent_vars and independent_var:
        tab1, tab2, tab3 = st.tabs(["MANOVA", "ANOVA by Variable", "Post-Hoc Tests (Tukey HSD)"])
        
        with tab1:
            st.header("Multivariate Analysis of Variance (MANOVA)")
            dep_vars_str = " + ".join(dependent_vars)
            formula = f"{dep_vars_str} ~ {independent_var}"
            
            try:
                st.write(f"**Formula:** `{formula}`")
                manova = MANOVA.from_formula(formula, data=df)
                manova_results = manova.mv_test()
                st.subheader("MANOVA Results")
                st.text(str(manova_results))
                st.info("If p-value < 0.05: Significant differences exist. Proceed to ANOVA.\nIf p-value ≥ 0.05: Stop here, no significant differences.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        with tab2:
            st.header("One-Way ANOVA by Variable")
            anova_results = {}
            
            for var in dependent_vars:
                try:
                    st.subheader(f"ANOVA for {var}")
                    formula = f"{var} ~ {independent_var}"
                    fit = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(fit, typ=1)
                    st.write(anova_table)
                    
                    p_value = anova_table.loc[independent_var, "PR(>F)"]
                    if p_value < 0.05:
                        st.success(f"✓ Significant (p = {p_value:.6f})")
                        anova_results[var] = "Significant"
                    else:
                        st.warning(f"✗ Not Significant (p = {p_value:.6f})")
                        anova_results[var] = "Not Significant"
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.subheader("ANOVA Summary")
            summary_df = pd.DataFrame({
                "Variable": list(anova_results.keys()),
                "Result": list(anova_results.values())
            })
            st.table(summary_df)
        
        with tab3:
            st.header("Post-Hoc Tests (Tukey HSD)")
            st.info("Only shown for variables with significant ANOVA results (p < 0.05)")
            
            for var in dependent_vars:
                try:
                    formula = f"{var} ~ {independent_var}"
                    fit = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(fit, typ=1)
                    p_value = anova_table.loc[independent_var, "PR(>F)"]
                    
                    if p_value < 0.05:
                        st.subheader(f"Tukey HSD for {var}")
                        tukey = pairwise_tukeyhsd(df[var], groups=df[independent_var])
                        st.write(tukey._results_table)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("Select variables from the sidebar")
else:
    st.info("👈 Upload a CSV file to get started!")

