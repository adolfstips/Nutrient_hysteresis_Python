# Baltic Sea Nutrient Hysteresis (Python Companion Repository)

This repository contains the standalone Python scripts, unified datasets, and validation scripts reproducing all main and supplementary figures for the study:
**"Deep water ecological memory regulates multidecadal recovery lags in the Baltic Sea ecosystem"**

All scripts are direct, mathematically verified translations of the original R code (`strucchange`, `stats::ccf`, Liang-Kleeman Information Flow Rate / System Impact, and `ggplot2` Okabe-Ito visualizations) into pure Python (`numpy`, `pandas`, `scipy`, `matplotlib`, `networkx`).

**Recent Updates:**
* Standardized all script filenames to use the uppercase `Plot_` prefix.
* Added `run_all.py` to execute all 12 scripts in batch mode and save all statistical console outputs to `collected_output.txt`.
* All figures (Main 1–5 and SM 1–7) utilize colorblind-friendly palettes (Okabe-Ito) and include empirical validation datasets from ICES and IOW ODIN.

## 📂 Repository Structure

### 📊 Data
* **`Baltic_variables_Mt_4_Causal_Input.csv`**: The master unified dataset (1954–2022) containing annual values for Total Nitrogen (TN) and Total Phosphorus (TP) loads, TTI (Trophic Transfer Index), GES, MSFD biogeochemical variables (DIP, DIN, CHLA, Secchi), Primary Production Rate (PPR), and Oxygen Deficit (`DO2_Mt`) partitioned across Deep, Shallow, and All-Baltic domains.
* **Observational & Validation Data**: Historical observations from the ICES and IOW ODIN databases alongside localized Blue2 model extractions for Supplementary Figures SM5, SM6, and SM7:
  * `Phosphate_EGB_ICES_18.24_58.58_150m_1960_2022.csv`, `EGB_183_586_model_mon_DIP.txt`, `EGB_183_586_model_annual_DIP.txt`
  * `Gotland_dip_gt50m.csv`, `Bornholm_dip_gt50m.csv`, `ICES_Bornholm_Gotland_DIP_annual_gt50m.csv`
  * `Arkona_annual_means_summary.csv`, `EGB_annual_means_summary.csv`, `ARK_model_DIN.txt`, `ARK_model_DIP.txt`, `EGB_model_DIN.txt`, `EGB_model_DIP.txt`

### 💻 Python Scripts & 📈 Output Figures

* **Master Batch Runner**
  * Script: `run_all.py`
  * Log Output: `collected_output.txt` (captures all breakpoints, CCF lags, System Impact scores, and model validation skill metrics)

#### Main Article
* **Figure 1: Asynchronous Structural Regime Shifts**
  * Script: `Plot_Fig1_Final_TNTP_MSFD_TTI_vars_breakpoints_2026_06_11.py`
  * Output: `Figure_1_Baltic_Asynchronous_Shifts.png`
* **Figure 2: Time Lag Analysis (TP Load)**
  * Script: `Plot_Fig2_Final_CCF_MSFD_vars_TP_load_2026_06_11.py`
  * Output: `Figure_2_Baltic_CCF_TP_HELCOM_deep.png`
* **Figure 3: Trophic Transfer Index (TTI) Over Time**
  * Script: `Plot_Fig3_Final_TTI_mean_3curves_Baltic_2026_06_11.py`
  * Output: `Figure_3_Baltic_TTI_deep_shallow_smoothed.png`
* **Figure 4: Phase-Space Trajectories (Deep Basin Hysteresis)**
  * Script: `Plot_Fig4_Final_Hysteresis_deep_vars_TP_load_2026_06_11.py`
  * Output: `Figure_4_Baltic_Hysteresis_Combined.png`
* **Figure 5: System Impact & IFR Causality Networks**
  * Script: `Plot_Fig5_Final_SI_IFR_network_causal_analysis_Baltic_ecosystem_2026_06_11.py`
  * Output: `Figure_5_Baltic_Causality_Combined.png`

#### Supplementary Material (SM)
* **SM Figure 1: Time Lag Analysis (TN Load)**
  * Script: `Plot_SM_Fig1_CCF_MSFD_vars_TN_load_2026_06_11.py`
  * Output: `Figure_SM1_Baltic_CCF_TN_HELCOM_deep.png`
* **SM Figure 2: Phase-Space Trajectories (Shallow Basin Hysteresis)**
  * Script: `Plot_SM_Fig2_Final_Hysteresis_shallow_vars_TP_load_2026_06_11.py`
  * Output: `Figure_SM2_Baltic_Hysteresis_Shallow.png`
* **SM Figure 3: Nutrients Narrative & CCF Composite**
  * Script: `Plot_SM_Fig3_BP_TTI_LAG_nutrients_narrative_2026_06_11.py`
  * Output: `Figure_SM3_Baltic_TTI_LAG_TP_CCF_Analysis_Composite.png`
* **SM Figure 4: Baltic Sea Productivity (PPR vs Chl-a)**
  * Script: `Plot_SM_Fig4_PPR_CHLA_annual_data_2026_06_11.py`
  * Output: `Figure_SM4_Baltic_PPR_CHLA.png`
* **SM Figure 5: Deep-Water (150m) Phosphorus Trap Verification**
  * Script: `Plot_SM_Fig5_Baltic_DIP_deep_150m_Obs_Model.py`
  * Output: `Figure_SM5_Baltic_Deep_150m_DIP.png`
* **SM Figure 6: Regional Deep-Water (>50m) Phosphorus Accumulation**
  * Script: `Plot_SM_Fig6_Baltic_DIP_deep_50m_ICES.py`
  * Output: `Figure_SM6_Baltic_Regional_Deep_DIP.png`
* **SM Figure 7: Surface Nutrient Observations vs. Model (ODIN)**
  * Script: `Plot_SM_Fig7_Baltic_DIP_DIN_ODIN_Model.py`
  * Output: `Figure_SM7_Baltic_ODIN_vs_Model_Combined.png`

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run all scripts at once (generates all `.png` figures and writes console statistics to `collected_output.txt`):
   ```bash
   python run_all.py
   ```
3. Or run any individual script directly:
   ```bash
   python Plot_Fig1_Final_TNTP_MSFD_TTI_vars_breakpoints_2026_06_11.py
   ```
