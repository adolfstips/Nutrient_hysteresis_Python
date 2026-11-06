#!/usr/bin/env python3
"""
Master Execution Script: Baltic Sea Nutrient Hysteresis (Python Repository)
Runs all main (Figures 1-5) and supplementary (SM Figures 1-7) scripts sequentially,
prints progress to the console, and logs all statistical and console outputs to
'collected_output.txt'.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

SCRIPTS = [
    "Plot_Fig1_Final_TNTP_MSFD_TTI_vars_breakpoints_2026_06_11.py",
    "Plot_Fig2_Final_CCF_MSFD_vars_TP_load_2026_06_11.py",
    "Plot_Fig3_Final_TTI_mean_3curves_Baltic_2026_06_11.py",
    "Plot_Fig4_Final_Hysteresis_deep_vars_TP_load_2026_06_11.py",
    "Plot_Fig5_Final_SI_IFR_network_causal_analysis_Baltic_ecosystem_2026_06_11.py",
    "Plot_SM_Fig1_CCF_MSFD_vars_TN_load_2026_06_11.py",
    "Plot_SM_Fig2_Final_Hysteresis_shallow_vars_TP_load_2026_06_11.py",
    "Plot_SM_Fig3_BP_TTI_LAG_nutrients_narrative_2026_06_11.py",
    "Plot_SM_Fig4_PPR_CHLA_annual_data_2026_06_11.py",
    "Plot_SM_Fig5_Baltic_DIP_deep_150m_Obs_Model.py",
    "Plot_SM_Fig6_Baltic_DIP_deep_50m_ICES.py",
    "Plot_SM_Fig7_Baltic_DIP_DIN_ODIN_Model.py",
]

OUTPUT_LOG = "collected_output.txt"

def log_and_print(msg, log_file):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

def main():
    start_total = time.time()
    success_count = 0

    with open(OUTPUT_LOG, "w", encoding="utf-8") as log_f:
        log_and_print("=" * 72, log_f)
        log_and_print("BALTIC SEA NUTRIENT HYSTERESIS - BATCH EXECUTION LOG", log_f)
        log_and_print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_f)
        log_and_print("=" * 72 + "\n", log_f)

        for idx, script in enumerate(SCRIPTS, 1):
            if not os.path.exists(script):
                log_and_print(f"[{idx}/{len(SCRIPTS)}] [SKIP] File not found: {script}\n", log_f)
                continue

            log_and_print(f"[{idx}/{len(SCRIPTS)}] Running: {script} ...", log_f)
            t0 = time.time()

            res = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True
            )
            dt = time.time() - t0

            if res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    log_and_print(f"    {line}", log_f)

            if res.returncode == 0:
                log_and_print(f"    --> [OK] Completed in {dt:.2f} s\n", log_f)
                success_count += 1
            else:
                log_and_print(f"    --> [ERROR] Failed (exit code {res.returncode}) after {dt:.2f} s:", log_f)
                for line in res.stderr.strip().splitlines():
                    log_and_print(f"        {line}", log_f)
                log_and_print("", log_f)

        total_dt = time.time() - start_total
        log_and_print("=" * 72, log_f)
        log_and_print(f"Batch run finished: {success_count}/{len(SCRIPTS)} scripts succeeded in {total_dt:.2f} s.", log_f)
        log_and_print(f"Full console output saved to: {OUTPUT_LOG}", log_f)
        log_and_print("=" * 72, log_f)

if __name__ == "__main__":
    main()
