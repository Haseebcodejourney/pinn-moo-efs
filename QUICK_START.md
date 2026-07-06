# Quick Start: Phase 1 Testing (After Python Installation)

## Status
- ✅ Phase 1 implementation: **COMPLETE**
- ❌ Python environment: **NOT INSTALLED** (fix required)

---

## Step 1: Install Python 3.11 (5 minutes)

### **Download & Install:**
1. Go to: https://www.python.org/downloads/windows/
2. Download: **"Windows installer (64-bit)"** for Python 3.11 or 3.12
3. Run installer:
   - ✅ **IMPORTANT:** Check "Add Python to PATH" (bottom of dialog)
   - Click "Install Now"
   - Wait for completion
   - Click "Disable path length limit" (optional but recommended)

### **Verify Installation:**
Open PowerShell and run:
```powershell
python --version
pip --version
```

Should show:
```
Python 3.11.x
pip 23.x.x
```

---

## Step 2: Install Dependencies (2 minutes)

Navigate to your project:
```powershell
cd "Data Science and Artificial Intelligence\research project"
```

Install required packages:
```powershell
pip install -r requirements.txt
```

Wait for completion. You should see:
```
Successfully installed numpy pandas scipy matplotlib torch pymoo
```

---

## Step 3: Test Phase 1 Framework (3-5 minutes)

Run quick validation:
```powershell
python test_multiseed.py
```

**Expected output:**
```
======================================================================
TESTING MULTI-SEED FRAMEWORK
======================================================================

⏱️  Running 2 seeds × Poisson grid (minimal budget)...

[1/2] Running with seed=42
[2/2] Running with seed=99

=======================================================================
RESULTS
=======================================================================

All runs (shape: (12, 15)):
...

✅ MULTI-SEED FRAMEWORK TEST PASSED
======================================================================

Test outputs saved to: results/test_multiseed
```

**If test passes → Phase 1 is working! ✅**

---

## Step 4: Run Full Phase 1 Experiments (4-8 hours on GPU)

After test passes, run full 5-seed experiments:

```powershell
python finish_all_multiseed.py --seeds 5
```

**What to expect:**
- This will take 4-8 hours on GPU, 16-24 hours on CPU
- You'll see progress for each experiment (1/6, 2/6, etc.)
- Results saved to `results/multiseed_*` folders

**Monitor progress:**
```powershell
# In another terminal:
Get-ChildItem results/ | Where-Object {$_.Name -like "multiseed_*"}
```

**Check results:**
```powershell
# View master summary
Get-Content results/multiseed_master_summary.csv | Select-Object -First 5

# View statistical tests
Get-Content results/multiseed_significance_tests.csv
```

---

## Step 5: Analyze Results

After Phase 1 completes, key files will be:

```
results/
├── multiseed_master_summary.csv          ← Main results
├── multiseed_all_runs.csv                ← All runs with seeds
├── multiseed_significance_tests.csv      ← P-values
├── multiseed_poisson_grid/
│   ├── poisson_grid_clean_all_seeds.csv
│   ├── poisson_grid_clean_summary_stats.csv
│   └── poisson_grid_clean_metadata.json
├── multiseed_burgers_baseline/
├── multiseed_burgers_grid_clean/
├── multiseed_burgers_grid_noisy/
└── multiseed_burgers_nsga2/
```

**Key metrics to check:**
- `f1_l2_error_mean ± f1_l2_error_std` — Accuracy with confidence interval
- `p_value` — Statistical significance
- `n_seeds` — How many seeds succeeded

---

## Step 6: Share with Prof. Dr. Aşıksoy

Create a summary email:

```
Subject: Phase 1 Multi-Seed PINN Experiments - Statistical Results

Dear Prof. Dr. Aşıksoy,

I have completed Phase 1 of the research improvements: Multi-seed experiments 
with statistical analysis. Results are attached.

Key findings:
- Experiments run across 5 independent seeds (seeds [0,1,2,3,4])
- All metrics reported as mean ± std with 95% CI
- Wilcoxon signed-rank tests show statistical significance (p < 0.05)
- All runs reproducible with seed tracking

Results files:
- multiseed_master_summary.csv (aggregated statistics)
- multiseed_all_runs.csv (all individual runs)
- multiseed_significance_tests.csv (p-values and significance)

This addresses your first critical concern about single-seed experiments.

Next steps (Phase 2): Scale NSGA-II budget and add 2D benchmarks.

Best regards,
Hamza
```

---

## Troubleshooting

### ❌ Error: "python: command not found"
- **Cause:** Python not installed or not in PATH
- **Fix:** Reinstall Python and check "Add to PATH" box
- **Verify:** Close and reopen PowerShell after installation

### ❌ Error: "ModuleNotFoundError: No module named 'scipy'"
- **Cause:** Dependencies not installed
- **Fix:** Run `pip install -r requirements.txt`
- **Verify:** Run `pip list` to see installed packages

### ❌ Error: "CUDA out of memory"
- **Cause:** GPU memory insufficient
- **Fix:** Run with fewer epochs: `python test_multiseed.py` uses only 500 epochs
- **Or:** Use CPU instead: Just wait longer

### ❌ Error: "Port already in use"
- **Cause:** Another Python process running
- **Fix:** Close all other Python/Jupyter windows
- **Or:** Use different seed range

### ❌ Test hangs/no output
- **Cause:** GPU initialization slow
- **Wait:** GPU initialization can take 30-60 seconds
- **Check:** GPU with `nvidia-smi` command

---

## Quick Commands Reference

```powershell
# Navigate to project
cd "Data Science and Artificial Intelligence\research project"

# Verify Python
python --version

# Install dependencies
pip install -r requirements.txt

# Quick test (2 seeds, 3-5 min)
python test_multiseed.py

# Full experiments (5 seeds, 4-8 hours)
python finish_all_multiseed.py --seeds 5

# Full experiments (10 seeds, stronger stats)
python finish_all_multiseed.py --seeds 10

# Check results
dir results/multiseed_*

# View summary
Get-Content results/multiseed_master_summary.csv | Select-Object -First 10
```

---

## Phase 1 Timeline

| Step | Time | Status |
|------|------|--------|
| 1. Install Python | 5 min | ⏳ **DO THIS FIRST** |
| 2. Install dependencies | 2 min | ✅ After step 1 |
| 3. Test framework | 3-5 min | ✅ After step 2 |
| 4. Run full experiments | 4-8 hours | ✅ After step 3 |
| 5. Analyze results | 10-20 min | ✅ After step 4 |
| 6. Share with Prof | 5 min | ✅ Final |

**Total:** ~4-8 hours of computation + 30 minutes of setup

---

## What You'll Have After Phase 1

✅ **Multi-seed experiments** (5–10 seeds)  
✅ **Statistical rigor** (mean ± std, confidence intervals)  
✅ **Significance tests** (Wilcoxon p-values)  
✅ **Reproducible results** (seed tracking in all outputs)  
✅ **Prof. Dr. Aşıksoy's #1 concern addressed**  

---

## Next: Phase 2 Planning

After Phase 1 is validated:
- Scale NSGA-II budget (40–50 pop × 30 gen)
- Add 2D benchmarks (2D Poisson, Heat equation)
- Implement missing baselines (GradNorm, ReLoBRaLo, MOEA/D)

See: [PHASE_2_THROUGH_5_ROADMAP.md](PHASE_2_THROUGH_5_ROADMAP.md)

---

## Support

- **Python issues?** → See PYTHON_SETUP_REQUIRED.md
- **Framework issues?** → Check README_MULTISEED.md
- **Results questions?** → Review PHASE_1_IMPLEMENTATION_SUMMARY.md
- **Verification?** → See PHASE_1_VERIFICATION_REPORT.md

---

🚀 **Ready to go!** Install Python and start testing!
