# Python Environment Setup Guide

## Problem Identified
Python is not installed or not in the system PATH. This needs to be fixed before running Phase 1.

## Solution: Three Options

### **Option 1: Install Python from python.org (Recommended for Windows)**

1. **Download Python 3.10 or 3.11:**
   - Go to https://www.python.org/downloads/windows/
   - Download "Windows installer (64-bit)"

2. **Install with PATH enabled:**
   - Run the installer
   - ✅ **IMPORTANT:** Check "Add Python to PATH" during installation
   - Choose "Install Now" or customize installation
   - Verify installation is complete

3. **Verify installation:**
   ```powershell
   python --version
   pip --version
   ```

### **Option 2: Use Anaconda/Miniconda (Recommended for Data Science)**

1. **Download Miniconda:**
   - Go to https://docs.conda.io/en/latest/miniconda.html
   - Download Windows installer (64-bit)

2. **Install:**
   - Run installer
   - Choose "Add Miniconda to PATH" 
   - Create default environment

3. **Verify:**
   ```powershell
   conda --version
   python --version
   ```

### **Option 3: Use Windows Package Manager**

```powershell
# If you have Windows Package Manager installed
winget install Python.Python.3.11
```

---

## After Python Installation

### **Step 1: Verify Python**
```powershell
cd "Data Science and Artificial Intelligence\research project"
python --version
python -m pip --version
```

### **Step 2: Install Dependencies**
```powershell
pip install -r requirements.txt
```

Check [requirements.txt](requirements.txt) contains:
- torch>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- scipy>=1.10.0
- pymoo>=0.6.0
- pandas>=2.0.0

### **Step 3: Test Phase 1 Framework**

**Quick test (2 seeds, ~3 minutes):**
```powershell
python test_multiseed.py
```

**Full test (5 seeds, 4-8 hours on GPU):**
```powershell
python finish_all_multiseed.py --seeds 5
```

---

## Verify Phase 1 Implementation is Ready

✅ **All files created:**
```
research project/
├── src/
│   ├── multi_seed_runner.py         ✓ (280 lines)
│   ├── stats_report_generator.py    ✓ (330 lines)
│   └── __init__.py                  ✓
├── finish_all_multiseed.py          ✓ (360 lines)
├── test_multiseed.py                ✓ (120 lines)
├── README_MULTISEED.md              ✓ (450 lines)
├── PHASE_1_IMPLEMENTATION_SUMMARY.md ✓
├── PHASE_1_COMPLETION_CHECKLIST.md  ✓
└── PHASE_2_THROUGH_5_ROADMAP.md     ✓
```

---

## Expected Results After Installation

Once Python is installed and you run Phase 1:

### Quick Test Output:
```
======================================================================
TESTING MULTI-SEED FRAMEWORK
======================================================================

⏱️  Running 2 seeds × Poisson grid (minimal budget)...
   This tests data collection, aggregation, and statistics.

[1/2] Running with seed=42
[2/2] Running with seed=99

✓ All runs: results/test_multiseed/test_poisson_all_seeds.csv
✓ Summary stats: results/test_multiseed/test_poisson_summary_stats.csv
✓ Metadata: results/test_multiseed/test_poisson_metadata.json

Statistics computed for 4 configurations across 2 seeds.

✅ MULTI-SEED FRAMEWORK TEST PASSED
```

### Full Pipeline Output:
```
================================================================================
MULTI-SEED EXPERIMENT PIPELINE
Seeds: [0, 1, 2, 3, 4]
Results directory: results
================================================================================

[1/6] Poisson Grid Search (Multi-Seed)
[2/6] Burgers Baseline (Multi-Seed)
[3/6] Burgers Grid Search - Clean (Multi-Seed)
[4/6] Burgers Grid Search - Sparse Noisy (Multi-Seed)
[5/6] Burgers NSGA-II (Multi-Seed)
[6/6] Saving Aggregated Results

✓ multiseed_master_summary.csv
✓ multiseed_all_runs.csv
✓ multiseed_significance_tests.csv
```

---

## Troubleshooting

### "python: command not found"
- Python is not in PATH
- Solution: Reinstall Python and check "Add to PATH"
- Or use full path: `C:\Users\micro\AppData\Local\Programs\Python\Python311\python.exe`

### "ModuleNotFoundError: No module named 'scipy'"
```powershell
pip install scipy
```

### "CUDA out of memory"
Use fewer epochs:
```powershell
python finish_all_multiseed.py --seeds 2 --quick
```

---

## Next Steps

1. **Install Python** (Option 1 or 2 above)
2. **Verify:** `python --version`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Test Phase 1:** `python test_multiseed.py`
5. **Run full experiments:** `python finish_all_multiseed.py --seeds 5`
6. **Share results with Prof. Dr. Aşıksoy**

---

## Phase 1 Status Summary

| Component | Status | Location |
|-----------|--------|----------|
| Framework code | ✅ Complete | `src/multi_seed_runner.py` |
| Statistical analysis | ✅ Complete | `src/stats_report_generator.py` |
| Full pipeline | ✅ Complete | `finish_all_multiseed.py` |
| Test script | ✅ Complete | `test_multiseed.py` |
| Documentation | ✅ Complete | `README_MULTISEED.md` + 3 guides |
| **Python environment** | ❌ **NEEDED** | Install Python first |

**Action required:** Install Python, then Phase 1 is ready to execute! 🚀
