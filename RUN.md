# How to run this project

Every command below is meant to be pasted into the **VS Code integrated terminal**
(`` Ctrl+` `` to open it). Windows PowerShell is the default on Windows; the
macOS/Linux equivalent is given wherever it differs.

> **Prefer not to install anything?** The portfolio site has a **Run this project**
> button that opens this repository in a free GitHub Codespace, installs the
> dependencies and runs the whole pipeline for you:
> <https://nikhil201716.github.io/nikhil-data-portfolio/pages/project.html?id=15>

---

## 1. Prerequisites

```powershell
python --version    # 3.11 or newer
git --version
```

### Optional — the local LLM stages

This project has stages that use a local model through [Ollama](https://ollama.com). They are **optional**: without it those stages are skipped or fall back to their deterministic control arm, and every other stage runs normally.

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:0.5b
ollama list
```

---

---

## 2. One-time setup

```powershell
git clone https://github.com/Nikhil201716/15-Aegis-Health-Plan-Platform.git
cd 15-Aegis-Health-Plan-Platform
```

Create and activate a virtual environment. This keeps the project's dependencies
from colliding with anything else on your machine.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

<details>
<summary>If PowerShell refuses to run the activation script</summary>

Windows blocks unsigned scripts by default. Allow them for your own user account only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
</details>

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```powershell
pip install -r requirements.txt
```

> **Tip:** with the venv active, VS Code will offer to select it as the interpreter.
> Accept — otherwise the Run and Debug buttons use your global Python and you get
> `ModuleNotFoundError`.

---

## 3. Run it

### Everything, in one command

```powershell
python run_all.py
```

That runs every stage below in order and is the normal way to use this repository.

### Or one stage at a time

Useful when you are changing a single stage and do not want to rebuild everything.

| # | Command | What it does |
|---|---|---|
| 1 | `python platform/ingest/generate_payer_data.py` | Generate payer data |
| 2 | `python platform/warehouse/build_warehouse.py` | Build warehouse |
| 3 | `python integrity/metric_regression.py` | Metric regression testing |
| 4 | `python analytics/survival/member_survival.py` | Member survival analysis |
| 5 | `python analytics/forecasting/hierarchical_forecast.py` | Hierarchical forecasting |
| 6 | `python integrity/upcoding_detection.py` | Upcoding + red team |
| 7 | `python analytics/risk/risk_adjustment.py` | Risk adjustment |
| 8 | `python ai/metric_copilot.py` | Metric copilot (local LLM) |
| 9 | `python qa/validation.py` | Validation & injected bugs |

Each stage produces the input the next one consumes, so run them in this order.


---

## 4. Explore the results

```powershell
python app/api/main.py
```

VS Code will offer to forward the port and open it in your browser.

The pipeline writes everything it measures into `reports/`. Those files are the
source of every number quoted on the portfolio site — nothing is typed by hand.

```powershell
ls reports
```

---

## 5. What a correct run looks like

Upcoding recall should collapse from about 0.95 to about 0.18 under the reduced_rate scenario. The metric regression should report 33 of 50 figures moved, and the traceability matrix 11 of 12 with REQ-012 named as the gap.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | the virtual environment is not active | re-run the activate command from step 2 |
| `FileNotFoundError` on a data file | an earlier stage was skipped | run the stages in the documented order, or use the one-command runner |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Numbers differ from the README | a seed or parameter changed | check the constants at the top of the generator script |
| `command not found` | dependency missing from this environment | `pip install -r requirements.txt` with the venv active |
| VS Code runs the wrong Python | interpreter not selected | `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `.venv` |

---

## 7. Finish

```powershell
deactivate
```

---

## More

- **The 60+ page technical notebook** for this project is in [`docs/`](docs/) — it
  covers the business problem, the mathematics derived from first principles, a
  guided tour of the code, worked numerical examples and exercises with solutions.
- **All fifteen projects:** <https://nikhil201716.github.io/nikhil-data-portfolio/>

*Generated from this repository's own pipeline runner, so the stage list cannot
drift from the code.*
