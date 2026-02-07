# Krishna Paudel - Portfolio Website

Personal portfolio website with an integrated quantitative stock analysis platform.

## Structure

```
frontend/               # Static site (deployed to Vercel)
  index.html            # Main portfolio page
  quant_proj.html       # Stock analysis dashboard
  quant_app.js          # Dashboard JS (reads cached_results.json)
  cached_results.json   # Model output (auto-updated by GitHub Actions)
  contact.html          # Contact page
  projects.html         # Projects page
  *.html                # Other sub-pages
  *.png, *.pdf          # Assets

model.py                        # XGBoost stock prediction model
run_model_github_actions.py     # Script that runs model + saves JSON
requirements.txt                # Python deps (GitHub Actions only)
.github/workflows/
  daily_model_run.yml           # Cron: runs model daily, commits new JSON
```

## How it works

- **Vercel** serves the `frontend/` folder as a static site.
- **GitHub Actions** runs `run_model_github_actions.py` daily (cron), which executes `model.py`, generates predictions, and commits `frontend/cached_results.json`.
- The quant dashboard (`quant_proj.html`) loads `cached_results.json` on page load to render charts and predictions.
