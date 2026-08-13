# Final Submission Checklist

## Files

- `Ayat_Elawej_EC499_Report_FINAL.docx`
- `Ayat_Elawej_EC499_Report_FINAL.pdf`
- Clean private repository pushed to `private-backup/main`
- Sanitized public repository containing no restricted acquisition tooling

## Report checks

- Final label population is 4,799 episodes and 1,121 faults.
- Full-outage totals are 2,398 events and 47 coordinated windows.
- July selected-HGB class distribution, confusion matrix, AUROC, and AUPRC are present.
- July is described as an out-of-time test with frozen model and threshold.
- Thermal evidence is described as an association with July false positives, not a proven cause.
- Reason codes are development-only; no July reason-code claim is made.
- Dashboard is described as implemented, read-only, and based on predicted-run segmentation.
- Controlled hardware testing is recorded as unavailable, not unfinished.
- Contents, figures, tables, and page numbers are refreshed.

## Repository checks

```powershell
python -m pytest -q
python -m compileall -q src scripts
git status --short
```

The final status should be clean after the final commit and push.

## Demonstration checks

```powershell
python -m streamlit run scripts/run_dashboard.py
```

- Network, Station, and Evidence tabs load.
- Play advances the simulated July clock without resetting the selected station.
- Full-outage health values are rounded and labelled as continued-outage projections.
- No dashboard action writes to `data/`.

## Manual items before sending

- Enter examiner names only if the department supplies them; otherwise leave the signature lines blank.
- Confirm the final submission date required by the department.
- Open the final PDF once on the submission computer and verify fonts and figures.
