# Wearable stress detection: how much of the accuracy is real?

A reproducible validation audit of stress-vs-baseline classification on **WESAD** wrist signals
(Empatica E4: BVP, EDA, skin temperature, accelerometer).

The same features and the same models are evaluated three ways. Only the validation protocol changes:

| Protocol | What it tests | Typical failure it exposes |
|---|---|---|
| **Random window k-fold** | Windows shuffled across folds | Overlapping windows from the same person sit in both train and test. The model can memorise the person, not the stress response. |
| **Leave-one-subject-out (LOSO)** | Train on 14 people, test on an unseen one | This is the number that predicts performance on a new user. |
| **LOSO + per-subject standardization** | Each person's features standardized on their own data | Quantifies what a short per-user calibration period buys. It is a deployment assumption, not free accuracy. |

The pipeline also reports, for the random k-fold protocol, the **share of test windows that overlap in time with a training window from the same subject**. That single number is often enough to explain an inflated result.

## Run it

1. Download WESAD from its official page (University of Siegen, ~2 GB) and unzip it, so that `WESAD/S2/S2.pkl` … `WESAD/S17/S17.pkl` exist.
2. Install and run:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m wesad_audit.run --data /path/to/WESAD --out results
```

Defaults: 60 s windows, 5 s step, wrist signals only. Change them with `--win` and `--step`.

Outputs in `results/`:

- `protocol_scores.csv`: balanced accuracy and macro-F1 per model and protocol, plus the train/test overlap rate
- `per_subject_scores.csv`: LOSO score for every held-out subject
- `loso_permutation_importance.csv`: which features still matter on people the model has never seen
- `fig_protocols.png`, `fig_per_subject.png`, `fig_importance.png`
- `run_info.json`: dataset, window settings, library versions, timestamp

## Features (29 per window)

- **BVP**: band-pass 0.7–3.5 Hz, peak detection → inter-beat intervals → HR mean/std, SDNN, RMSSD, pNN50; HRV LF/HF power from the IBI series resampled at 4 Hz (Welch).
- **EDA**: low-pass 1 Hz; level, range, slope; phasic component (signal minus ~8 s median tonic); SCR rate.
- **Temperature**: mean, std, slope.
- **Accelerometer**: magnitude statistics, band power 0–3 Hz and 3–8 Hz, dominant frequency, per-axis std.

## Smoke test without WESAD

```bash
python tests/make_synthetic.py /tmp/synth
PYTHONPATH=src python -m wesad_audit.run --data /tmp/synth --out /tmp/synth_out
```

The synthetic subjects only check that the code runs end to end. Their numbers say nothing about WESAD, and `run_info.json` marks them as synthetic.

## Caveats

- Per-subject standardization uses all of a test subject's windows, including their class balance. A real device would calibrate on a separate rest period; treat this protocol as an upper bound.
- Labels are the WESAD protocol conditions (baseline vs TSST stress), not momentary self-reports.
- Wrist signals only. Chest (RespiBAN) signals are ignored on purpose, since most products ship a wrist device.

## Results

_Filled in from `results/` after the run on the real WESAD data._

## Reference

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., Van Laerhoven, K. (2018). *Introducing WESAD, a multimodal dataset for wearable stress and affect detection.* ICMI 2018.

## Author

Haithem Lamouchi, PhD in Mathematics (time-frequency analysis). I audit signal and time-series ML pipelines: validation, leakage, and representation choices.
