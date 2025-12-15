# TTBB Harmonizer (AI-Assisted A Cappella Arranger)

A hybrid AI system that generates **TTBB (Tenor 1, Tenor 2, Baritone, Bass)** harmonizations for a given melody.  
The project combines:
- **Data-driven chord prediction** (trained on Bach chorales via `music21`)
- **Rule/constraint-based reasoning** for voice-leading quality
- **Heuristic search** to choose the best voicing at each timestep
- **Quantitative evaluation** of harmonization quality + baseline comparison

---

## Features

### 1) Learned Harmonic Motion (Markov Chord Model)
The system trains a chord transition model on Bach chorales and learns probabilities of moving from one Roman-numeral chord to the next.

### 2) TTBB Constraint System (Voice Leading + Ranges)
The harmonizer scores candidate voicings using:
- Voice range compliance (T1/T2/Bari/Bass)
- Voice crossing penalties (T1 ≥ T2 ≥ Bari ≥ Bass)
- Voice spacing penalties (prefer adjacent voices within ~octave)
- Parallel fifths / octaves penalties
- Smooth motion preference (penalizes large leaps)

### 3) Heuristic Search for Best Voicing
Given a chosen chord and the melody note, the system generates candidate pitches for T2/Bari/Bass and selects the lowest-penalty voicing.

### 4) Evaluation + Baseline
The evaluator reports:
- Range violations
- Voice crossing count
- Parallel fifths / octaves
- Consonance rate across vertical intervals
- Voice-leading smoothness score
- Chord “completeness” (distinct pitch classes)

Also compares the AI harmonizer against a simple baseline that stacks fixed intervals below the melody.

---

## Project Structure

- `main.py`  
  Runs training + generates harmonizations for multiple test melodies + writes outputs and summary JSON.

- `harmonizer.py`  
  Core harmonizer implementation:
  - `ChordTransitionModel`
  - `VoiceLeadingConstraints`
  - `HarmonizationSearch`
  - `TTBBHarmonizer`

- `evaluation.py`  
  Evaluation framework and baseline comparison:
  - `HarmonizationEvaluator`
  - `generate_baseline_harmonization`
  - `compare_with_baseline`

- `outputs/` (auto-created)  
  Stores generated MIDI, MusicXML, and JSON evaluation artifacts.

---

## Requirements

- Python 3.9+ recommended
- `music21`

Install:
```bash
pip install music21

