# ECG Signal Analysis

Modular Python workflows for ECG anomaly detection and arrhythmia classification.
Designed with object-oriented components for preprocessing, feature extraction, and modeling.
Built around the MIT-BIH Arrhythmia Database (AAMI 5-class labels).

---

## ✨ Features

* Preprocessing: bandpass filtering, resampling, baseline wander removal
* Beat segmentation: R-peak detection (NeuroKit2) and fixed-window extraction
* Feature extraction: morphology, time-domain HRV, wavelet/frequency features
* Anomaly detection: unsupervised workflows (Isolation Forest, autoencoders)
* Classification: supervised workflows (LogReg, LightGBM, CNNs)
* Extensible design: add new FeatureExtractor or Model classes
* Dataset ready: MIT-BIH Arrhythmia with AAMI N/S/V/F/Q classes

---

## ⚙️ Installation

git clone [https://github.com/yourname/ecg-signal-analysis.git](https://github.com/yourname/ecg-signal-analysis.git)</br>
cd ecg-signal-analysis</br>
pip install -r requirements.txt

---

## 🚀 Usage

1. Download MIT-BIH dataset
   python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mitdb')"

2. Prepare beats
   python scripts/build_beats_mitbih.py

3. Train anomaly detector
   python scripts/train_anomaly.py

4. Train arrhythmia classifier
   python scripts/train_classify.py

---

## 📊 Example Results

* Anomaly detection: AUROC ≈ 0.9 on MIT-BIH (normal vs abnormal beats)
* Classification: macro-F1 ≈ 0.75 with logistic baseline features

(Results depend on exact splits and preprocessing.)

---

## 🤝 Contributing

PRs welcome!
Ideas: add deep CNN/LSTM models, new feature extractors, or additional datasets (e.g., PTB-XL).

---

## ⚠️ Disclaimer

This repository is for research and educational purposes only.
It is not a certified medical device and must not be used for clinical decision making.

---
