# Classification System for Tumor Genetics of Glioblastoma Multiforme

Automated pipeline for IDH mutation classification in Glioblastoma Multiforme (GBM) using multiparametric MRI. Integrates HD-BET skull stripping, SegResNet BraTS23 tumor segmentation, PyRadiomics feature extraction, and two classification approaches: Machine Learning (Logistic Regression + Radiomics, AUC=0.914) and Deep Learning (MedicalNet ResNet-18, AUC=0.901). Includes a Streamlit web interface for clinical inference. Developed as part of a Biomedical Engineering undergraduate thesis at Pontificia Universidad Javeriana Cali.

---

## Pipeline

```
Imagen MRI (T1, T1GD, T2, FLAIR)
        │
        ▼
HD-BET (Skull Stripping)
        │
        ▼
SegResNet BraTS23 (Segmentación Whole Tumor)
        │
        ▼
PyRadiomics (1130 features → 30 seleccionadas)
        │
        ├──► Machine Learning (LR + Radiomics)    AUC = 0.9138
        │
        └──► Deep Learning (MedicalNet ResNet-18)  AUC = 0.9014
```

---

## Dataset

- **Fuente:** UPENN-GBM (The Cancer Imaging Archive)
- **Total casos:** 722 | **Pacientes únicos:** 681
- **Labels IDH:** 527 casos (458 wildtype, 69 mutado) — ratio 6.6:1
- **Hold-out:** 106 casos (92 wildtype, 14 mutado) — idéntico para ML y DL
- **Modalidades:** T1, T1GD (contraste), T2, FLAIR

---

## Modelos

### Machine Learning — Regresión Logística + Radiomics

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.9138 |
| F1-Score | 0.6471 |
| Sensibilidad | 0.7857 |
| Especificidad | 0.9022 |
| MCC | 0.5953 |
| TP / FP / TN / FN | 11 / 9 / 83 / 3 |

- Validación cruzada estratificada 5-fold sobre 421 casos de desarrollo
- 30 features radiómicas seleccionadas por ANOVA + Random Forest
- Umbral de clasificación: 0.5

---

### Deep Learning 01 — SegResNet BraTS23

Usado para segmentación del tumor (Whole Tumor) en todos los casos.

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.8012 |
| F1-Score | 0.4706 |
| Sensibilidad | 0.8571 |
| Especificidad | 0.7283 |
| MCC | 0.4158 |
| TP / FP / TN / FN | 12 / 25 / 67 / 2 |

---

### Deep Learning 02 — MedicalNet ResNet-18

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.9014 |
| F1-Score | 0.4364 |
| Sensibilidad | 0.8571 |
| Especificidad | 0.6848 |
| MCC | 0.3767 |
| TP / FP / TN / FN | 12 / 29 / 63 / 2 |

- Backbone ResNet-18 preentrenado con pesos MedicalNet
- Adaptado a 4 canales de entrada (T1, T1GD, T2, FLAIR)
- Umbral de clasificación: 0.5

---

## Estructura del repositorio

```
gbm-idh-classifier/
├── notebooks/
│   ├── SS_01_skull_stripping_.ipynb             # HD-BET skull stripping
│   ├── SS_02_segmentacion.ipynb                 # Segmentación SegResNet BraTS23
│   ├── SS_03_radiomics.ipynb                    # Extracción PyRadiomics
│   ├── SS_04_clasificador.ipynb                 # Clasificador ML
│   ├── DL_01_segresnet.ipynb                    # Clasificador DL SegResNet
│   └── DL_02_medicalnet.ipynb                   # Clasificador DL MedicalNet
├── Interfaz/
│   ├── SS_06_interfaz.ipynb                     # Notebook para lanzar interfaz
│   ├── app.py                                   # Interfaz Streamlit
│   └── gbm_pipeline.py                          # Pipeline de inferencia
├── Dataset/
│   └── labels_idh.csv                           # Labels IDH (527 casos)
├── manual_de_usuario.pdf                        # Manual de usuario
├── .gitignore
└── README.md
```

---

## Datos y modelos en Google Drive

Debido al tamaño de los archivos, los modelos entrenados, pesos preentrenados
y características radiómicas están disponibles en Google Drive.

### Carpeta completa GBM_TESIS_V2

Si deseas acceder a todos los archivos del proyecto y navegar libremente:

[Acceder a GBM_TESIS_V2 completa](https://drive.google.com/drive/folders/1skkD-oTI_QsDw4OWqod606icRTngug0I?usp=sharing)

---

### Modelos entrenados y características radiómicas

[Descargar modelos y radiomics](https://drive.google.com/drive/folders/1ZVJDkAqlf24cW39kAZ1d9h5PpPFTK7Xx?usp=sharing)

| Carpeta | Contenido |
|---------|-----------|
| `Modelos/clasificador_v2/models/` | Modelos ML: LR, RF, SVM, XGBoost, LightGBM (.joblib) |
| `Modelos/clasificador_v2/figures/` | Figuras de resultados ML |
| `Modelos/dl_clasificador_v2/medicalnet/models/` | `medicalnet_best.pth` (127 MB) |
| `Modelos/dl_clasificador_v2/segresnet/models/` | `segresnet_best.pth` (16.8 MB) |
| `Modelos/radiomics/features/` | `features_t1ce.csv` — 1130 features (722 casos) |

---

### Pesos preentrenados

[Descargar pesos preentrenados](https://drive.google.com/drive/folders/1Xubb35F2nw0z6aU707kTmpF2ZK_T_fZy?usp=sharing)

| Archivo | Destino en GBM_TESIS_V2 | Tamaño |
|---------|--------------------------|--------|
| `checkpoint_final.pth` | `Pesos preentrenados /hdbet_weights/release_2.0.0/fold_all/` | 117.8 MB |
| `resnet_18.pth` | `Pesos preentrenados /medicalnet_weights/` | 125.9 MB |
| `brats_mri_segmentation/` | `Pesos preentrenados /models/` | — |

---

## Instalación y uso

### Requisitos

```bash
pip install streamlit monai nibabel scipy numpy pandas
pip install torch torchvision scikit-learn joblib
pip install pyradiomics SimpleITK pydicom hd-bet
```

### Estructura de carpetas en Google Drive

```
GBM_TESIS_V2/
├── Dataset/
│   └── labels_idh.csv
├── Modelos/                        ← desde este repositorio + Drive modelos
│   ├── ss_radiomics_config.json
│   ├── clasificador_v2/
│   ├── dl_clasificador_v2/
│   ├── inference_pipeline_v2/
│   └── radiomics/
├── Pesos preentrenados /           ← desde Drive pesos
│   ├── hdbet_weights/
│   ├── medicalnet_weights/
│   └── models/
└── Interfaz/
    ├── app.py
    ├── gbm_pipeline.py
    ├── SS_06_interfaz.ipynb
    └── Manual_de_usuario/
        └── manual_de_usuario.pdf
```

### Ejecutar el pipeline completo

Corre los notebooks en orden desde Google Colab:

```
1. SS_01_skull_stripping_adaptado.ipynb
2. SS_02_segmentacion_adaptado.ipynb
3. SS_03_radiomics_adaptado.ipynb
4. SS_04_clasificador_corregido.ipynb
5. DL_01_segresnet_adaptado.ipynb
6. DL_02_medicalnet_adaptado.ipynb
```

### Lanzar la interfaz Streamlit

```
1. Abrir Interfaz/SS_06_interfaz.ipynb en Google Colab
2. Correr las celdas en orden
3. Copiar la URL generada por cloudflared
4. Abrir la URL en el navegador
```

---

## Tecnologías utilizadas

| Herramienta | Uso |
|-------------|-----|
| HD-BET | Skull stripping automático |
| SegResNet BraTS23 (MONAI) | Segmentación tumoral |
| PyRadiomics | Extracción de características radiómicas |
| Scikit-learn | Modelos de Machine Learning |
| MedicalNet ResNet-18 | Clasificación IDH con Deep Learning |
| PyTorch | Framework de Deep Learning |
| Streamlit | Interfaz web de inferencia |
| Google Colab | Entorno de ejecución con GPU |

---

## Autores

Desarrollado como proyecto de tesis de pregrado en Ingeniería Biomédica —
**Pontificia Universidad Javeriana Cali, Colombia**

---

## Aviso clínico

Esta herramienta ha sido desarrollada con fines exclusivamente investigativos.
Los resultados no constituyen un diagnóstico médico y no deben ser utilizados
como sustituto de la evaluación clínica por un profesional de la salud calificado.
