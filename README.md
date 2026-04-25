GBM IDH Classifier
Automated pipeline for IDH mutation classification in Glioblastoma Multiforme (GBM) using multiparametric MRI. Integrates HD-BET skull stripping, SegResNet BraTS23 tumor segmentation, PyRadiomics feature extraction, and two classification approaches: Machine Learning (Logistic Regression + Radiomics, AUC=0.914) and Deep Learning (MedicalNet ResNet-18, AUC=0.901). Includes a Streamlit web interface for clinical inference. Developed as part of a Biomedical Engineering doctoral thesis at Pontificia Universidad Javeriana Cali.

Pipeline
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

Dataset

Fuente: UPENN-GBM (The Cancer Imaging Archive)
Total casos: 722 | Pacientes únicos: 681
Labels IDH: 527 casos (458 wildtype, 69 mutado) — ratio 6.6:1
Hold-out: 106 casos (92 wildtype, 14 mutado) — idéntico para ML y DL
Modalidades: T1, T1GD (contraste), T2, FLAIR


Modelos
Machine Learning — Regresión Logística + Radiomics
MétricaValorAUC-ROC0.9138F1-Score0.6471Sensibilidad0.7857Especificidad0.9022MCC0.5953TP / FP / TN / FN11 / 9 / 83 / 3

Validación cruzada estratificada 5-fold sobre 421 casos de desarrollo
30 features radiómicas seleccionadas por ANOVA + Random Forest
Umbral de clasificación: 0.5


Deep Learning 01 — SegResNet BraTS23
Usado para segmentación del tumor (Whole Tumor) en todos los casos.
MétricaValorAUC-ROC0.8012F1-Score0.4706Sensibilidad0.8571Especificidad0.7283MCC0.4158TP / FP / TN / FN12 / 25 / 67 / 2

Deep Learning 02 — MedicalNet ResNet-18
MétricaValorAUC-ROC0.9014F1-Score0.4364Sensibilidad0.8571Especificidad0.6848MCC0.3767TP / FP / TN / FN12 / 29 / 63 / 2

Backbone ResNet-18 preentrenado con pesos MedicalNet
Adaptado a 4 canales de entrada (T1, T1GD, T2, FLAIR)
Umbral de clasificación: 0.5


Estructura del repositorio
gbm-idh-classifier/
├── SS_01_skull_stripping_adaptado.ipynb     # HD-BET skull stripping
├── SS_02_segmentacion_adaptado.ipynb        # Segmentación SegResNet BraTS23
├── SS_03_radiomics_adaptado.ipynb           # Extracción PyRadiomics
├── SS_04_clasificador_corregido.ipynb       # Clasificador ML
├── DL_01_segresnet_adaptado.ipynb           # Clasificador DL SegResNet
├── DL_02_medicalnet_adaptado.ipynb          # Clasificador DL MedicalNet
├── Interfaz/
│   ├── SS_06_interfaz.ipynb                 # Notebook para lanzar interfaz
│   ├── app.py                               # Interfaz Streamlit
│   └── gbm_pipeline.py                     # Pipeline de inferencia
├── Modelos/
│   ├── ss_radiomics_config.json             # Configuración features
│   ├── clasificador_v2/
│   │   └── results/
│   │       ├── cv_results.csv               # Resultados cross-validation
│   │       ├── holdout_ids.csv              # IDs hold-out ML
│   │       └── holdout_metrics.csv          # Métricas hold-out ML
│   ├── inference_pipeline_v2/
│   │   ├── best_model.joblib                # Mejor modelo ML (LR)
│   │   └── pipeline_config.json            # Config inferencia ML
│   ├── dl_clasificador_v2/
│   │   ├── dl_train_ids.csv                 # Split entrenamiento DL
│   │   ├── dl_val_ids.csv                   # Split validación DL
│   │   ├── dl_holdout_ids.csv               # Split hold-out DL
│   │   ├── medicalnet/
│   │   │   └── medicalnet_config_v2.json    # Config MedicalNet
│   │   └── segresnet/
│   │       └── segresnet_config_v2.json     # Config SegResNet
│   └── radiomics/
│       ├── params/
│       │   └── radiomics_params.yaml        # Parámetros PyRadiomics
│       └── selected_features/
│           └── features_selected.csv        # 30 features seleccionadas
├── Dataset/
│   └── labels_idh_corregido.csv             # Labels IDH (527 casos)
├── .gitignore
└── README.md

Datos y modelos en Google Drive
Debido al tamaño de los archivos, los modelos entrenados, pesos preentrenados
y características radiómicas están disponibles en Google Drive.
Modelos entrenados y características radiómicas
Descargar modelos y radiomics
CarpetaContenidoModelos/clasificador_v2/models/Modelos ML: LR, RF, SVM, XGBoost, LightGBM (.joblib)Modelos/clasificador_v2/figures/Figuras de resultados MLModelos/dl_clasificador_v2/medicalnet/models/medicalnet_best.pth (127 MB)Modelos/dl_clasificador_v2/segresnet/models/segresnet_best.pth (16.8 MB)Modelos/radiomics/features/features_t1ce.csv — 1130 features (722 casos)

Pesos preentrenados
Descargar pesos preentrenados
ArchivoDestino en GBM_TESIS_V2Tamañocheckpoint_final.pthPesos preentrenados /hdbet_weights/release_2.0.0/fold_all/117.8 MBresnet_18.pthPesos preentrenados /medicalnet_weights/125.9 MBbrats_mri_segmentation/Pesos preentrenados /models/—

Instalación y uso
Requisitos
bashpip install streamlit monai nibabel scipy numpy pandas
pip install torch torchvision scikit-learn joblib
pip install pyradiomics SimpleITK pydicom hd-bet
Estructura de carpetas en Google Drive
GBM_TESIS_V2/
├── Dataset/
│   └── labels_idh_corregido.csv
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
    └── SS_06_interfaz.ipynb
Ejecutar el pipeline completo
Corre los notebooks en orden desde Google Colab:
1. SS_01_skull_stripping_adaptado.ipynb
2. SS_02_segmentacion_adaptado.ipynb
3. SS_03_radiomics_adaptado.ipynb
4. SS_04_clasificador_corregido.ipynb
5. DL_01_segresnet_adaptado.ipynb
6. DL_02_medicalnet_adaptado.ipynb
Lanzar la interfaz Streamlit
1. Abrir Interfaz/SS_06_interfaz.ipynb en Google Colab
2. Correr las celdas en orden
3. Copiar la URL generada por cloudflared
4. Abrir la URL en el navegador

Tecnologías utilizadas
HerramientaUsoHD-BETSkull stripping automáticoSegResNet BraTS23 (MONAI)Segmentación tumoralPyRadiomicsExtracción de características radiómicasScikit-learnModelos de Machine LearningMedicalNet ResNet-18Clasificación IDH con Deep LearningPyTorchFramework de Deep LearningStreamlitInterfaz web de inferenciaGoogle ColabEntorno de ejecución con GPU

Autores
Desarrollado como proyecto de tesis doctoral en Ingeniería Biomédica —
Pontificia Universidad Javeriana Cali, Colombia

Aviso clínico
Esta herramienta ha sido desarrollada con fines exclusivamente investigativos.
Los resultados no constituyen un diagnóstico médico y no deben ser utilizados
como sustituto de la evaluación clínica por un profesional de la salud calificado.

Aviso clínico
Esta herramienta ha sido desarrollada con fines exclusivamente investigativos.
Los resultados no constituyen un diagnóstico médico y no deben ser utilizados
como sustituto de la evaluación clínica por un profesional de la salud calificado.
