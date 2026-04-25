"""
app.py - Interfaz Streamlit GBM IDH Classifier
Importa GBMPipeline desde gbm_pipeline.py.

Ubicacion: GBM_TESIS/Interfaz/app.py
"""

import sys
import io
import shutil
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# Rutas adaptadas a la estructura de GBM_TESIS
# app.py y gbm_pipeline.py estan en: GBM_TESIS/Interfaz/
# ============================================================
BASE_DRIVE   = Path('/content/drive/MyDrive/GBM_TESIS')
INTERFAZ_DIR = BASE_DRIVE / 'Interfaz'
sys.path.insert(0, str(INTERFAZ_DIR))
from gbm_pipeline import GBMPipeline

WORK_DIR = Path('/content/work_streamlit')
WORK_DIR.mkdir(exist_ok=True)

_KEYWORDS_MOD = {
    't1ce': ['t1ce', 't1gd', 't1c', 'postcontrast', 'post_contrast',
             'contrast', 'gd', 'gadolinium', 'ce', 't1+c'],
    't1':   ['t1w', 't1_', '_t1', 'mprage', 'spgr', 'precontrast',
             'pre_contrast', 't1pre'],
    't2':   ['t2w', 't2_', '_t2', 't2star', 'fse', 'tse'],
    'flair':['flair', 'fl_', '_fl', 'flr', 'fluid'],
}

st.set_page_config(
    page_title='GBM IDH Classifier',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f4f6f9; }
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 2.5rem 3rem; border-radius: 16px; margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .header-title { color:#ffffff; font-size:1.9rem; font-weight:700; margin:0; letter-spacing:-0.5px; }
    .header-subtitle { color:#94a3b8; font-size:0.95rem; margin-top:0.4rem; font-weight:300; }
    .header-meta { margin-top:1.2rem; display:flex; gap:0.75rem; flex-wrap:wrap; }
    .badge { background:rgba(255,255,255,0.08); color:#cbd5e1; padding:0.25rem 0.85rem;
             border-radius:20px; font-size:0.75rem; font-weight:500;
             border:1px solid rgba(255,255,255,0.12); letter-spacing:0.3px; }
    .result-mutado {
        background:linear-gradient(135deg,#fef2f2 0%,#fee2e2 100%);
        border:1.5px solid #fca5a5; border-radius:14px;
        padding:2rem 2.5rem; text-align:center; margin:1.2rem 0;
    }
    .result-wildtype {
        background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);
        border:1.5px solid #86efac; border-radius:14px;
        padding:2rem 2.5rem; text-align:center; margin:1.2rem 0;
    }
    .result-indicator { display:inline-block; width:12px; height:12px; border-radius:50%; margin-bottom:0.8rem; }
    .result-indicator-mutado   { background:#ef4444; box-shadow:0 0 12px #ef4444; }
    .result-indicator-wildtype { background:#22c55e; box-shadow:0 0 12px #22c55e; }
    .result-label { font-size:1.9rem; font-weight:700; letter-spacing:-0.5px; margin:0; }
    .result-mutado .result-label   { color:#b91c1c; }
    .result-wildtype .result-label { color:#15803d; }
    .result-conf  { font-size:1rem; margin-top:0.4rem; color:#64748b; font-weight:400; }
    .result-model { font-size:0.78rem; color:#94a3b8; margin-top:0.5rem;
                    text-transform:uppercase; letter-spacing:0.6px; }
    .metrics-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:0.75rem; margin-top:1.2rem; }
    .metric-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;
                   padding:1rem; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
    .metric-value { font-size:1.35rem; font-weight:700; color:#0f172a; }
    .metric-label { font-size:0.68rem; color:#94a3b8; margin-top:0.2rem;
                    text-transform:uppercase; letter-spacing:0.7px; font-weight:500; }
    .upload-label { font-weight:600; color:#334155; font-size:0.85rem; margin-bottom:0.25rem; }
    .mod-detected { font-size:0.75rem; color:#15803d; font-weight:500; margin-top:0.2rem; }
    .mod-warning  { font-size:0.75rem; color:#b45309; font-weight:500; margin-top:0.2rem; }
    .slicer-label { font-size:0.78rem; color:#64748b; margin-bottom:0.2rem;
                    text-transform:uppercase; letter-spacing:0.5px; font-weight:500; }
    .confidence-bar-wrap { margin:1rem 0 0.5rem 0; }
    .confidence-label { font-size:0.78rem; color:#64748b; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.4rem; }
    .confidence-high   { color:#15803d; font-weight:700; font-size:0.85rem; }
    .confidence-medium { color:#b45309; font-weight:700; font-size:0.85rem; }
    .confidence-low    { color:#b91c1c; font-weight:700; font-size:0.85rem; }
    .disclaimer { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px;
                  padding:0.75rem 1rem; font-size:0.78rem; color:#92400e;
                  margin-top:1.5rem; line-height:1.5; }
    .serie-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;
                  padding:1rem; margin-bottom:0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap:6px; background:transparent; }
    .stTabs [data-baseweb="tab"] { background:#ffffff; border-radius:8px; border:1px solid #e2e8f0;
        padding:0.45rem 1.4rem; font-weight:600; font-size:0.85rem; color:#475569; }
    .stTabs [aria-selected="true"] { background:#0f172a !important; color:#ffffff !important;
        border-color:#0f172a !important; }
    .stButton > button { background:#0f172a; color:white; border:none; border-radius:10px;
        padding:0.65rem 2rem; font-weight:600; font-size:0.95rem; width:100%; letter-spacing:0.2px; }
    .stButton > button:hover { background:#1e3a5f; box-shadow:0 4px 14px rgba(15,23,42,0.2); }
    .empty-state { text-align:center; padding:5rem 2rem; color:#94a3b8; }
    .empty-state-title { font-size:1.1rem; font-weight:600; color:#64748b; margin-bottom:0.5rem; }
    .empty-state-text  { font-size:0.88rem; }
    #MainMenu, footer, header { visibility:hidden; }
    .block-container { padding-top:1.5rem; max-width:1100px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Utilidades
# ============================================================

@st.cache_resource(show_spinner='Inicializando modelos...')
def cargar_pipeline():
    return GBMPipeline()


def detectar_modalidad(nombre_archivo: str) -> str:
    nombre = nombre_archivo.lower().replace('-', '_').replace(' ', '_')
    for ext in ['.nii.gz', '.nii', '.dcm', '.zip']:
        if nombre.endswith(ext):
            nombre = nombre[:-len(ext)]
            break
    mapeo_exacto = {
        't1ce': 't1ce', 't1gd': 't1ce', 't1c': 't1ce', 'ce': 't1ce',
        't1':   't1',
        't2':   't2',
        'flair': 'flair', 'fl': 'flair',
    }
    if nombre in mapeo_exacto:
        return mapeo_exacto[nombre]
    for mod in ['t1ce', 'flair', 't2', 't1']:
        for kw in _KEYWORDS_MOD[mod]:
            if kw in nombre:
                return mod
    return None


def save_upload(uploaded_file, work_dir: Path, name: str) -> str:
    path = work_dir / name
    with open(path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return str(path)


def generar_reporte_pdf(resultado: dict, t1ce_path: str,
                         mask_data: np.ndarray) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_buffer = io.BytesIO()
    t1ce_data  = nib.load(t1ce_path).get_fdata()
    best_sl    = int(np.argmax(mask_data.sum(axis=(0, 1))))

    t1ce_sl   = t1ce_data[:, :, best_sl]
    t1ce_norm = (t1ce_sl - t1ce_sl.min()) / (t1ce_sl.max() - t1ce_sl.min() + 1e-8)
    mask_sl   = mask_data[:, :, best_sl]
    overlay   = np.stack([t1ce_norm] * 3, axis=-1)
    color     = [0.75, 0.1, 0.1] if resultado['prediccion'] == 'mutado' else [0.1, 0.45, 0.8]
    for c_idx, c_val in enumerate(color):
        overlay[:, :, c_idx] = np.clip(overlay[:, :, c_idx] + mask_sl * 0.45 * c_val, 0, 1)

    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(11, 8.5), facecolor='white')
        fig.text(0.5, 0.95, 'GBM IDH Classifier — Reporte de Analisis',
                 ha='center', va='top', fontsize=14, fontweight='700', color='#0f172a')
        fig.text(0.5, 0.91,
                 f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}  '
                 f'|  Modelo: {resultado["modelo"]}',
                 ha='center', va='top', fontsize=9, color='#64748b')

        color_res = '#b91c1c' if resultado['prediccion'] == 'mutado' else '#15803d'
        label_res = 'IDH Mutado' if resultado['prediccion'] == 'mutado' else 'IDH Wildtype'
        prob_pred = (resultado['prob_mutado']
                     if resultado['prediccion'] == 'mutado'
                     else resultado['prob_wildtype'])

        fig.text(0.5, 0.83, label_res,
                 ha='center', va='top', fontsize=20, fontweight='700', color=color_res)
        fig.text(0.5, 0.78,
                 f'Confianza: {prob_pred*100:.1f}%  '
                 f'|  P(mutado): {resultado["prob_mutado"]*100:.1f}%  '
                 f'|  P(wildtype): {resultado["prob_wildtype"]*100:.1f}%',
                 ha='center', va='top', fontsize=10, color='#334155')
        fig.text(0.5, 0.73,
                 f'AUC del modelo: {resultado["auc_modelo"]:.4f}',
                 ha='center', va='top', fontsize=9, color='#475569')

        ax1 = fig.add_axes([0.05, 0.15, 0.42, 0.50])
        ax1.imshow(t1ce_norm.T, cmap='gray', origin='lower')
        ax1.set_title(f'T1ce original  —  Slice axial {best_sl}',
                      fontsize=9, color='#334155', pad=8)
        ax1.axis('off')

        ax2 = fig.add_axes([0.53, 0.15, 0.42, 0.50])
        ax2.imshow(overlay.transpose(1, 0, 2), origin='lower')
        patch = mpatches.Patch(color=color_res, label='Region tumoral')
        ax2.legend(handles=[patch], loc='lower right', fontsize=7,
                   facecolor='white', edgecolor='#e2e8f0')
        ax2.set_title('Segmentacion tumoral — SegResNet BraTS23',
                      fontsize=9, color='#334155', pad=8)
        ax2.axis('off')

        fig.text(0.5, 0.06,
                 'AVISO: Este reporte ha sido generado por una herramienta de investigacion. '
                 'No constituye diagnostico medico.',
                 ha='center', va='bottom', fontsize=7, color='#92400e', style='italic')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer.read()


def indicador_confianza(prob_pred: float, prediccion: str) -> str:
    """Retorna HTML del indicador de confianza segun distancia al umbral."""
    distancia = abs(prob_pred - 0.5)
    if distancia >= 0.35:
        nivel     = 'Alta'
        css_class = 'confidence-high'
        descripcion = f'El modelo clasifica este caso como {prediccion} con alta certeza (P={prob_pred*100:.1f}%).'
    elif distancia >= 0.15:
        nivel     = 'Moderada'
        css_class = 'confidence-medium'
        descripcion = f'El modelo clasifica este caso como {prediccion} con certeza moderada (P={prob_pred*100:.1f}%). Se recomienda revisión clínica complementaria.'
    else:
        nivel     = 'Baja — Zona de incertidumbre'
        css_class = 'confidence-low'
        descripcion = f'La probabilidad ({prob_pred*100:.1f}%) está cerca del umbral de decisión (50%). El resultado debe interpretarse con precaución.'
    return f'''
    <div class="confidence-bar-wrap">
        <p class="confidence-label">Confianza de la predicción</p>
        <p class="{css_class}">{nivel}</p>
        <p style="font-size:0.82rem; color:#475569; margin-top:0.2rem;">{descripcion}</p>
    </div>'''


def mostrar_slicer(t1ce_path: str, mask_data: np.ndarray, resultado: dict):
    from scipy.ndimage import zoom as _zoom
    t1ce_data = nib.load(t1ce_path).get_fdata()

    # Alinear mascara al shape de t1ce si son diferentes
    if t1ce_data.shape != mask_data.shape:
        factores  = tuple(t / m for t, m in zip(t1ce_data.shape, mask_data.shape))
        mask_data = _zoom(mask_data, factores, order=0).astype(np.uint8)

    n_slices  = t1ce_data.shape[2]
    best_sl   = int(np.argmax(mask_data.sum(axis=(0, 1))))

    st.markdown('<p class="slicer-label">Navegacion de slices axiales</p>',
                unsafe_allow_html=True)

    sl = st.slider(
        label='slice_axial',
        min_value=0,
        max_value=n_slices - 1,
        value=best_sl,
        step=1,
        format='Slice %d',
        label_visibility='collapsed',
    )

    t1ce_sl   = t1ce_data[:, :, sl]
    t1ce_norm = (t1ce_sl - t1ce_sl.min()) / (t1ce_sl.max() - t1ce_sl.min() + 1e-8)
    mask_sl   = mask_data[:, :, sl]

    overlay = np.stack([t1ce_norm] * 3, axis=-1)
    color   = [0.75, 0.1, 0.1] if resultado['prediccion'] == 'mutado' else [0.1, 0.45, 0.8]
    for c_idx, c_val in enumerate(color):
        overlay[:, :, c_idx] = np.clip(
            overlay[:, :, c_idx] + mask_sl * 0.45 * c_val, 0, 1
        )

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        fig1, ax1 = plt.subplots(figsize=(5, 5), facecolor='white')
        ax1.imshow(t1ce_norm.T, cmap='gray', origin='lower')
        ax1.set_title(f'T1ce original  —  Slice {sl}',
                      fontsize=10, color='#334155', pad=10, fontweight='600')
        ax1.axis('off')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_img2:
        fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor='white')
        ax2.imshow(overlay.transpose(1, 0, 2), origin='lower')
        color_hex = '#b91c1c' if resultado['prediccion'] == 'mutado' else '#15803d'
        ax2.set_title(f'Segmentacion tumoral  —  Slice {sl}',
                      fontsize=10, color='#334155', pad=10, fontweight='600')
        patch = mpatches.Patch(color=color_hex, label='Region tumoral')
        ax2.legend(handles=[patch], loc='lower right', fontsize=8,
                   facecolor='white', edgecolor='#e2e8f0', framealpha=0.95)
        ax2.axis('off')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="header-container">
    <p class="header-title">GBM IDH Classifier</p>
    <p class="header-subtitle">
        Clasificacion automatica del estado IDH en Glioblastoma Multiforme
        mediante segmentacion volumetrica y modelos de aprendizaje profundo
    </p>
    <div class="header-meta">
        <span class="badge">SegResNet BraTS23</span>
        <span class="badge">LR + Radiomics  |  AUC 0.914</span>
        <span class="badge">MedicalNet ResNet-18  |  AUC 0.901</span>
        <span class="badge">HD-BET Skull Stripping</span>
        <span class="badge">DICOM / NIfTI</span>
    </div>
</div>
""", unsafe_allow_html=True)

pipeline = cargar_pipeline()
tab1, tab2 = st.tabs(['  Cargar Imagenes  ', '  Resultados  '])

# ============================================================
# Pestana 1: Carga de imagenes
# ============================================================
with tab1:

    modo_entrada = st.radio(
        'Formato de entrada',
        ['NIfTI (.nii / .nii.gz)', 'DICOM (archivo ZIP)'],
        horizontal=True,
    )

    st.divider()

    # ----------------------------------------------------------
    # Modo NIfTI
    # ----------------------------------------------------------
    if modo_entrada == 'NIfTI (.nii / .nii.gz)':
        st.markdown('##### Modalidades MRI')
        st.caption('Sube los archivos en cualquier orden. El sistema detectara la modalidad automaticamente.')

        archivos_subidos = st.file_uploader(
            'Carga las 4 modalidades',
            type=['gz', 'nii'],
            accept_multiple_files=True,
            label_visibility='collapsed',
        )

        modalidades_detectadas = {}
        modalidades_sin_detectar = []

        if archivos_subidos:
            for archivo in archivos_subidos:
                mod = detectar_modalidad(archivo.name)
                if mod:
                    modalidades_detectadas[mod] = archivo
                else:
                    modalidades_sin_detectar.append(archivo.name)

            st.markdown('**Estado de deteccion:**')
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            cols_mod = {'t1': col_d1, 't1ce': col_d2, 't2': col_d3, 'flair': col_d4}
            labels   = {'t1': 'T1', 't1ce': 'T1ce', 't2': 'T2', 'flair': 'FLAIR'}

            for mod, col in cols_mod.items():
                with col:
                    if mod in modalidades_detectadas:
                        st.markdown(
                            f'<p class="upload-label">{labels[mod]}</p>'
                            f'<p class="mod-detected">Detectado: {modalidades_detectadas[mod].name}</p>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<p class="upload-label">{labels[mod]}</p>'
                            f'<p class="mod-warning">Pendiente</p>',
                            unsafe_allow_html=True,
                        )

            if modalidades_sin_detectar:
                st.warning(
                    f'No se pudo detectar la modalidad de: {", ".join(modalidades_sin_detectar)}. '
                    'Renombra el archivo incluyendo la modalidad (ej: paciente_T1ce.nii.gz).'
                )

            faltantes = [m for m in ['t1', 't1ce', 't2', 'flair']
                         if m not in modalidades_detectadas]
            if faltantes:
                st.info(f'Falta subir: {", ".join([labels[m] for m in faltantes])}')

        files_ok_nifti = len(modalidades_detectadas) == 4

    # ----------------------------------------------------------
    # Modo DICOM ZIP
    # ----------------------------------------------------------
    else:
        st.markdown('##### Archivos DICOM')
        st.caption(
            'Sube uno o varios archivos ZIP con imagenes DICOM. '
            'Nota: preferiblemente nombra cada archivo segun la secuencia '
            '(ej: t1.zip, t2.zip, t1ce.zip, flair.zip).'
        )

        zip_files = st.file_uploader(
            'Sube los ZIPs con los DICOMs',
            type=['zip'],
            accept_multiple_files=True,
            label_visibility='collapsed',
        )

        # Reanalisar si cambian los archivos subidos
        zip_names_actual = sorted([f.name for f in zip_files]) if zip_files else []
        zip_names_previo = st.session_state.get('zip_names_previo', [])

        if zip_files and zip_names_actual != zip_names_previo:
            st.session_state.pop('series_analizadas', None)
            st.session_state.pop('series_dicom', None)
            st.session_state.pop('asignaciones_dicom', None)
            st.session_state['zip_names_previo'] = zip_names_actual

        if zip_files and not st.session_state.get('series_analizadas'):
            with st.spinner(f'Analizando {len(zip_files)} archivo(s) DICOM...'):
                try:
                    work_dir_zip = WORK_DIR / 'dicom_analysis'
                    if work_dir_zip.exists():
                        shutil.rmtree(str(work_dir_zip))
                    work_dir_zip.mkdir(parents=True)

                    # Guardar todos los ZIPs
                    zip_paths = []
                    for i, zf in enumerate(zip_files):
                        zp = save_upload(zf, work_dir_zip, f'entrada_{i}.zip')
                        zip_paths.append(zp)

                    series = pipeline.analizar_zip_dicom(zip_paths, work_dir_zip)

                    st.session_state['series_dicom']      = series
                    st.session_state['zip_paths']         = zip_paths
                    st.session_state['series_analizadas'] = True
                    st.success(f'{len(series)} series encontradas en {len(zip_files)} archivo(s).')
                except Exception as ex:
                    st.error(f'Error analizando DICOM: {str(ex)}')

        if st.session_state.get('series_analizadas') and zip_files:
            series = st.session_state['series_dicom']

            st.markdown('**Series detectadas — Confirma o corrige la modalidad de cada serie:**')

            asignaciones = {}
            for s in series:
                col_info, col_select = st.columns([3, 1])
                with col_info:
                    desc = s['descripcion'] or 'Sin descripcion'
                    seq  = s['secuencia']   or 'Sin secuencia'
                    st.markdown(
                        f'<div class="serie-card">'
                        f'<strong>Serie {s["serie"]}</strong> — {s["n_archivos"]} archivos<br>'
                        f'<small>Descripcion: {desc} | Secuencia: {seq}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_select:
                    opciones    = ['T1', 'T1ce', 'T2', 'FLAIR', 'Ignorar']
                    sugerida    = s['modalidad_sugerida']
                    idx_defecto = {'t1': 0, 't1ce': 1, 't2': 2,
                                   'flair': 3, None: 4}.get(sugerida, 4)
                    seleccion = st.selectbox(
                        f'Modalidad serie {s["serie"]}',
                        opciones,
                        index=idx_defecto,
                        key=f'serie_{s["serie"]}',
                        label_visibility='collapsed',
                    )
                    if seleccion != 'Ignorar':
                        mod_key = {'t1': 't1', 't1ce': 't1ce',
                                   't2': 't2', 'flair': 'flair'}.get(
                            seleccion.lower(), None
                        )
                        if mod_key:
                            asignaciones[mod_key] = s

            st.session_state['asignaciones_dicom'] = asignaciones

            faltantes_dicom = [m for m in ['t1', 't1ce', 't2', 'flair']
                                if m not in asignaciones]
            if faltantes_dicom:
                labels_e = {'t1': 'T1', 't1ce': 'T1ce', 't2': 'T2', 'flair': 'FLAIR'}
                st.warning(
                    f'Faltan asignar: {", ".join([labels_e[m] for m in faltantes_dicom])}. '
                    'Revisa las series de arriba o sube ZIPs adicionales.'
                )
            else:
                st.success('Las 4 modalidades estan asignadas. Puedes ejecutar el analisis.')

        files_ok_dicom = (
            st.session_state.get('series_analizadas', False) and
            bool(zip_files) and
            len(st.session_state.get('asignaciones_dicom', {})) == 4
        )

    # ----------------------------------------------------------
    # Configuracion del analisis
    # ----------------------------------------------------------
    st.divider()
    st.markdown('##### Configuracion del analisis')

    modelo_sel = st.radio(
        'Modelo de clasificacion IDH',
        ['Machine Learning  —  LR + Radiomics',
         'Deep Learning  —  MedicalNet ResNet-18'],
        index=0,
    )

    st.divider()

    # Boton de analisis
    if modo_entrada == 'NIfTI (.nii / .nii.gz)':
        boton_ok = files_ok_nifti
    else:
        boton_ok = files_ok_dicom

    if not boton_ok:
        if modo_entrada == 'NIfTI (.nii / .nii.gz)':
            st.info('Carga las 4 modalidades NIfTI para continuar.')
        else:
            if not zip_files:
                st.info('Sube uno o varios archivos ZIP con los DICOMs para continuar.')

    if st.button('Ejecutar analisis', disabled=not boton_ok, use_container_width=True):

        progreso_container = st.container()
        with progreso_container:
            barra    = st.progress(0, text='Iniciando pipeline...')
            log_area = st.empty()

        def actualizar_progreso(mensaje: str):
            if 'Reorientando' in mensaje or 'Convirtiendo' in mensaje:
                pct = 10
            elif 'Detectando' in mensaje or 'Skull' in mensaje or 'stripped' in mensaje:
                pct = 30
            elif 'Segment' in mensaje:
                pct = 60
            else:
                pct = 85
            barra.progress(pct, text=mensaje)
            log_area.markdown(f'`{mensaje}`')

        try:
            work_dir = WORK_DIR / 'current'
            if work_dir.exists():
                shutil.rmtree(str(work_dir))
            work_dir.mkdir(parents=True)

            # Preparar paths NIfTI
            if modo_entrada == 'NIfTI (.nii / .nii.gz)':
                paths = {}
                for mod, archivo in modalidades_detectadas.items():
                    ext = '.nii.gz' if archivo.name.endswith('.gz') else '.nii'
                    paths[mod] = save_upload(archivo, work_dir, f'{mod}{ext}')
            else:
                # Convertir series DICOM confirmadas a NIfTI
                barra.progress(5, text='Convirtiendo DICOM a NIfTI...')
                asignaciones = st.session_state['asignaciones_dicom']
                nifti_dir    = work_dir / 'nifti_from_dicom'
                nifti_dir.mkdir(exist_ok=True)
                paths = {}
                for mod, serie_info in asignaciones.items():
                    out_path = str(nifti_dir / f'{mod}.nii.gz')
                    actualizar_progreso(f'Convirtiendo DICOM a NIfTI: {mod.upper()}...')
                    pipeline.convertir_serie_dicom(
                        serie_info['archivos'], out_path
                    )
                    paths[mod] = out_path

            modelo    = 'ml' if 'LR' in modelo_sel else 'dl'
            resultado = pipeline.ejecutar(
                paths=paths,
                work_dir=work_dir,
                modelo=modelo,
                callback=actualizar_progreso,
            )

            barra.progress(100, text='Analisis completado.')
            log_area.empty()

            mask_data = nib.load(
                str(work_dir / 'segmentation.nii.gz')
            ).get_fdata().astype('uint8')

            # Usar t1ce en el mismo espacio que la mascara para visualizacion
            # Prioridad: registered > reoriented > original
            t1ce_registered = work_dir / 'registered' / 't1ce_registered.nii.gz'
            t1ce_reoriented  = work_dir / 'reoriented' / 't1ce_LPS.nii.gz'
            if t1ce_registered.exists():
                t1ce_vis = str(t1ce_registered)
            elif t1ce_reoriented.exists():
                t1ce_vis = str(t1ce_reoriented)
            else:
                t1ce_vis = paths['t1ce']

            # Verificar que t1ce y mascara tienen el mismo shape
            import nibabel as nib_check
            t1ce_shape = nib_check.load(t1ce_vis).shape[:2]
            mask_shape = mask_data.shape[:2]
            if t1ce_shape != mask_shape:
                # Fallback: usar reorientada que es la referencia del segmentador
                if t1ce_reoriented.exists():
                    t1ce_vis = str(t1ce_reoriented)

            st.session_state['resultado']  = resultado
            st.session_state['mask_data']  = mask_data
            st.session_state['t1ce_path']  = t1ce_vis
            st.session_state['procesado']  = True

            # Limpiar estado DICOM para permitir nuevo analisis
            if modo_entrada == 'DICOM (archivo ZIP)':
                st.session_state.pop('series_analizadas', None)
                st.session_state.pop('series_dicom', None)
                st.session_state.pop('asignaciones_dicom', None)

            st.success('Analisis completado. Revisa la pestana Resultados.')

        except Exception as ex:
            barra.empty()
            log_area.empty()
            st.error(f'Error en el pipeline: {str(ex)}')

    st.markdown("""
    <div class="disclaimer">
        <strong>Aviso clinico:</strong> Esta herramienta ha sido desarrollada con fines
        exclusivamente investigativos en el marco de un proyecto de tesis de grado.
        Los resultados no constituyen un diagnostico medico y no deben ser utilizados
        como sustituto de la evaluacion clinica por un profesional de la salud calificado.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Pestana 2: Resultados
# ============================================================
with tab2:
    if not st.session_state.get('procesado', False):
        st.markdown("""
        <div class="empty-state">
            <p class="empty-state-title">Sin resultados disponibles</p>
            <p class="empty-state-text">
                Carga las imagenes en la pestana anterior y ejecuta el analisis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        resultado  = st.session_state['resultado']
        mask_data  = st.session_state['mask_data']
        t1ce_path  = st.session_state['t1ce_path']
        prediccion = resultado['prediccion']
        prob_pred  = (resultado['prob_mutado']
                      if prediccion == 'mutado'
                      else resultado['prob_wildtype'])
        css_class  = 'result-mutado'   if prediccion == 'mutado' else 'result-wildtype'
        ind_class  = ('result-indicator-mutado'
                      if prediccion == 'mutado'
                      else 'result-indicator-wildtype')
        label      = 'IDH Mutado' if prediccion == 'mutado' else 'IDH Wildtype'

        # Resultado principal
        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-indicator {ind_class}"></div>
            <p class="result-label">{label}</p>
            <p class="result-conf">
                Probabilidad: <strong>{prob_pred*100:.1f}%</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                P(mutado) {resultado['prob_mutado']*100:.1f}%
                &nbsp;&nbsp;
                P(wildtype) {resultado['prob_wildtype']*100:.1f}%
            </p>
            <p class="result-model">{resultado['modelo']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Indicador de confianza
        st.markdown(
            indicador_confianza(prob_pred, prediccion),
            unsafe_allow_html=True
        )

        # Metricas: probabilidades + metricas del modelo
        sens = resultado.get('sensibilidad', 0.0)
        spec = resultado.get('especificidad', 0.0)
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{resultado['prob_mutado']*100:.1f}%</div>
                <div class="metric-label">P(Mutado)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{resultado['prob_wildtype']*100:.1f}%</div>
                <div class="metric-label">P(Wildtype)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{resultado['auc_modelo']:.4f}</div>
                <div class="metric-label">AUC del modelo</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sens*100:.1f}%</div>
                <div class="metric-label">Sensibilidad</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{spec*100:.1f}%</div>
                <div class="metric-label">Especificidad</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Slicer interactivo
        st.markdown('##### Exploracion volumetrica del tumor')
        mostrar_slicer(t1ce_path, mask_data, resultado)

        st.divider()

        # Exportar reporte PDF
        st.markdown('##### Exportar reporte')
        if st.button('Generar reporte PDF', use_container_width=False):
            with st.spinner('Generando reporte...'):
                try:
                    pdf_bytes  = generar_reporte_pdf(resultado, t1ce_path, mask_data)
                    nombre_pdf = f'GBM_IDH_Reporte_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
                    st.download_button(
                        label='Descargar reporte PDF',
                        data=pdf_bytes,
                        file_name=nombre_pdf,
                        mime='application/pdf',
                        use_container_width=False,
                    )
                except Exception as ex:
                    st.error(f'Error generando PDF: {str(ex)}')

        st.markdown("""
        <div class="disclaimer">
            <strong>Aviso clinico:</strong> Esta herramienta ha sido desarrollada con fines
            exclusivamente investigativos. Los resultados no constituyen un diagnostico medico.
        </div>
        """, unsafe_allow_html=True)
