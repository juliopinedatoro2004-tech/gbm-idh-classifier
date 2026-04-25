"""
gbm_pipeline.py
Pipeline completo para clasificacion IDH en Glioblastoma Multiforme.

Pasos:
    1. Conversion DICOM -> NIfTI (si aplica) con SimpleITK
    2. Reorientacion a LPS estandar
    3. Deteccion automatica de craneo
    4. Skull stripping con HD-BET (si tiene craneo)
    5. Segmentacion tumoral con SegResNet BraTS23
    6. Clasificacion IDH: ML (LR + Radiomics) o DL (MedicalNet ResNet-18)

Uso desde la interfaz:
    from gbm_pipeline import GBMPipeline
    pipeline = GBMPipeline()
    resultado = pipeline.ejecutar(paths, modelo='ml')
"""

import json
import shutil
import zipfile
import subprocess
import warnings
import logging
import numpy as np
import nibabel as nib
import nibabel.orientations as nibo
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.ERROR)

# ============================================================
# Rutas adaptadas a la estructura de GBM_TESIS
# Los archivos .py estan en: GBM_TESIS/Interfaz/
# ============================================================
_BASE          = Path('/content/drive/MyDrive/GBM_TESIS')
_SS_DIR        = _BASE / 'SS_PIPELINE'
_BUNDLE_PATH   = _BASE / 'MODELO PREENTRENADO/models/brats_mri_segmentation'
_PARAMS_PATH   = _SS_DIR / 'radiomics/params/radiomics_params.yaml'
_ML_DIR        = _SS_DIR / 'inference_pipeline_v2'
_MN_DIR        = _BASE / 'medicalnet_weights'
_MN_WEIGHTS    = _MN_DIR / 'resnet18.pth'
_MN_CONFIG     = _SS_DIR / 'dl_clasificador_v2/medicalnet/medicalnet_config_v2.json'
_MN_MODEL_PATH = _SS_DIR / 'dl_clasificador_v2/medicalnet/models/medicalnet_best.pth'
_HDBET_DRIVE   = _BASE / 'hdbet_weights'
_HDBET_CACHE   = Path('/root/hd-bet_params')
_CROP_SIZE     = (64, 64, 64)

_MIN_VOXELES_TUMOR = 500
_MAX_VOXELES_TUMOR = 500_000

# Keywords para deteccion de modalidad por SequenceName/SeriesDescription
_KEYWORDS_SERIE = {
    't1ce':  ['tfl3d', 'mprage', 'spgr', 'vibe', 't1ce', 't1gd', 'postcontrast',
              'gd', 'gadolinium', 'ce', 't1+c', 't1c'],
    't1':    ['se2d', 't1_se', 'tse_t1', 'precontrast', 't1pre', 'se_t1'],
    't2':    ['tse2d', 't2', 'fse_t2', 't2_tse', 't2w'],
    'flair': ['spcir', 'flair', 'tirm', 'ir_fse', 'fl'],
}


# ============================================================
# Funciones de preprocesamiento
# ============================================================

def _normalizar_volumen(vol: np.ndarray) -> np.ndarray:
    nz = vol[vol > 0]
    if len(nz) == 0:
        return vol
    return (vol - nz.mean()) / (nz.std() + 1e-8)


def _crop_centrado_en_tumor(imagen: np.ndarray, mascara: np.ndarray,
                             crop_size: tuple = _CROP_SIZE) -> np.ndarray:
    coords = np.where(mascara > 0)
    if len(coords[0]) > 0:
        centro = [int(np.mean(c)) for c in coords]
    else:
        centro = [s // 2 for s in imagen.shape[:3]]
    slices = []
    for i, (c, s) in enumerate(zip(centro, imagen.shape[:3])):
        half  = crop_size[i] // 2
        start = max(0, c - half)
        end   = min(s, start + crop_size[i])
        start = max(0, end - crop_size[i])
        slices.append(slice(start, end))
    crop      = imagen[slices[0], slices[1], slices[2]]
    pad_width = [(0, max(0, crop_size[i] - crop.shape[i])) for i in range(3)]
    return np.pad(crop, pad_width, mode='constant', constant_values=0)


# ============================================================
# Clase principal
# ============================================================

class GBMPipeline:
    """Pipeline completo para clasificacion IDH en GBM."""

    def __init__(self, base_drive: Path = None, device: str = None):
        self.base       = Path(base_drive) if base_drive else _BASE
        self.ss_dir     = self.base / 'SS_PIPELINE'
        self.bundle     = self.base / 'MODELO PREENTRENADO/models/brats_mri_segmentation'
        self.params     = self.ss_dir / 'radiomics/params/radiomics_params.yaml'
        self.ml_dir     = self.ss_dir / 'inference_pipeline_v2'
        self.mn_dir     = self.base / 'medicalnet_weights'
        self.mn_weights = self.mn_dir / 'resnet18.pth'
        self.mn_config  = self.ss_dir / 'dl_clasificador_v2/medicalnet/medicalnet_config_v2.json'
        self.mn_model   = self.ss_dir / 'dl_clasificador_v2/medicalnet/models/medicalnet_best.pth'
        self.hdbet_src  = self.base / 'hdbet_weights'
        self.hdbet_dst  = Path('/root/hd-bet_params')

        self.device = torch.device(device if device else
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))

        self._seg_model  = None
        self._preprocess = None
        self._clf        = None
        self._ml_cfg     = None
        self._dl_model   = None
        self._dl_cfg     = None

    # ----------------------------------------------------------
    # DICOM: analisis de series
    # ----------------------------------------------------------

    def analizar_zip_dicom(self, zip_paths: list,
                            work_dir: Path) -> list:
        """
        Extrae uno o varios ZIPs con DICOMs, lee los headers de todos
        los archivos encontrados y retorna una lista de series para
        confirmacion por parte del usuario.

        Parametros
        ----------
        zip_paths : list
            Lista de rutas a archivos ZIP. Puede contener uno o varios.
        work_dir : Path
            Directorio de trabajo temporal.

        Retorna lista de dicts:
            [{'serie': '6', 'n_archivos': 192, 'descripcion': '...',
              'secuencia': '*tfl3d1_16ns', 'modalidad_sugerida': 't1ce',
              'archivos': [...]}, ...]
        """
        import pydicom
        from collections import defaultdict

        extract_dir = work_dir / 'dicom_raw'
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Descomprimir todos los ZIPs en subcarpetas separadas
        for i, zip_path in enumerate(zip_paths):
            dest = extract_dir / f'zip_{i}'
            dest.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(str(dest))

        # Recolectar todos los archivos DICOM de todas las carpetas
        archivos = list(extract_dir.rglob('*.dcm')) + list(extract_dir.rglob('*.DCM'))
        if not archivos:
            for f in extract_dir.rglob('*'):
                if f.is_file() and f.suffix == '':
                    try:
                        with open(f, 'rb') as fh:
                            fh.seek(128)
                            if fh.read(4) == b'DICM':
                                archivos.append(f)
                    except:
                        pass

        if not archivos:
            raise ValueError(
                'No se encontraron archivos DICOM en los ZIPs subidos. '
                'Verifica que los archivos sean DICOM (.dcm).'
            )

        series = defaultdict(lambda: {
            'n_archivos': 0, 'descripcion': '',
            'secuencia': '', 'protocolo': '',
            'modalidad_sugerida': None, 'archivos': [],
            'carpeta': '',
        })

        for f in archivos:
            try:
                ds          = pydicom.dcmread(str(f), stop_before_pixels=True)
                serie_num   = str(getattr(ds, 'SeriesNumber', 'unknown'))
                descripcion = str(getattr(ds, 'SeriesDescription', ''))
                secuencia   = str(getattr(ds, 'SequenceName', ''))
                protocolo   = str(getattr(ds, 'ProtocolName', ''))
                series[serie_num]['n_archivos'] += 1
                series[serie_num]['descripcion'] = descripcion
                series[serie_num]['secuencia']   = secuencia
                series[serie_num]['protocolo']   = protocolo
                series[serie_num]['archivos'].append(str(f))
                series[serie_num]['carpeta']     = str(f.parent)
            except:
                continue

        # Inferir modalidad sugerida por keywords
        for serie_num, info in series.items():
            texto = (info['descripcion'] + ' ' +
                     info['secuencia'] + ' ' +
                     info['protocolo']).lower()
            # t1ce primero para no confundir con t1
            for mod in ['t1ce', 'flair', 't2', 't1']:
                for kw in _KEYWORDS_SERIE[mod]:
                    if kw in texto:
                        info['modalidad_sugerida'] = mod
                        break
                if info['modalidad_sugerida']:
                    break

        resultado = []
        for num in sorted(series.keys(),
                          key=lambda x: int(x) if x.isdigit() else 999):
            info = series[num]
            resultado.append({
                'serie':              num,
                'n_archivos':         info['n_archivos'],
                'descripcion':        info['descripcion'],
                'secuencia':          info['secuencia'],
                'protocolo':          info['protocolo'],
                'modalidad_sugerida': info['modalidad_sugerida'],
                'archivos':           info['archivos'],
                'carpeta':            info['carpeta'],
            })

        return resultado

    def convertir_serie_dicom(self, archivos: list,
                               out_path: str) -> str:
        """
        Convierte una lista de archivos DICOM de una serie a NIfTI
        usando SimpleITK. Maneja orientaciones oblicuas correctamente.
        """
        import SimpleITK as sitk

        carpeta = str(Path(archivos[0]).parent)
        reader  = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(carpeta)

        if series_ids:
            files_sorted = reader.GetGDCMSeriesFileNames(carpeta, series_ids[0])
        else:
            files_sorted = sorted(archivos)

        reader.SetFileNames(files_sorted)
        imagen = reader.Execute()
        sitk.WriteImage(imagen, out_path)
        return out_path

    # ----------------------------------------------------------
    # Paso 0b: Reorientacion a LPS
    # ----------------------------------------------------------

    def reorientar(self, img_path: str, out_path: str,
                   orientacion: tuple = ('L', 'P', 'S')) -> str:
        img         = nib.load(img_path)
        orig_ornt   = nibo.io_orientation(img.affine)
        target_ornt = nibo.axcodes2ornt(orientacion)
        transform   = nibo.ornt_transform(orig_ornt, target_ornt)
        img_r       = img.as_reoriented(transform)
        nib.save(img_r, out_path)
        return out_path

    # ----------------------------------------------------------
    # Paso 0c: Registro rigido al espacio de T1ce
    # ----------------------------------------------------------

    def registrar_al_t1ce(self, paths: dict, work_dir: Path,
                           callback=None) -> dict:
        """
        Registra T1, T2 y FLAIR al espacio geometrico de T1ce usando
        registro rigido con SimpleITK. Necesario cuando las modalidades
        tienen diferente resolucion, campo de vision o numero de slices.
        """
        import SimpleITK as sitk

        reg_dir = work_dir / 'registered'
        reg_dir.mkdir(exist_ok=True)

        fixed     = sitk.ReadImage(paths['t1ce'], sitk.sitkFloat32)
        paths_reg = {'t1ce': paths['t1ce']}

        for mod in ['t1', 't2', 'flair']:
            if callback:
                callback(f'Registrando {mod.upper()} al espacio de T1ce...')

            moving   = sitk.ReadImage(paths[mod], sitk.sitkFloat32)
            out_path = str(reg_dir / f'{mod}_registered.nii.gz')

            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )

            reg = sitk.ImageRegistrationMethod()
            reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            reg.SetMetricSamplingStrategy(reg.RANDOM)
            reg.SetMetricSamplingPercentage(0.10)
            reg.SetInterpolator(sitk.sitkLinear)
            reg.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=100,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
            )
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            reg.SetInitialTransform(initial_transform, inPlace=False)

            final_transform = reg.Execute(fixed, moving)

            moving_resampled = sitk.Resample(
                moving, fixed, final_transform,
                sitk.sitkLinear, 0.0, moving.GetPixelID(),
            )
            sitk.WriteImage(moving_resampled, out_path)
            paths_reg[mod] = out_path

        return paths_reg

    def necesita_registro(self, paths: dict) -> bool:
        """
        Determina si las modalidades necesitan registro comparando
        sus dimensiones fisicas en mm. Si difieren mas de 10mm en
        cualquier dimension, el registro es necesario.
        """
        import SimpleITK as sitk
        try:
            ref    = sitk.ReadImage(paths['t1ce'])
            ref_sz = [ref.GetSize()[i] * ref.GetSpacing()[i] for i in range(3)]
            for mod in ['t1', 't2', 'flair']:
                img    = sitk.ReadImage(paths[mod])
                img_sz = [img.GetSize()[i] * img.GetSpacing()[i] for i in range(3)]
                if any(abs(ref_sz[i] - img_sz[i]) > 10 for i in range(3)):
                    return True
            return False
        except:
            return False

    # ----------------------------------------------------------
    # Paso 1: Deteccion automatica de craneo
    # ----------------------------------------------------------

    def detectar_craneo(self, path_t1ce: str) -> bool:
        data  = nib.load(path_t1ce).get_fdata()
        ratio = (data > 0).sum() / data.size
        return bool(ratio > 0.25)

    # ----------------------------------------------------------
    # Paso 2: Skull stripping
    # ----------------------------------------------------------

    def skull_strip(self, paths: dict, work_dir: Path,
                    callback=None) -> dict:
        if self.hdbet_src.exists() and not self.hdbet_dst.exists():
            shutil.copytree(str(self.hdbet_src), str(self.hdbet_dst),
                            dirs_exist_ok=True)

        strip_dir = work_dir / 'stripped'
        strip_dir.mkdir(exist_ok=True)
        paths_out = {}

        for mod, nombre in [('t1','T1'), ('t1ce','T1GD'),
                             ('t2','T2'), ('flair','FLAIR')]:
            if callback:
                callback(f'Skull stripping: {nombre}')
            out = strip_dir / f'{nombre}_stripped.nii.gz'
            cmd = [
                'hd-bet', '-i', paths[mod], '-o', str(out),
                '-device', 'cuda' if torch.cuda.is_available() else 'cpu',
                '--disable_tta',
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f'HD-BET fallo en {mod}: {result.stderr.decode()[-200:]}'
                )
            paths_out[mod] = str(out)

        return paths_out

    # ----------------------------------------------------------
    # Paso 3: Segmentacion tumoral
    # ----------------------------------------------------------

    def _cargar_segmentador(self):
        if self._seg_model is not None:
            return
        from monai.bundle import load
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd,
            Orientationd, Spacingd, NormalizeIntensityd,
        )
        self._seg_model = load(
            name=str(self.bundle), source='disk', net_override={}
        )
        self._seg_model.to(self.device)
        self._seg_model.eval()
        self._preprocess = Compose([
            LoadImaged(keys=['t1', 't1ce', 't2', 'flair']),
            EnsureChannelFirstd(keys=['t1', 't1ce', 't2', 'flair']),
            Orientationd(keys=['t1', 't1ce', 't2', 'flair'], axcodes='RAS'),
            Spacingd(keys=['t1', 't1ce', 't2', 'flair'],
                     pixdim=(1., 1., 1.), mode='bilinear'),
            NormalizeIntensityd(keys=['t1', 't1ce', 't2', 'flair'],
                                nonzero=True, channel_wise=True),
        ])

    def segmentar(self, paths: dict, work_dir: Path,
                  callback=None) -> np.ndarray:
        from monai.inferers import sliding_window_inference
        if callback:
            callback('Segmentando tumor...')
        self._cargar_segmentador()

        data  = self._preprocess(paths)
        image = torch.stack([
            data['t1'][0], data['t1ce'][0],
            data['t2'][0], data['flair'][0],
        ], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=image, roi_size=(128, 128, 128),
                sw_batch_size=2, predictor=self._seg_model,
                overlap=0.5, mode='gaussian',
            )

        wt_binary = (torch.sigmoid(logits)[0, 1].cpu().numpy() > 0.5).astype(np.uint8)
        ref_img   = nib.load(paths['t1ce'])
        orig_ornt = nibo.io_orientation(ref_img.affine)
        ras_ornt  = nibo.io_orientation(nib.as_closest_canonical(ref_img).affine)
        transform = nibo.ornt_transform(ras_ornt, orig_ornt)
        pred_nii  = nib.Nifti1Image(wt_binary, nib.as_closest_canonical(ref_img).affine)
        pred_reor = pred_nii.as_reoriented(transform)
        nib.save(pred_reor, str(work_dir / 'segmentation.nii.gz'))

        mask_data = pred_reor.get_fdata().astype(np.uint8)
        n_vox     = int(mask_data.sum())

        if n_vox < _MIN_VOXELES_TUMOR:
            raise ValueError(
                f'Segmentacion insuficiente: {n_vox} voxeles detectados. '
                f'Minimo esperado: {_MIN_VOXELES_TUMOR}. '
                'Verifica que las imagenes correspondan a un caso de GBM.'
            )
        if n_vox > _MAX_VOXELES_TUMOR:
            raise ValueError(
                f'Segmentacion excesiva: {n_vox} voxeles detectados. '
                f'Maximo esperado: {_MAX_VOXELES_TUMOR}. '
                'Posible error de preprocesamiento o skull stripping incompleto.'
            )

        return mask_data

    # ----------------------------------------------------------
    # Paso 4a: Clasificacion ML
    # ----------------------------------------------------------

    def _cargar_ml(self):
        if self._clf is not None:
            return
        self._clf = joblib.load(str(self.ml_dir / 'best_model.joblib'))
        with open(self.ml_dir / 'pipeline_config.json') as f:
            self._ml_cfg = json.load(f)

    def clasificar_ml(self, t1ce_path: str, mask_data: np.ndarray,
                      work_dir: Path, callback=None) -> dict:
        from radiomics import featureextractor
        if callback:
            callback('Extrayendo features radiomicas...')
        self._cargar_ml()

        img        = nib.as_closest_canonical(nib.load(t1ce_path))
        img_data   = img.get_fdata().astype(np.float32)
        brain_mask = img_data > 0
        zooms      = img.header.get_zooms()[:3]
        mask_pp    = mask_data.copy()

        if not all(abs(z - 1.0) < 0.01 for z in zooms):
            factors    = tuple(float(z) for z in zooms)
            img_data   = zoom(img_data, factors, order=1)
            mask_pp    = zoom(mask_pp,  factors, order=0)
            brain_mask = img_data > 0

        if brain_mask.sum() > 0:
            p_low    = np.percentile(img_data[brain_mask], 0.5)
            p_high   = np.percentile(img_data[brain_mask], 99.5)
            img_data = np.clip(img_data, p_low, p_high)
            mean_v   = img_data[brain_mask].mean()
            std_v    = img_data[brain_mask].std()
            img_norm = np.zeros_like(img_data)
            img_norm[brain_mask] = (img_data[brain_mask] - mean_v) / (std_v + 1e-8)
        else:
            img_norm = img_data

        n_vox = int(mask_pp.sum())
        if n_vox < 50:
            raise ValueError(f'Mascara tumoral muy pequena: {n_vox} voxeles')

        affine_1mm = np.diag([1., 1., 1., 1.])
        img_out  = work_dir / 'img_pp.nii.gz'
        mask_out = work_dir / 'mask_pp.nii.gz'
        nib.save(nib.Nifti1Image(img_norm, affine_1mm), str(img_out))
        nib.save(nib.Nifti1Image(mask_pp,  affine_1mm), str(mask_out))

        if callback:
            callback('Clasificando con LR + Radiomics...')
        ext      = featureextractor.RadiomicsFeatureExtractor(str(self.params))
        result   = ext.execute(str(img_out), str(mask_out))
        features = {}
        for k, v in result.items():
            if k.startswith('diagnostics_'):
                continue
            try:
                val = float(np.squeeze(v))
                if np.isfinite(val):
                    features[k] = val
            except:
                continue

        feat_names = self._ml_cfg['feature_names']
        for fn in feat_names:
            if fn not in features:
                features[fn] = 0.0

        X     = np.array([features[fn] for fn in feat_names]).reshape(1, -1)
        proba = self._clf.predict_proba(X)[0, 1]

        metricas_ml = self._ml_cfg.get('metrics', {})
        return {
            'prediccion':    'mutado' if proba >= 0.5 else 'wildtype',
            'prob_mutado':   float(proba),
            'prob_wildtype': float(1 - proba),
            'auc_modelo':    self._ml_cfg.get('auc_holdout',
                             metricas_ml.get('auc', 0.9138)),
            'sensibilidad':  metricas_ml.get('recall',
                             self._ml_cfg.get('sensitivity', 0.7857)),
            'especificidad': metricas_ml.get('specificity', 0.9022),
            'modelo':        'ML LR + Radiomics',
        }

    # ----------------------------------------------------------
    # Paso 4b: Clasificacion DL
    # ----------------------------------------------------------

    def _cargar_dl(self):
        if self._dl_model is not None:
            return
        from monai.networks.nets import resnet18
        backbone = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
        state = torch.load(str(self.mn_weights), map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.replace('module.', ''): v for k, v in state.items()}
        backbone.load_state_dict(state, strict=False)
        pesos_orig = backbone.conv1.weight.data
        pesos_new  = pesos_orig.repeat(1, 4, 1, 1, 1) / 4.0
        backbone.conv1 = nn.Conv3d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = pesos_new
        for param in backbone.bn1.parameters():    param.requires_grad = False
        for param in backbone.layer1.parameters(): param.requires_grad = False
        for param in backbone.layer2.parameters(): param.requires_grad = False
        encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.act, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self._dl_model = nn.Sequential(
            encoder,
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
        self._dl_model.load_state_dict(
            torch.load(str(self.mn_model), map_location=self.device)
        )
        self._dl_model.to(self.device)
        self._dl_model.eval()
        with open(self.mn_config) as f:
            self._dl_cfg = json.load(f)

    def clasificar_dl(self, paths: dict, mask_data: np.ndarray,
                      callback=None) -> dict:
        if callback:
            callback('Clasificando con MedicalNet...')
        self._cargar_dl()

        vols = []
        for mod in ['t1', 't1ce', 't2', 'flair']:
            vol = nib.load(paths[mod]).get_fdata().astype(np.float32)
            vol = _normalizar_volumen(vol)
            vol = _crop_centrado_en_tumor(vol, mask_data)
            vols.append(vol)
        volumen = np.stack(vols, axis=-1)
        imagen  = torch.tensor(
            volumen, dtype=torch.float32
        ).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

        self._dl_model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(self._dl_model(imagen)).item()

        umbral = self._dl_cfg.get('threshold', 0.5)

        return {
            'prediccion':    'mutado' if prob >= umbral else 'wildtype',
            'prob_mutado':   float(prob),
            'prob_wildtype': float(1 - prob),
            'auc_modelo':    self._dl_cfg.get('auc_holdout', 0.9014),
            'sensibilidad':  self._dl_cfg.get('sensitivity', 0.8571),
            'especificidad': self._dl_cfg.get('specificity', 0.6848),
            'modelo':        'DL MedicalNet ResNet-18',
        }

    # ----------------------------------------------------------
    # Pipeline orquestador
    # ----------------------------------------------------------

    def ejecutar(self, paths: dict, work_dir: Path,
                 modelo: str = 'ml',
                 callback=None) -> dict:
        """
        Ejecuta el pipeline completo de forma automatica.

        Parametros
        ----------
        paths : dict
            Rutas NIfTI confirmadas: t1, t1ce, t2, flair.
        work_dir : Path
            Directorio de trabajo temporal.
        modelo : str
            'ml' para LR + Radiomics, 'dl' para MedicalNet.
        callback : callable, opcional
            Funcion que recibe un string con el paso actual.
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Paso 0b: Reorientacion a LPS
        if callback:
            callback('Reorientando imagenes a LPS...')
        reor_dir = work_dir / 'reoriented'
        reor_dir.mkdir(exist_ok=True)
        paths_reor = {}
        for mod, path in paths.items():
            out = str(reor_dir / f'{mod}_LPS.nii.gz')
            paths_reor[mod] = self.reorientar(path, out)

        # Paso 0c: Registro rigido si las modalidades tienen espacios diferentes
        if self.necesita_registro(paths_reor):
            if callback:
                callback('Registrando modalidades al espacio de T1ce...')
            paths_reor = self.registrar_al_t1ce(paths_reor, work_dir, callback=callback)

        # Paso 1-2: Deteccion y skull stripping automatico
        if callback:
            callback('Detectando craneo...')
        tiene_craneo = self.detectar_craneo(paths_reor['t1ce'])

        if tiene_craneo:
            paths_proc = self.skull_strip(paths_reor, work_dir, callback=callback)
        else:
            if callback:
                callback('Imagenes ya stripped. Omitiendo HD-BET...')
            paths_proc = paths_reor

        # Paso 3: Segmentacion
        mask_data = self.segmentar(paths_proc, work_dir, callback=callback)

        # Paso 4: Clasificacion
        if modelo == 'ml':
            return self.clasificar_ml(
                paths_proc['t1ce'], mask_data, work_dir, callback=callback
            )
        elif modelo == 'dl':
            return self.clasificar_dl(paths_proc, mask_data, callback=callback)
        else:
            raise ValueError(f'Modelo desconocido: {modelo}. Usar "ml" o "dl".')
