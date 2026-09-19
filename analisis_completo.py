# =============================================================================
# PROYECTO: ANALISIS DE DATOS DEL IDEAM
# Estructura en 3 Modulos:
# 1. Fundamentos y Preparacion de Datos
# 2. Analisis Estadistico
# 3. Visualizacion, Modelo Predictivo y Storytelling
# =============================================================================

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurar estilo visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*70)
print("INICIANDO PROYECTO IDEAM")
print("="*70)

# =============================================================================
# MODULO 1: FUNDAMENTOS Y PREPARACION DE DATOS
# =============================================================================
print("\n--- MODULO 1: PREPARACION DE DATOS ---")

try:
    df = pd.read_csv('data/proyecto.csv', encoding='latin1')
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print("Error: No se encontro 'data/proyecto.csv'. Asegurate de la ruta.")
    exit()

# Limpieza de coordenadas
df['LONGITUD'] = pd.to_numeric(df['LONGITUD'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
df['LATITUD'] = pd.to_numeric(df['LATITUD'].astype(str).str.replace(',', '.', regex=False), errors='coerce')

# Limpieza de Altitud (quitar puntos de miles)
df['Altitud'] = pd.to_numeric(df['Altitud'].astype(str).str.replace('.', '', regex=False), errors='coerce')

# Fechas
df['Fecha_instalacion_dt'] = pd.to_datetime(df['Fecha_instalacion'], format='%d/%m/%Y', errors='coerce')
df['Anio_instalacion'] = df['Fecha_instalacion_dt'].dt.year

# Eliminar duplicados y filas con nulos criticos
df = df.drop_duplicates()
df = df.dropna(subset=['Altitud', 'LATITUD', 'LONGITUD'])

print(f"Limpieza completada. Filas finales: {len(df)}")
df.to_csv('data/ideam_limpio.csv', index=False)
print("Dataset limpio guardado como 'data/ideam_limpio.csv'")

# =============================================================================
# MODULO 2: ANALISIS ESTADISTICO
# =============================================================================
print("\n" + "="*70)
print("MODULO 2: ANALISIS ESTADISTICO")
print("="*70)

media_alt = df['Altitud'].mean()
mediana_alt = df['Altitud'].median()
moda_alt = df['Altitud'].mode()[0]

print(f"\n[1] Tendencia Central de la Altitud:")
print(f"    - Media: {media_alt:.2f} msnm")
print(f"    - Mediana: {mediana_alt:.2f} msnm")
print(f"    - Moda: {moda_alt} msnm")

rango = df['Altitud'].max() - df['Altitud'].min()
varianza = df['Altitud'].var()
std = df['Altitud'].std()

print(f"\n[2] Medidas de Dispersion de la Altitud:")
print(f"    - Rango: {rango} m")
print(f"    - Varianza: {varianza:.2f}")
print(f"    - Desviacion Estandar: {std:.2f} m")
print(f"    - Interpretacion: La desviacion estandar de {std:.0f} m indica que las estaciones")
print(f"      estan muy dispersas a lo largo de la geografia colombiana.")

print(f"\n[3] Top 5 Departamentos con mas estaciones:")
print(df['Departamento'].value_counts().head(5))

Q1 = df['Altitud'].quantile(0.25)
Q3 = df['Altitud'].quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR
outliers = df[(df['Altitud'] < limite_inf) | (df['Altitud'] > limite_sup)]
print(f"\n[4] Deteccion de Outliers en Altitud:")
print(f"    - Limites: {limite_inf:.2f} m a {limite_sup:.2f} m")
print(f"    - Se detectaron {len(outliers)} valores atipicos.")

# =============================================================================
# MODULO 3: VISUALIZACION, MODELO PREDICTIVO Y STORYTELLING
# =============================================================================
print("\n" + "="*70)
print("MODULO 3: VISUALIZACION, MODELO PREDICTIVO Y STORYTELLING")
print("="*70)

# --- 3.1 VISUALIZACIONES BASICAS ---
print("\n[3.1] Generando visualizaciones exploratorias...")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
top_deptos = df['Departamento'].value_counts().head(10)
sns.barplot(x=top_deptos.values, y=top_deptos.index, palette="viridis", hue=top_deptos.index, legend=False)
plt.title('Top 10 Departamentos con mas Estaciones IDEAM')
plt.xlabel('Cantidad')

plt.subplot(2, 2, 2)
sns.histplot(df['Altitud'], bins=30, kde=True, color='blue')
plt.title('Distribucion de la Altitud')
plt.xlabel('Altitud (msnm)')

plt.subplot(2, 2, 3)
sns.boxplot(x='Categoria', y='Altitud', data=df, palette="Set2", hue='Categoria', legend=False)
plt.xticks(rotation=45, ha='right')
plt.title('Altitud segun Categoria de Estacion')

plt.subplot(2, 2, 4)
sns.scatterplot(x='LONGITUD', y='LATITUD', hue='Estado', data=df, alpha=0.6, palette="deep")
plt.title('Distribucion Geografica (Lat vs Lon)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()
plt.show()

# --- 3.2 MODELOS PREDICTIVOS (REGRESIONES) ---
print("\n[3.2] Entrenando Modelos Predictivos...")

df_model = df.dropna(subset=['Altitud', 'LATITUD', 'LONGITUD', 'Anio_instalacion']).copy()
df_model = df_model[df_model['Altitud'] > 0]  # Para logaritmica

def evaluar_modelo(nombre, y_real, y_pred):
    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    print(f"\n    --- Metricas para {nombre} ---")
    print(f"    R2   : {r2:.4f}  (0=malo, 1=perfecto)")
    print(f"    RMSE : {rmse:.2f} m")
    print(f"    MAE  : {mae:.2f} m")
    return {"r2": r2, "rmse": rmse, "mae": mae}

# 3.2.1 Regresion Lineal Simple (Latitud)
X_s = df_model[['LATITUD']]
y_s = df_model['Altitud']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
mod_lineal = LinearRegression().fit(X_train_s, y_train_s)
y_pred_lineal = mod_lineal.predict(X_test_s)
met_lineal = evaluar_modelo("Regresion Lineal Simple", y_test_s, y_pred_lineal)

# 3.2.2 Regresion Lineal Multiple (Latitud, Longitud, Anio)
X_m = df_model[['LATITUD', 'LONGITUD', 'Anio_instalacion']]
y_m = df_model['Altitud']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)
mod_multiple = LinearRegression().fit(X_train_m, y_train_m)
y_pred_multiple = mod_multiple.predict(X_test_m)
met_multiple = evaluar_modelo("Regresion Lineal Multiple", y_test_m, y_pred_multiple)

# 3.2.3 Regresion Logaritmica
y_log = np.log(df_model['Altitud'])
X_log = df_model[['LATITUD', 'LONGITUD']]
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
mod_log = LinearRegression().fit(X_train_l, y_train_l)
y_pred_log = np.exp(mod_log.predict(X_test_l))
met_log = evaluar_modelo("Regresion Logaritmica", np.exp(y_test_l), y_pred_log)

# --- 3.3 COMPARACION Y VALIDACION CRUZADA ---
print("\n[3.3] Comparacion y Validacion Cruzada...")

comparacion = pd.DataFrame({
    'Modelo': ['Lineal Simple', 'Multiple', 'Logaritmica'],
    'R2': [met_lineal['r2'], met_multiple['r2'], met_log['r2']],
    'RMSE (m)': [met_lineal['rmse'], met_multiple['rmse'], met_log['rmse']],
    'MAE (m)': [met_lineal['mae'], met_multiple['mae'], met_log['mae']],
})
print("\n")
print(comparacion.to_string(index=False))

mejor_modelo = comparacion.loc[comparacion['R2'].idxmax(), 'Modelo']
print(f"\n    El modelo con mejor R2 es: {mejor_modelo}")

# Validacion cruzada K-Fold para el modelo multiple
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(mod_multiple, X_m, y_m, cv=kf, scoring='r2')
print(f"\n    Validacion Cruzada (k=5) para Regresion Multiple:")
print(f"    R2 promedio: {cv_scores.mean():.4f}")
print(f"    Desviacion estandar del R2: {cv_scores.std():.4f}")

# --- 3.4 GRAFICAS DE REGRESIONES ---
print("\n[3.4] Generando graficas de regresiones...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_test_s, y_test_s, alpha=0.4, color='blue', label='Datos reales')
axes[0].scatter(X_test_s, y_pred_lineal, alpha=0.6, color='red', label='Prediccion')
axes[0].set_title(f'Regresion Lineal Simple\nR2 = {met_lineal["r2"]:.3f}')
axes[0].set_xlabel('Latitud')
axes[0].set_ylabel('Altitud (msnm)')
axes[0].legend()

axes[1].scatter(y_test_m, y_pred_multiple, alpha=0.5, color='green')
axes[1].plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], 'r--', lw=2)
axes[1].set_title(f'Regresion Multiple (Real vs Predicho)\nR2 = {met_multiple["r2"]:.3f}')
axes[1].set_xlabel('Altitud Real (msnm)')
axes[1].set_ylabel('Altitud Predicha (msnm)')

axes[2].scatter(np.exp(y_test_l), y_pred_log, alpha=0.5, color='purple')
axes[2].plot([np.exp(y_test_l).min(), np.exp(y_test_l).max()],
             [np.exp(y_test_l).min(), np.exp(y_test_l).max()], 'r--', lw=2)
axes[2].set_title(f'Regresion Logaritmica\nR2 = {met_log["r2"]:.3f}')
axes[2].set_xlabel('Altitud Real (msnm)')
axes[2].set_ylabel('Altitud Predicha (msnm)')

plt.tight_layout()
plt.show()

# --- 3.5 STORYTELLING Y APLICACION PROFESIONAL ---
print("\n[3.5] Storytelling y Aplicacion Profesional:")
print(f"""
    HISTORIA DEL MODELO:
    1. QUE HACE: Predecimos la Altitud de una estacion del IDEAM usando su
       ubicacion geografica (Latitud, Longitud) y el anio de instalacion.
    2. COMO FUNCIONA:
       - Lineal Simple: Altitud = b0 + b1*Latitud
       - Multiple: Altitud = b0 + b1*Lat + b2*Lon + b3*Anio
       - Logaritmica: log(Altitud) = b0 + b1*Lat + b2*Lon
    3. RESULTADOS:
       - Lineal Simple : R2 = {met_lineal['r2']:.4f}, RMSE = {met_lineal['rmse']:.2f} m
       - Multiple      : R2 = {met_multiple['r2']:.4f}, RMSE = {met_multiple['rmse']:.2f} m
       - Logaritmica   : R2 = {met_log['r2']:.4f}, RMSE = {met_log['rmse']:.2f} m
    4. VALIDACION: R2 promedio en K-Fold = {cv_scores.mean():.4f}
    5. INTERPRETACION: Un R2 bajo indica que la geografia colombiana es muy
       compleja (cordilleras, valles, costas) y las coordenadas no bastan
       para predecir la altitud.
    6. APLICACION PROFESIONAL: Permite al IDEAM priorizar mantenimiento
       en zonas de dificil acceso y planificar nuevas estaciones.
""")

# =============================================================================
# EXTRA: GENERACION DE DATOS PARA DASHBOARD WEB (TU CODIGO ORIGINAL)
# =============================================================================
print("="*70)
print("EXTRA: GENERANDO data.js PARA DASHBOARD WEB")
print("="*70)

out = {}
out['total'] = int(len(df))
out['departamentos_unicos'] = int(df['Departamento'].nunique())
out['entidades_unicas'] = int(df['Entidad'].nunique())
out['municipios_unicos'] = int(df['Municipio'].nunique())

out['estado'] = [{"label": k, "value": int(v)} for k, v in df['Estado'].value_counts().items()]
out['categoria'] = [{"label": k, "value": int(v)} for k, v in df['Categoria'].value_counts().items()]
out['tecnologia'] = [{"label": k, "value": int(v)} for k, v in df['Tecnologia'].value_counts().items()]
out['departamento_top12'] = [{"label": k, "value": int(v)} for k, v in df['Departamento'].value_counts().head(12).items()]
out['entidad_top8'] = [{"label": k, "value": int(v)} for k, v in df['Entidad'].value_counts().head(8).items()]

valid_dates = df.dropna(subset=['Fecha_instalacion_dt'])
decades = (valid_dates['Fecha_instalacion_dt'].dt.year // 10 * 10)
out['decada'] = [{"label": f"{int(d)}", "value": int(v)} for d, v in decades.value_counts().sort_index().items()]

estado_code = {'Activa': 'A', 'Suspendida': 'S', 'En Mantenimiento': 'M'}
pts = df[['LONGITUD', 'LATITUD', 'Estado']].dropna()
by_estado = {'A': [], 'S': [], 'M': []}
for r in pts.itertuples():
    by_estado[estado_code[r.Estado]].append([round(r.LONGITUD, 3), round(r.LATITUD, 3)])
out['points_by_estado'] = by_estado
out['metricas_modelos'] = comparacion.to_dict(orient='records')

with open('data.js', 'w', encoding='utf-8') as f:
    f.write('window.STATION_DATA = ')
    json.dump(out, f, ensure_ascii=False)
    f.write(';')

print("'data.js' generado exitosamente para tu dashboard web.")
print("="*70)
print("FIN DEL PROYECTO")
print("="*70)