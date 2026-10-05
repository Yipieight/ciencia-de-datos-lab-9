"""Construye y ejecuta el notebook. Ejecutar con: uv run scripts/preparar_entrega.py."""
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Laboratorio_9_Particion_Datos.ipynb"

# Transcripción del bloque del enunciado: no se cambian sus parámetros.
MODELO = '''
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, recall_score

NUM = ['distancia_km', 'total_pedido_gtq']
CAT = ['zona', 'categoria_restaurante', 'metodo_pago']
FEATURES = NUM + CAT

def nuevo_modelo():
    """M1. Siempre se crea nuevo; nunca se reutiliza uno ya entrenado."""
    prep = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('esc', StandardScaler())]), NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT),
    ])
    return Pipeline([
        ('prep', prep),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])

def entrenar_y_medir(train, test, target='baja'):
    """Entrena M1 con 'train', mide en 'test' y devuelve F1 y recall."""
    m = nuevo_modelo()
    m.fit(train[FEATURES], train[target])
    pred = m.predict(test[FEATURES])
    return {'f1': f1_score(test[target], pred),
            'recall': recall_score(test[target], pred),
            'n_test': len(test),
            'positivos_test': int(test[target].sum())}
'''
BASELINE = '''
def medir_baseline(train, test, target='baja'):
    """M0: modelo "tonto". Predice siempre la clase más común."""
    d = DummyClassifier(strategy='most_frequent')
    d.fit(train[FEATURES], train[target])
    return f1_score(test[target], d.predict(test[FEATURES]))
'''

cells = []


def md(source: str, tag: str | None = None):
    cell = nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())
    if tag:
        cell.metadata["respuesta"] = tag
    cells.append(cell)


def code(source: str):
    cells.append(nbformat.v4.new_code_cell(textwrap.dedent(source).strip()))


def build():
    md("""
    # Laboratorio 9 - ¿Importa cómo partimos los datos?
    **Universidad Rafael Landívar · Facultad de Ingeniería**  
    **Ciencia de Datos · Sección 2 · Segundo semestre 2026**  
    Docente: Ing. Max Cerna

    | Integrante | Carné | Rol propuesto | Responsabilidad |
    |---|---|---|---|
    | GARCÍA ELÍAS JOSÉ ANDRÉS | 1106423 | Quien programa | Escribe y ejecuta el código |
    | XIQUIN CUMES OSCAR DANIEL | 1118423 | Quien registra | Registra predicciones y resultados |
    | RUANO LAM JUAN LUIS DANIEL | 1247723 | Quien cuestiona | Pide predicciones y anota desacuerdos |

    **Roles:** distribución propuesta en el orden de nombres recibido; el grupo
    debe ajustarla si no corresponde al trabajo de cada integrante.
    **Estado:** análisis ejecutado; revisar el razonamiento en grupo antes de entregar.

    **Uso de IA.** Se utilizó Codex para leer el enunciado, preparar el código,
    proponer hipótesis antes de ejecutar, ejecutar las particiones, comprobar
    resultados y redactar un borrador de explicaciones. La IA aportó organización,
    reproducibilidad y apoyo para distinguir las situaciones que simula cada evaluación.
    Las hipótesis son propuestas con IA y no se atribuyen a una conversación
    del trío. El grupo debe revisar y poder explicar todas las respuestas.
    No se inventó una conversación con otro grupo.

    **Objetivo:** comparar particiones usando el mismo M1 (regresión logística)
    y M0 (DummyClassifier). El F1 corresponde a la clase positiva **baja = 1**,
    no al promedio macro ni a la exactitud. Cada F1 de M1 se presenta junto a M0.
    """)
    md("""
    ## Preparación: cargar y reconocer los datos
    Se usa una copia sin modificar del CSV de Descargas. No se eliminan
    duplicados ni se normalizan categorías en estos experimentos: la guía
    prescribe usar esta base y mantener comparabilidad entre grupos.
    Los problemas de calidad se documentan como limitaciones.

    Las hipótesis iniciales quedaron registradas en HIPOTESIS_PREVIAS.md antes
    del primer entrenamiento. Aquí se conservan antes de la celda de cada parte.
    """)
    code("""
    import os
    import sys
    import json
    import hashlib
    import platform
    import tempfile
    from pathlib import Path
    from importlib.metadata import version

    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'lab9-matplotlib'))
    import pandas as pd, numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display
    %matplotlib inline

    RAIZ = Path.cwd()
    RUTA_DATOS = RAIZ / 'data' / 'pedidos_ruta_verde.csv'
    if not RUTA_DATOS.exists():
        # Permite ejecutar desde output/ o con el CSV al lado del notebook.
        candidatos = [RAIZ.parent / 'data' / 'pedidos_ruta_verde.csv',
                      RAIZ / 'pedidos_ruta_verde.csv']
        RUTA_DATOS = next((p for p in candidatos if p.exists()), RUTA_DATOS)
    if not RUTA_DATOS.exists():
        raise FileNotFoundError('Coloque pedidos_ruta_verde.csv en data/ o junto al notebook.')
    if RUTA_DATOS.parent.name == 'data':
        RAIZ = RUTA_DATOS.parent.parent
    SALIDA = RAIZ / 'output'
    FIGURAS = SALIDA / 'graficas'
    FIGURAS.mkdir(parents=True, exist_ok=True)
    SEMILLAS = [0, 1, 7, 42, 2026]
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.precision', 4)
    pd.set_option('display.max_colwidth', 100)

    df = pd.read_csv(RUTA_DATOS, parse_dates=['fecha'])
    df.info()
    display(df.isna().sum().rename('Nulos').to_frame())
    display(df.head())
    """)
    code("""
    reconocimiento = pd.DataFrame({
        'Dato': ['Filas / pedidos registrados', 'IDs de pedido únicos', 'Restaurantes',
                 'Fecha mínima', 'Fecha máxima', 'Entrega mínima (min)',
                 'Entrega máxima (min)', 'Duplicados exactos', 'IDs repetidos'],
        'Valor': [len(df), df.pedido_id.nunique(), df.restaurante.nunique(),
                  str(df.fecha.min().date()), str(df.fecha.max().date()),
                  df.tiempo_entrega_min.min(), df.tiempo_entrega_min.max(),
                  int(df.duplicated().sum()), int(df.pedido_id.duplicated().sum())]
    })
    display(reconocimiento)
    display(df.restaurante.value_counts().rename('Pedidos').to_frame())
    fuera_reg = df[~((df.tiempo_entrega_min > 0) & (df.tiempo_entrega_min < 200))]
    display(fuera_reg[['pedido_id', 'tiempo_entrega_min']])
    """)
    md("Respuesta del reconocimiento: se completa con los resultados medidos.", "reconocimiento")
    md("""
    ### Las dos preguntas de negocio
    1. Clasificación: ¿este pedido recibirá calificación baja?
    2. Regresión: ¿cuántos minutos tardará?

    La guía prepara ambas bases, pero sus partes A-F fijan un clasificador y
    reportan F1. Por ello, se prepara reg tal como indica el enunciado y se
    responde la pregunta de disponibilidad de columnas; no se inventa un modelo
    de regresión ni un MAE que la guía no solicita.
    """)
    code("""
    # Pregunta 1 (clasificación): ¿este pedido recibirá calificación baja?
    cal = df.dropna(subset=['calificacion']).copy()
    cal['baja'] = (cal['calificacion'] <= 2).astype(int)
    print(cal['baja'].mean())

    # Pregunta 2 (regresión): ¿cuántos minutos tardará el pedido?
    reg = df[(df.tiempo_entrega_min > 0) & (df.tiempo_entrega_min < 200)].copy()
    print(f'Clasificación: {len(cal)} filas, {int(cal.baja.sum())} positivos, '
          f'{100 * cal.baja.mean():.2f}% de calificación baja.')
    print(f'Regresión: {len(reg)} filas; se excluyen {len(df) - len(reg)} tiempos por el filtro.')
    """)
    md("""
    ### Pregunta 1. ¿Qué columnas no pueden usarse al ingresar el pedido?
    | Columna | ¿Existe cuando entra el pedido? | Decisión y justificación |
    |---|---|---|
    | pedido_id | Sí, como identificador | No usar como señal de calidad: identifica una fila, no explica la entrega |
    | fecha | Sí, fecha/hora de creación | Puede usarse si representa la creación, no la fecha final de entrega |
    | departamento | Sí, del destino | Disponible si el destino ya está registrado |
    | municipio | Sí, del destino | Disponible al registrar la dirección |
    | zona | Sí, del destino | Disponible al registrar la dirección |
    | restaurante | Sí | El negocio seleccionado ya se conoce |
    | categoria_restaurante | Sí, del catálogo | Disponible para restaurantes registrados |
    | metodo_pago | Sí, si se elige al ordenar | Revisar que no sea el método final corregido después |
    | total_pedido_gtq | Sí, monto inicial | Usar el monto cotizado al ordenar, no un monto corregido después |
    | tiempo_entrega_min | **No** | Es el resultado futuro que queremos predecir; usarlo revelaría la respuesta |
    | distancia_km | Depende de cómo se mida | Una distancia estimada con direcciones conocidas sí; kilómetros realmente recorridos, no |
    | calificacion | **No** | El cliente califica después de recibir el pedido |
    | baja (derivada) | **No** | Se deriva de calificacion; es la respuesta del clasificador |

    Las exclusiones inequívocas para anticipar minutos son tiempo_entrega_min
    y calificacion; baja tampoco sería un insumo. La disponibilidad se decide
    por el momento de captura del dato, no solo por su nombre. El laboratorio
    no describe el origen exacto de distancia_km; esa disponibilidad requiere
    confirmación para producción. Para A-D se conservan las cinco variables
    que el docente fijó.
    """)
    md("""
    ## Definición del modelo
    Se copian los bloques M1 y M0 del enunciado sin cambiar parámetros ni variables.
    Cada evaluación crea un modelo nuevo. La mediana, la escala y las categorías
    se aprenden al llamar fit con el train, dentro del Pipeline.
    """)
    code(MODELO)
    code(BASELINE)
    md("""
    ### Pregunta 2. Predicción previa del F1 de M0
    **Hipótesis previa:** F1 = 0 si la clase mayoritaria del train es "no baja".
    Al responder siempre 0, M0 no encuentra verdaderos positivos de la clase 1.
    Tener muchos aciertos de la clase mayoritaria no implica un buen F1 para
    detectar las calificaciones bajas. La condición de mayoría se comprueba
    para cada partición, en lugar de asumirla sin verificar.
    """)
    code("""
    # Solo organiza tablas; no cambia los modelos ni las particiones.
    def tabla_semillas(resultados):
        tabla = resultados[['semilla', 'n_test', 'positivos_test', 'pct_positivos',
                            'f1', 'f1_baseline', 'recall']].copy()
        tabla.columns = ['Semilla', 'Pedidos test', 'Positivos test', '% positivos',
                         'F1 M1', 'F1 M0', 'Recall M1']
        minimo = {c: tabla[c].min() for c in tabla.columns if c != 'Semilla'}
        maximo = {c: tabla[c].max() for c in tabla.columns if c != 'Semilla'}
        rango = {'Semilla': 'mín - máx', **{
            c: f'{minimo[c]:.4f} - {maximo[c]:.4f}' if c in ['F1 M1', 'F1 M0', 'Recall M1']
            else f'{minimo[c]:.2f} - {maximo[c]:.2f}'
            for c in minimo}}
        return pd.concat([tabla, pd.DataFrame([rango])], ignore_index=True)

    def comprobar_particion(train, test):
        assert len(train) > 0 and len(test) > 0
        assert set(train.index).isdisjoint(test.index)
        assert len(train) + len(test) == len(cal)
        assert train.baja.nunique() == 2 and test.baja.nunique() == 2
        assert train.baja.mean() < 0.5, 'M0 ya no respondería siempre no baja.'

    registros_splits = []
    def registrar_split(nombre, train, test):
        # Se registra el índice original de cada fila para auditar las particiones.
        registros_splits.extend({'experimento': nombre, 'fila_csv': int(i),
                                 'pedido_id': int(cal.loc[i, 'pedido_id']), 'conjunto': conjunto}
                                for conjunto, bloque in [('train', train), ('test', test)]
                                for i in bloque.index)
    """)
    md("""
    ## Parte A. Cinco resultados con la misma base
    **Predicción previa:** los F1 serán parecidos, pero no idénticos. Las semillas
    cambian los pedidos de train/test y el número y la dificultad de los positivos.
    No esperamos que cambiar la semilla mejore por sí mismo el algoritmo.
    """)
    code("""
    from sklearn.model_selection import train_test_split

    filas = []
    for semilla in [0, 1, 7, 42, 2026]:
        train, test = train_test_split(cal, test_size=0.2, random_state=semilla)
        comprobar_particion(train, test)
        r = entrenar_y_medir(train, test)
        r['semilla'] = semilla
        r['f1_baseline'] = medir_baseline(train, test)
        r['pct_positivos'] = 100 * test.baja.mean()
        filas.append(r)
        registrar_split(f'A_{semilla}', train, test)

    resultados_a = pd.DataFrame(filas)
    display(tabla_semillas(resultados_a))
    resultados_a.to_csv(SALIDA / 'parte_A_aleatorio.csv', index=False)
    """)
    md("Interpretación de la Parte A y respuestas 3-5.", "A")
    md("""
    ## Parte B. Forzar la proporción
    **Predicción previa:** stratify estabilizará la proporción de positivos.
    Puede reducir la dispersión de F1, pero no garantiza elevar su promedio ni
    eliminar diferencias de dificultad entre pedidos.
    """)
    code("""
    filas = []
    for semilla in SEMILLAS:
        train, test = train_test_split(cal, test_size=0.2, random_state=semilla,
                                       stratify=cal['baja'])
        comprobar_particion(train, test)
        r = entrenar_y_medir(train, test)
        r['semilla'] = semilla
        r['f1_baseline'] = medir_baseline(train, test)
        r['pct_positivos'] = 100 * test.baja.mean()
        filas.append(r)
        registrar_split(f'B_{semilla}', train, test)

    resultados_b = pd.DataFrame(filas)
    display(tabla_semillas(resultados_b))
    resumen_ab = pd.DataFrame([
        {'Partición': nombre, 'F1 promedio M1': res.f1.mean(),
         'F1 promedio M0': res.f1_baseline.mean(), 'F1 mínimo M1': res.f1.min(),
         'F1 máximo M1': res.f1.max(), 'Rango F1 M1': res.f1.max() - res.f1.min(),
         'Desv. F1 M1 (muestral)': res.f1.std(ddof=1),
         '% positivos mínimo': res.pct_positivos.min(),
         '% positivos máximo': res.pct_positivos.max()}
        for nombre, res in [('A: aleatorio', resultados_a), ('B: estratificado', resultados_b)]
    ])
    display(resumen_ab)
    resultados_b.to_csv(SALIDA / 'parte_B_estratificado.csv', index=False)
    resumen_ab.to_csv(SALIDA / 'resumen_A_B.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(SEMILLAS))
    ax.plot(x, resultados_a.f1, 'o-', label='A: F1 M1 aleatorio', color='#0f766e')
    ax.plot(x, resultados_b.f1, 's-', label='B: F1 M1 estratificado', color='#b45309')
    ax.plot(x, resultados_a.f1_baseline, 'x--', color='#64748b',
            label='F1 M0 (A y B)')
    ax.set(xticks=x, xticklabels=SEMILLAS, xlabel='Semilla', ylabel='F1 de la clase baja',
           title='El mismo modelo: distintas particiones', ylim=(-0.02, 1))
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURAS / '01_semillas.png', bbox_inches='tight')
    plt.show()
    """)
    md("Interpretación de la Parte B y respuestas 6-8.", "B")
    md("""
    ## Parte C. Separar por fecha
    ### 1. Comportamiento mes a mes
    La tasa se calcula sobre pedidos con calificación observada. Se muestra
    también el número de pedidos sin calificación, para distinguir la tasa
    observada de una tasa desconocida sobre todos los pedidos.
    """)
    code("""
    mensual = cal.groupby(cal.fecha.dt.to_period('M')).baja.agg(
        pedidos_calificados='size', bajas='sum', proporcion='mean')
    mensual['pct_bajas'] = 100 * mensual.proporcion
    totales_mes = df.groupby(df.fecha.dt.to_period('M')).size()
    mensual['pedidos_totales'] = totales_mes
    mensual['sin_calificacion'] = mensual.pedidos_totales - mensual.pedidos_calificados
    display(mensual[['pedidos_totales', 'pedidos_calificados', 'sin_calificacion',
                     'bajas', 'pct_bajas']])
    mensual.to_csv(SALIDA / 'prevalencia_mensual.csv')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colores = ['#0f766e' if str(m) < '2026-07' else '#b45309' for m in mensual.index]
    barras = ax.bar(mensual.index.astype(str), mensual.pct_bajas, color=colores)
    ax.bar_label(barras, labels=[f'{v:.1f}%' for v in mensual.pct_bajas], padding=4)
    ax.set(xlabel='Mes', ylabel='% de pedidos calificados con nota ≤ 2',
           title='Calificaciones bajas por mes: enero-junio / julio-agosto',
           ylim=(0, min(100, mensual.pct_bajas.max() + 14)))
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURAS / '02_prevalencia_mensual.png', bbox_inches='tight')
    plt.show()
    """)
    md("""
    ### 2. Entrenar con el pasado y probar con el futuro
    **Predicción previa, conservada:** esperamos peor F1 que con el split
    aleatorio si el servicio cambia en los últimos meses. Esta hipótesis se
    registró antes de medir incluso la prevalencia mensual; no se corrige después
    de mirar la gráfica. El resultado decidirá si se cumple.
    """)
    code("""
    train_t = cal[cal.fecha < '2026-07-01']
    test_t = cal[cal.fecha >= '2026-07-01']
    comprobar_particion(train_t, test_t)
    assert train_t.fecha.max() < test_t.fecha.min()
    r_t = entrenar_y_medir(train_t, test_t)
    r_t['f1_baseline'] = medir_baseline(train_t, test_t)
    registrar_split('C_temporal', train_t, test_t)
    display(pd.DataFrame([{'Conjunto': nombre, 'Pedidos': len(bloque),
                           'Desde': bloque.fecha.min().date(), 'Hasta': bloque.fecha.max().date(),
                           'Positivos': int(bloque.baja.sum()), '% positivos': 100* bloque.baja.mean()}
                          for nombre, bloque in [('train', train_t), ('test', test_t)]]))
    display(pd.DataFrame([r_t]).rename(columns={'f1': 'F1 M1', 'f1_baseline': 'F1 M0'}))

    # Diagnóstico adicional: mismo M1 y misma partición; explica de dónde sale el F1.
    from sklearn.metrics import confusion_matrix, precision_score
    m_t = nuevo_modelo()
    m_t.fit(train_t[FEATURES], train_t.baja)
    pred_t = m_t.predict(test_t[FEATURES])
    tn, fp, fn, tp = confusion_matrix(test_t.baja, pred_t, labels=[0, 1]).ravel()
    diagnostico_t = {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
                    'precision': float(precision_score(test_t.baja, pred_t))}
    display(pd.DataFrame([diagnostico_t]))
    assert np.isclose(f1_score(test_t.baja, pred_t), r_t['f1'])
    """)
    md("Interpretación de la Parte C y respuestas 9-11.", "C")
    md("""
    ## Parte D. Separar por restaurante
    ### 1. Restaurantes presentes a ambos lados de un split aleatorio
    Se usa la semilla 42 de A para contar grupos compartidos y pedidos
    repetidos entre train/test, conservando las filas originales.
    """)
    code("""
    train_a42, test_a42 = train_test_split(cal, test_size=0.2, random_state=42)
    compartidos = sorted(set(train_a42.restaurante) & set(test_a42.restaurante))
    ids_compartidos = sorted(set(train_a42.pedido_id) & set(test_a42.pedido_id))
    print('Restaurantes de test presentes en train:', len(compartidos),
          'de', test_a42.restaurante.nunique())
    print(compartidos)
    print('IDs de pedido presentes a ambos lados:', len(ids_compartidos))
    print('Categorías por restaurante (pueden actuar como señales indirectas del grupo):')
    display(cal.groupby('restaurante').categoria_restaurante.unique().to_frame())
    """)
    md("""
    ### 2. Dejar dos restaurantes completos fuera
    **Predicción previa:** esperamos peor F1 al probar solo en dos restaurantes
    nunca vistos. Aunque restaurante no sea una variable de FEATURES, los
    pedidos del mismo negocio pueden compartir categoría, zona y patrones.
    """)
    code("""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=2/7, random_state=42)
    i_tr, i_te = next(gss.split(cal, cal['baja'], groups=cal['restaurante']))
    train_g, test_g = cal.iloc[i_tr], cal.iloc[i_te]
    comprobar_particion(train_g, test_g)
    assert set(train_g.restaurante).isdisjoint(test_g.restaurante)
    assert train_g.restaurante.nunique() == 5 and test_g.restaurante.nunique() == 2
    print('Train:', train_g.restaurante.unique())
    print('Test:', test_g.restaurante.unique())
    r_g = entrenar_y_medir(train_g, test_g)
    r_g['f1_baseline'] = medir_baseline(train_g, test_g)
    registrar_split('D_grupos', train_g, test_g)
    display(pd.DataFrame([{'Conjunto': nombre, 'Pedidos': len(bloque),
                           'Restaurantes': bloque.restaurante.nunique(),
                           'Positivos': int(bloque.baja.sum()), '% positivos': 100* bloque.baja.mean()}
                          for nombre, bloque in [('train', train_g), ('test', test_g)]]))
    display(pd.DataFrame([r_g]).rename(columns={'f1': 'F1 M1', 'f1_baseline': 'F1 M0'}))
    categorias_nuevas = {
        col: sorted(set(test_g[col].dropna()) - set(train_g[col].dropna()))
        for col in CAT
    }
    print('Valores categóricos del test no vistos en train:', categorias_nuevas)
    """)
    md("Interpretación de la Parte D y respuestas 12-13.", "D")
    md("""
    ## Parte E. Un atajo tentador
    **Predicción previa:** la versión 1 podría tener mayor F1 porque calcula el
    promedio con las respuestas del test. La existencia de fuga no garantiza
    un aumento visible del F1 en una partición concreta.

    Ambas versiones se comparan con **la misma semilla 42, el mismo stratify y
    las mismas filas**. También se muestra B con semilla 42 como control.
    Se conserva intacto nuevo_modelo para A-D. El enunciado pide ampliar NUM
    aquí; una fábrica separada añade solo riesgo_rest y conserva exactamente
    la imputación, la escala, la codificación y los parámetros de M1.
    """)
    code("""
    NUM_E = NUM + ['riesgo_rest']
    FEATURES_E = NUM_E + CAT

    def nuevo_modelo_e():
        prep = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('esc', StandardScaler())]), NUM_E),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CAT),
        ])
        return Pipeline([
            ('prep', prep),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ])

    def medir_e(train, test):
        m = nuevo_modelo_e()
        m.fit(train[FEATURES_E], train.baja)
        pred = m.predict(test[FEATURES_E])
        return {'f1': f1_score(test.baja, pred),
                'recall': recall_score(test.baja, pred),
                'n_test': len(test), 'positivos_test': int(test.baja.sum()),
                'f1_baseline': medir_baseline(train, test)}

    # Versión 1: calcular con TODO y luego partir.
    prom = cal.groupby('restaurante')['baja'].mean()
    cal_v1 = cal.assign(riesgo_rest=cal['restaurante'].map(prom))
    train_v1, test_v1 = train_test_split(cal_v1, test_size=0.2, random_state=42,
                                        stratify=cal_v1['baja'])

    # Versión 2: partir primero y calcular solo con train.
    train, test = train_test_split(cal, test_size=0.2, random_state=42,
                                   stratify=cal['baja'])
    prom_tr = train.groupby('restaurante')['baja'].mean()
    train_v2 = train.assign(riesgo_rest=train['restaurante'].map(prom_tr))
    test_v2 = test.assign(riesgo_rest=test['restaurante'].map(prom_tr))

    assert train_v1.index.equals(train_v2.index) and test_v1.index.equals(test_v2.index)
    comprobar_particion(train_v2, test_v2)
    assert NUM == ['distancia_km', 'total_pedido_gtq'] and len(FEATURES) == 5
    r_e1, r_e2 = medir_e(train_v1, test_v1), medir_e(train_v2, test_v2)
    registrar_split('E_v1', train_v1, test_v1)
    registrar_split('E_v2', train_v2, test_v2)
    b42 = resultados_b.loc[resultados_b.semilla == 42].iloc[0]
    tabla_e = pd.DataFrame([
        {'Versión': 'Control: B semilla 42, sin riesgo_rest', 'F1 M1': b42.f1,
         'F1 M0': b42.f1_baseline, 'Recall M1': b42.recall},
        {'Versión': 'E1: promedio con todo (fuga)', 'F1 M1': r_e1['f1'],
         'F1 M0': r_e1['f1_baseline'], 'Recall M1': r_e1['recall']},
        {'Versión': 'E2: promedio solo con train', 'F1 M1': r_e2['f1'],
         'F1 M0': r_e2['f1_baseline'], 'Recall M1': r_e2['recall']},
    ])
    display(tabla_e)
    display(pd.concat([prom.rename('Riesgo con todo'), prom_tr.rename('Riesgo solo train')],
                      axis=1))
    print('Restaurantes de test sin promedio en train:', int(test_v2.riesgo_rest.isna().sum()))
    tabla_e.to_csv(SALIDA / 'parte_E_promedio_restaurante.csv', index=False)
    """)
    md("Interpretación de la Parte E y respuestas 14-16.", "E")
    md("""
    ## Parte F. Recomendación para Ruta Verde
    La partición elegida debe corresponder al uso que el gerente quiera medir.
    Un F1 de calificaciones bajas no es una medida de error en minutos ni una
    probabilidad de éxito de cada pedido.
    """)
    md("Recomendación, escenarios y respuesta 17.", "F")
    md("""
    ## 12. Tabla comparativa
    La fila B presenta el **promedio aritmético de las cinco semillas**,
    no el F1 de un modelo único ni el F1 de predicciones concatenadas.
    La tabla E anterior conserva el control B42 para una comparación con E.
    """)
    code("""
    mejor_a = resultados_a.loc[resultados_a.f1.idxmax()]
    peor_a = resultados_a.loc[resultados_a.f1.idxmin()]
    comparativa = pd.DataFrame([
        {'Experimento': f'A: aleatorio, mejor semilla ({int(mejor_a.semilla)})',
         'F1 M1': mejor_a.f1, 'F1 M0': mejor_a.f1_baseline,
         'Situación real': 'Pedidos mezclados de negocios conocidos; resultado elegido por ser el mejor'},
        {'Experimento': f'A: aleatorio, peor semilla ({int(peor_a.semilla)})',
         'F1 M1': peor_a.f1, 'F1 M0': peor_a.f1_baseline,
         'Situación real': 'Misma población actual; muestra variación por partición'},
        {'Experimento': 'B: stratify, promedio de 5 semillas',
         'F1 M1': resultados_b.f1.mean(), 'F1 M0': resultados_b.f1_baseline.mean(),
         'Situación real': 'Comparar modelos con datos actuales y proporción de clases controlada'},
        {'Experimento': 'C: por fecha (julio-agosto)',
         'F1 M1': r_t['f1'], 'F1 M0': r_t['f1_baseline'],
         'Situación real': 'Entrenar con el pasado y predecir pedidos posteriores'},
        {'Experimento': 'D: por restaurante',
         'F1 M1': r_g['f1'], 'F1 M0': r_g['f1_baseline'],
         'Situación real': 'Pedidos de dos restaurantes no vistos en el entrenamiento'},
        {'Experimento': 'E: versión 1 (promedio con todo)',
         'F1 M1': r_e1['f1'], 'F1 M0': r_e1['f1_baseline'],
         'Situación real': 'Evaluación contaminada por respuestas del test; no reproduce producción'},
        {'Experimento': 'E: versión 2 (promedio con train)',
         'F1 M1': r_e2['f1'], 'F1 M0': r_e2['f1_baseline'],
         'Situación real': 'Negocios con historial en train; no valida por sí sola el futuro'},
    ])
    display(comparativa)
    comparativa.to_csv(SALIDA / 'tabla_comparativa.csv', index=False)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    etiquetas = ['A: mejor', 'A: peor', 'B: promedio', 'C: temporal',
                 'D: restaurantes', 'E1: con fuga', 'E2: solo train']
    y = np.arange(len(comparativa))
    colores = ['#0f766e', '#0f766e', '#0f766e', '#b45309', '#475569', '#b91c1c', '#0f766e']
    ax.barh(y - .16, comparativa['F1 M1'], height=.30, color=colores, label='M1')
    ax.scatter(comparativa['F1 M0'], y + .16, marker='x', color='#111827', label='M0')
    for i, fila in comparativa.iterrows():
        ax.text(fila['F1 M1'] + .012, i - .16, f"{fila['F1 M1']:.4f}", va='center')
        ax.text(.012, i + .16, f"M0 = {fila['F1 M0']:.4f}", va='center',
                color='#475569', fontsize=9)
    ax.set(yticks=y, yticklabels=etiquetas, xlim=(0, 1),
           xlabel='F1 de la clase baja', title='Qué desempeño mide cada experimento')
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURAS / '03_comparativa.png', bbox_inches='tight')
    plt.show()
    """)
    md("Reflexión para la clase.", "reflexion")
    md("""
    ## Comprobaciones y reproducibilidad
    Los controles siguientes verifican condiciones que afectan la interpretación:
    separación de filas, orden temporal, separación de grupos, mayoría de M0
    e identidad de filas entre E1 y E2. No se seleccionan semillas para elevar
    el resultado ni se cambia el umbral del clasificador.
    """)
    code("""
    datos_auditoria = {
        'filas': len(df), 'ids_unicos': int(df.pedido_id.nunique()),
        'restaurantes': int(df.restaurante.nunique()),
        'fecha_min': str(df.fecha.min().date()), 'fecha_max': str(df.fecha.max().date()),
        'tiempo_min': float(df.tiempo_entrega_min.min()),
        'tiempo_max': float(df.tiempo_entrega_min.max()),
        'nulos': {k: int(v) for k,v in df.isna().sum().items()},
        'duplicados': int(df.duplicated().sum()),
        'ids_repetidos': int(df.pedido_id.duplicated().sum()),
        'restaurantes_pedidos': {k: int(v) for k,v in df.restaurante.value_counts().items()},
        'n_cal': len(cal), 'n_reg': len(reg), 'positivos': int(cal.baja.sum()),
        'prevalencia': float(cal.baja.mean()),
        'tiempos_no_positivos': int((df.tiempo_entrega_min <= 0).sum()),
        'tiempos_ge_200': int((df.tiempo_entrega_min >= 200).sum()),
        'a': resultados_a.to_dict('records'), 'b': resultados_b.to_dict('records'),
        'temporal': r_t, 'grupos': r_g, 'e1': r_e1, 'e2': r_e2,
        'temporal_diagnostico': diagnostico_t,
        'n_train_t': len(train_t), 'prevalencia_train_t': float(train_t.baja.mean()),
        'prevalencia_test_t': float(test_t.baja.mean()),
        'restaurantes_test_g': sorted(test_g.restaurante.unique().tolist()),
        'n_train_g': len(train_g), 'prevalencia_train_g': float(train_g.baja.mean()),
        'prevalencia_test_g': float(test_g.baja.mean()),
        'categorias_nuevas_g': categorias_nuevas,
        'restaurantes_compartidos_a42': len(compartidos),
        'ids_compartidos_a42': len(ids_compartidos),
        'corr_positivos_f1_a': float(resultados_a.positivos_test.corr(resultados_a.f1)),
        'mensual': mensual.reset_index().assign(fecha=lambda x: x.fecha.astype(str)).to_dict('records'),
        'comparativa': comparativa.to_dict('records'),
        'sha256_csv': hashlib.sha256(RUTA_DATOS.read_bytes()).hexdigest(),
        'versiones': {p: version(p) for p in ['pandas', 'numpy', 'scikit-learn', 'matplotlib']},
        'python': platform.python_version(),
    }
    (SALIDA / 'resultados.json').write_text(json.dumps(datos_auditoria, ensure_ascii=False,
                                                     indent=2), encoding='utf-8')
    pd.DataFrame(registros_splits).to_csv(SALIDA / 'particiones_auditables.csv', index=False)
    print('Comprobaciones aprobadas: particiones, fecha, grupos y comparación E1/E2.')
    print('SHA-256 del CSV:', datos_auditoria['sha256_csv'])
    print('Versiones:', datos_auditoria['versiones'])
    print('Python:', datos_auditoria['python'])
    """)
    md("""
    ## Referencias y límites del análisis
    - **Enunciado:** Laboratorio 9 - Estrategias de Partición de Datos,
      Ing. Max Cerna, Universidad Rafael Landívar, segundo semestre de 2026.
    - **Datos:** pedidos_ruta_verde.csv proporcionado junto con los materiales locales.
    - [scikit-learn: validación y estrategias de partición](https://scikit-learn.org/stable/modules/cross_validation.html).
    - [scikit-learn: fuga de información y uso de Pipeline](https://scikit-learn.org/stable/common_pitfalls.html).
    - [scikit-learn: GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
    - [scikit-learn: TargetEncoder y codificación cruzada](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html).

    Las cinco semillas no son cinco muestras independientes de un negocio nuevo.
    Su dispersión es descriptiva, no un intervalo de confianza. Un solo corte
    temporal y dos negocios reservados tampoco garantizan el resultado de
    septiembre o de cualquier restaurante nuevo. Los nulos de calificación
    pueden introducir un sesgo si los clientes que no califican son distintos.
    Se conserva el CSV original para reproducir la guía; sus duplicados y
    variantes de categorías requieren revisión antes de una aplicación real.
    """)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3 (UV)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    })
    nbformat.write(nb, NOTEBOOK)
    return nb


def execute(nb):
    # Usa el intérprete de UV, sin instalar kernels globales en la máquina.
    with tempfile.TemporaryDirectory(prefix="lab9-kernel-") as temporary:
        kernel = Path(temporary) / "lab9"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3 (UV)", "language": "python",
        }), encoding="utf-8")
        km = KernelManager(kernel_name="lab9", transport="ipc",
                           kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        NotebookClient(nb, km=km, timeout=180, resources={"metadata": {"path": str(ROOT)}},
                       allow_errors=False).execute()
    nbformat.write(nb, NOTEBOOK)
    return nb


def html_export(nb):
    body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    (ROOT / "output" / "Laboratorio_9_Lectura.html").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    if "--ejecutar" in sys.argv:
        notebook = nbformat.read(NOTEBOOK, as_version=4)
    else:
        notebook = build()
    execute(notebook)
    html_export(notebook)
    print(f"Notebook ejecutado sin errores: {NOTEBOOK.name}")
