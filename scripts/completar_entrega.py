"""Redacta respuestas a partir de resultados reales y verifica la entrega."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "Laboratorio_9_Particion_Datos.ipynb"
r = json.loads((ROOT / "output" / "resultados.json").read_text(encoding="utf-8"))
a, b = pd.DataFrame(r["a"]), pd.DataFrame(r["b"])
t, g, e1, e2 = r["temporal"], r["grupos"], r["e1"], r["e2"]
best, worst = a.loc[a.f1.idxmax()], a.loc[a.f1.idxmin()]
delta_a = a.f1.max() - a.f1.min()
delta_b = b.f1.max() - b.f1.min()
nb = nbformat.read(NB_PATH, as_version=4)

# Respuestas concretas basadas en esta ejecución, sin reescribir las hipótesis.
respuestas = {
    "reconocimiento": f"""
### Reconocimiento del archivo
Hay **{r['filas']} filas de pedidos**, correspondientes a **{r['ids_unicos']} IDs
distintos**, de **{r['restaurantes']} restaurantes**, entre el
**1 de enero y el 31 de agosto de 2026**. La diferencia entre filas e IDs se
debe a **{r['duplicados']} duplicados exactos**.

Solo tienen nulos **distancia_km ({r['nulos']['distancia_km']})** y
**calificacion ({r['nulos']['calificacion']})**; las otras diez columnas no
tienen nulos. El mínimo de tiempo_entrega_min es **{r['tiempo_min']:.1f} minutos**
y el máximo **{r['tiempo_max']:.1f} minutos**. Hay **{r['tiempos_no_positivos']}
tiempos no positivos**, que son imposibles como duración de una entrega.
Los tiempos cercanos a 200 minutos son extremos, pero no son imposibles solo
por ser altos. El filtro indicado conserva **{r['n_reg']} filas** para reg;
no se aplican recortes adicionales.

Al quitar únicamente calificaciones faltantes quedan **{r['n_cal']} pedidos**:
**{r['positivos']} bajas** y **{r['n_cal'] - r['positivos']} no bajas**, con una
tasa de **{100 * r['prevalencia']:.2f}%**. Los tiempos negativos no se excluyen
de cal porque no son variables de M1 y el enunciado no pide ese filtro para
clasificación. No se imputan las respuestas faltantes.
""",
    "A": f"""
### Resultado frente a la predicción
El F1 varió de **{worst.f1:.4f}** (semilla {int(worst.semilla)}) a
**{best.f1:.4f}** (semilla {int(best.semilla)}), frente a **M0 = 0.0000**
en todas las semillas. Acertamos en que no serían idénticos, pero la expectativa
de resultados "parecidos" fue demasiado optimista: la amplitud es considerable
para un modelo de desempeño bajo.

### Pregunta 3. Diferencia entre mejor y peor F1
La diferencia es **{delta_a:.4f}**, equivalente a **{100 * delta_a:.2f} puntos**
si expresamos F1 en una escala de 0 a 100. El mejor resultado supera en más
del doble al peor. Es una diferencia grande para escoger un número de
desempeño con una única partición.

### Pregunta 4. Otro grupo obtiene un F1 diferente: ¿quién tiene razón?
Ambos resultados pueden ser correctos si usan distintas semillas y reproducen
el mismo procedimiento. Si usan las mismas filas, semilla y versiones deberían
coincidir; si no, habría que revisar limpieza, orden de filas, variables y
ejecución. No realizamos aquí una conversación con otro grupo; esa parte de
la actividad debe hacerse en clase.

Para describir la sensibilidad al reparto aleatorio reportaríamos el promedio
**M1 = {a.f1.mean():.4f}; M0 = {a.f1_baseline.mean():.4f}**, acompañado del
rango observado **M1 = {a.f1.min():.4f}-{a.f1.max():.4f}; M0 = 0.0000**,
las cinco semillas y el tamaño de test. No elegiríamos la mejor semilla
para anunciar desempeño. Este promedio es una descripción de estas cinco
particiones; no constituye una garantía ni un intervalo de confianza.
Para pedidos futuros escogeríamos otra evaluación, la temporal de C.

### Pregunta 5. Relación entre positivos de test y F1
Cada test contiene 119 pedidos, pero en A hay entre **{int(a.positivos_test.min())}
y {int(a.positivos_test.max())} positivos**, es decir, entre
**{a.pct_positivos.min():.2f}% y {a.pct_positivos.max():.2f}%**. Con pocos
positivos, un acierto o un error cambia mucho el resultado. La semilla 2026
contiene solo 3 positivos y da **M1 = 0.0952; M0 = 0.0000**, aunque su recall
es 0.6667: encontrar dos de tres casos no evita penalizar falsas alarmas.

En estas cinco observaciones hay asociación positiva descriptiva
(correlación ≈ {r['corr_positivos_f1_a']:.2f}), pero no una regla de que
"más positivos siempre produce mayor F1": la semilla 0 tiene 15 positivos y
la 1 tiene 12, y la 1 obtiene mayor F1. El resultado también depende de
verdaderos positivos, falsos positivos, falsos negativos y de los pedidos
que quedaron en train.
""",
    "B": f"""
### Pregunta 6. ¿Cambió el promedio, la diferencia entre semillas o ambos?
**Ambos.** El promedio pasa de **M1 = {a.f1.mean():.4f}; M0 = 0.0000** en A
a **M1 = {b.f1.mean():.4f}; M0 = 0.0000** en B, un aumento pequeño de
**{b.f1.mean() - a.f1.mean():.4f}** en F1. La amplitud entre semillas disminuye
de **{delta_a:.4f}** a **{delta_b:.4f}**, y la desviación muestral pasa de
**{a.f1.std(ddof=1):.4f}** a **{b.f1.std(ddof=1):.4f}**.

La predicción previa se cumple: B conserva **11 positivos en los 119 pedidos
de cada test (9.24%)**, mientras A varía de 3 a 15. El 9.24% es cercano al
9.14% global; las cantidades enteras impiden una igualdad exacta. Aunque la
dispersión baja, no desaparece.

### Pregunta 7. ¿Stratify hace que el modelo prediga mejor?
Stratify cambia el reparto para conservar aproximadamente la proporción de
clases en train y test. No agrega información, cambia el algoritmo ni garantiza
subir F1. Ayuda a que una comparación no dependa tanto de haber reservado muy
pocos positivos. En esta ejecución el promedio subió ligeramente, pero ese
resultado no es una garantía del método. Tampoco corrige por sí solo el orden
temporal, los restaurantes compartidos o los duplicados.

### Pregunta 8. ¿Dónde sería indispensable y dónde importaría menos?
Es especialmente necesario en clasificación con una clase muy rara, por
ejemplo fraude, fallas o calificaciones bajas, si un reparto al azar podría
dejar un test sin casos de interés. Requiere suficientes ejemplos de cada
clase; no crea positivos que no existen.

En un conjunto grande, equilibrado y sin dependencias temporales o por grupo,
la diferencia puede ser pequeña porque el azar ya conserva proporciones
parecidas. En regresión no existen clases naturales para estratificar.
Si el uso es temporal o en grupos nuevos, esa estructura determina la
partición, aunque las proporciones resulten distintas.
""",
    "C": f"""
### Pregunta 9. ¿Acertó la predicción y cómo se explica el resultado?
Sí: con fecha se obtuvo **M1 = {t['f1']:.4f}; M0 = {t['f1_baseline']:.4f}**,
inferior a los promedios aleatorio y estratificado. Se entrenó con
**{r['n_train_t']} pedidos de enero-junio** y se probó con **{t['n_test']}
pedidos de julio-agosto**, de los cuales **{t['positivos_test']}** son positivos.

La gráfica no muestra una subida gradual: julio tiene **1 baja de 76
calificados (1.32%)** y agosto **13 de 73 (17.81%)**. La tasa conjunta de
train es **{100*r['prevalencia_train_t']:.2f}%** y la de test
**{100*r['prevalencia_test_t']:.2f}%**, bastante parecidas. Por tanto, no basta
con decir que "hay más bajas": la composición de pedidos y su relación con
las variables pueden ser diferentes.

En test, M1 encuentra **1 de las 14 bajas**, pierde **13** y genera
**39 falsas alarmas**: recall **0.0714** y precisión **0.0250**.
Esto explica directamente el F1 bajo. El corte revela una transferencia
temporal pobre en esta muestra. Un cambio en el servicio o en los patrones
es una explicación plausible, pero este único corte no prueba su causa;
también intervienen tamaño, selección y dificultad de los pedidos.

### Pregunta 10. ¿Qué medición se parece al uso en septiembre?
La **temporal**, porque entrena con pedidos anteriores y evalúa pedidos
posteriores. No permite aprender de agosto para evaluar febrero. Para el gerente
reportaríamos el resultado de C como evidencia histórica relevante para
el uso futuro, indicando sus tamaños y su bajo recall. No lo presentaríamos
como un F1 ya medido en septiembre, porque el archivo termina en agosto.

### Pregunta 11. ¿Qué tiene de raro entrenar con agosto y probar febrero?
En febrero todavía no existían los pedidos de agosto. Una evaluación así
permite que el modelo aprenda del futuro de los casos evaluados, y responde a
una situación distinta de la operación real. El split aleatorio puede servir
para comparar modelos sobre la población actual, pero no mide fielmente
el reto de anticipar el siguiente periodo.
""",
    "D": f"""
### Pregunta 12. ¿Qué pasó con F1 y por qué?
En el split aleatorio de A con semilla 42, **los 7 restaurantes del test
también están en train**. La separación por grupos reserva **Burger Xelajú y
Café Chuwa** completos: hay **{r['n_train_g']} pedidos de 5 restaurantes en
train** y **{g['n_test']} pedidos de 2 restaurantes en test**, sin negocios
compartidos.

El resultado es **M1 = {g['f1']:.4f}; M0 = {g['f1_baseline']:.4f}**, frente a
**M1 = {a.f1.mean():.4f}; M0 = 0.0000** de promedio aleatorio y
**M1 = {b.f1.mean():.4f}; M0 = 0.0000** de promedio estratificado.
La hipótesis de una caída respecto de esos promedios se cumple; D no tiene
por qué ser menor que cada semilla individual, y supera la peor de A.

Aunque restaurante no está en FEATURES, sus pedidos comparten señales
como categoría y zona. **Cafetería** no aparece en el train de D y
OneHotEncoder ignora esa categoría nueva; también aparece la variante
**TRANSFERENCIA** sin ejemplos previos. Esto ofrece una explicación concreta
de la dificultad, sin demostrar que sea su única causa. Además, los positivos
son **{100*r['prevalencia_test_g']:.2f}% en test** y
**{100*r['prevalencia_train_g']:.2f}% en train**: el experimento cambia la
composición, no solo el número de negocios.

También hay **{r['ids_compartidos_a42']} IDs de pedido compartidos entre train y
test de A42** por duplicados exactos. Se conservan para seguir la guía, pero
esta repetición puede volver optimista una evaluación aleatoria. Antes de
una validación operativa se revisaría la identidad de los pedidos.

### Pregunta 13. ¿Qué medición describe mejor un restaurante nuevo?
La **D**, porque el modelo se evalúa en restaurantes de los que no vio pedidos.
Ese resultado estima mejor la transferencia a negocios nuevos que un test
que comparte los 7 restaurantes con train. No garantiza el desempeño de
cualquier nuevo negocio: solo se reservaron dos. Si además se predice un
periodo futuro, una evaluación operativa debería respetar ambas condiciones,
restaurante nuevo y fecha posterior.
""",
    "E": f"""
### Pregunta 14. ¿Cuál versión dio mejor y qué información tenía E1?
**Empataron:** E1 da **M1 = {e1['f1']:.4f}; M0 = {e1['f1_baseline']:.4f}** y
E2 **M1 = {e2['f1']:.4f}; M0 = {e2['f1_baseline']:.4f}**. El control B42 sin
riesgo_rest también da **M1 = {float(b.loc[b.semilla == 42, 'f1'].iloc[0]):.4f};
M0 = 0.0000**. La predicción de que E1 sería mayor **no se cumplió** y se
conserva en el notebook.

E1 conoce la etiqueta baja de los pedidos de test al calcular el promedio
por restaurante; incluso la etiqueta del pedido evaluado participa en el
valor que se usa para predecirlo. E2 calcula ese promedio solo con train.
Los promedios son diferentes, pero no producen una mejora de F1 en esta
partición. El empate de una métrica discreta no elimina la fuga de E1 ni
prueba que sus probabilidades sean iguales.

Una explicación plausible es que categoria_restaurante ya distingue muchos
negocios de esta base, por lo que riesgo_rest aporta información parcialmente
redundante. También puede cambiar probabilidades sin cambiar los aciertos
que cuenta F1. No alteramos semillas ni modelos para forzar que E1 gane.

### Pregunta 15. ¿Qué pasaría si E1 se presenta como resultado oficial?
En producción no se conocen las calificaciones de pedidos que todavía no
ocurren. No se puede construir el mismo promedio usando sus etiquetas.
El desempeño real puede ser menor o distinto del medido, porque se evaluó con
información inaccesible al momento de predecir. En este caso no observamos
una ventaja numérica de E1, pero el procedimiento sigue siendo inválido como
evidencia del desempeño esperado.

E2 evita usar respuestas del test. Sin embargo, este split aleatorio todavía
no reproduce fechas futuras. En una implementación real, el promedio debe
usar solo calificaciones que ya hayan llegado antes de predecir. Un negocio
nuevo tampoco tiene historial: map produciría un nulo y el Pipeline lo
imputaría con la mediana de train, un fallback que debe declararse y evaluarse.
En esta partición todos los negocios del test sí tienen historial en train.

### Pregunta 16. ¿Qué otro cálculo tendría un problema similar antes de partir?
Calcular la **mediana de imputación** o la **media y escala** sobre toda la
base antes del split haría participar al test en el preprocesamiento.
La codificación de categorías también debe ajustarse con train.
Estas fugas no utilizan la etiqueta como E1, pero sí información que la
evaluación reservó para simular datos nuevos.

En M1 el Pipeline evita ese problema: fit aprende SimpleImputer,
StandardScaler y OneHotEncoder usando solo train, y predict aplica esas
transformaciones al test.

**Límite adicional de E2:** el promedio de entrenamiento contiene la propia
etiqueta de cada fila de train. No es fuga de test, pero puede favorecer el
ajuste dentro del entrenamiento. Para un modelo de producción se consideraría
codificación cruzada o estadísticas históricas anteriores a cada pedido.
Se conserva la fórmula de E2 solicitada por la guía, sin agregar otro experimento.
""",
    "F": f"""
### Número que reportaríamos al gerente
Para anticipar pedidos futuros de Ruta Verde, reportaríamos
**F1 M1 = {t['f1']:.4f} frente a F1 M0 = {t['f1_baseline']:.4f}**,
obtenido entrenando con **{r['n_train_t']} pedidos de enero-junio** y evaluando
**{t['n_test']} pedidos de julio-agosto**. Hay **{t['positivos_test']} bajas
en test; el modelo solo detecta **1** (recall **7.14%**) y genera **39** falsas
alarmas (precisión **2.50%**).

La evidencia es débil para usar el modelo como detector de bajas:
superar un Dummy de F1 cero no basta para recomendarlo en operación. El gerente
debe recibir esta limitación junto al número. Es una medición histórica con
un único corte, no una garantía de septiembre. Antes de decidir una adopción
harían falta más periodos de evaluación y objetivos de negocio para tolerar
errores; esas mejoras quedan fuera de los experimentos solicitados.

### Forma de partir para los tres escenarios
| Escenario | Partición recomendada | Justificación |
|---|---|---|
| (a) Cada noche se predicen pedidos del día siguiente | Temporal, con ventanas sucesivas de pasado y futuro | Solo usar pedidos y calificaciones disponibles hasta esa noche; probar en fechas posteriores |
| (b) El modelo se usará en restaurantes nuevos | Por restaurante, como D | Reservar negocios completos; no reutilizar sus pedidos para aprender |
| (c) Comparar dos modelos con los datos actuales | Estratificada, con las mismas filas y semillas para ambos | Controlar la proporción de clases y comparar ambos bajo las mismas condiciones |

Para (c), esta recomendación supone una comparación dentro de la población
actual. Si la decisión final trata de uso futuro o restaurantes nuevos, también
se debe validar con la estructura de (a) o (b), respectivamente.
Las calificaciones pueden llegar después del pedido: una fecha pasada del pedido
no garantiza que su etiqueta ya existiera al hacer una predicción nocturna.

### Pregunta 17. ¿La medición más alta y la más confiable son la misma?
La más alta fue **A, semilla {int(best.semilla)}: M1 = {best.f1:.4f};
M0 = {best.f1_baseline:.4f}**. Para el objetivo de pedidos futuros, la más
pertinente y confiable metodológicamente es **C: M1 = {t['f1']:.4f};
M0 = {t['f1_baseline']:.4f}**, porque respeta el orden en que los datos estarían
disponibles. **No son la misma.**

La confiabilidad depende del escenario, no de qué número sea mayor. Para
restaurantes nuevos elegiríamos D; para comparar modelos actuales, B.
E1 no es una evaluación válida para producción aunque no supere a E2.
""",
    "reflexion": f"""
## 13. Para llevar a la clase
**Reflexión propuesta con apoyo de IA; revisar y discutir en el grupo.**

El hallazgo que más nos sorprendió fue que agregar un promedio de calificaciones
bajas calculado con toda la base no mejoró el F1 frente a calcularlo solo con
train: ambas versiones dieron **M1 = {e1['f1']:.4f}; M0 = 0.0000**.
Nuestra hipótesis inicial era que la versión con información del test obtendría
un resultado mayor. El empate nos obliga a separar dos preguntas: si el número
subió y si el procedimiento es válido. La versión 1 sigue usando respuestas
que no existirían al recibir un pedido, aun cuando la métrica no muestre
una ventaja. No sería correcto cambiar las semillas para fabricar el resultado
que esperábamos.

También fue importante observar la fragilidad del reparto aleatorio. Con el
mismo modelo, A varió entre **M1 = {a.f1.min():.4f}-{a.f1.max():.4f};
M0 = 0.0000**, y el test pasó de 3 a 15 calificaciones bajas. Al reservar
los meses posteriores, el desempeño fue **M1 = {t['f1']:.4f}; M0 = 0.0000**,
con solo una baja detectada de catorce. Esto sugiere que elegir el resultado
más alto no ayuda a tomar una decisión realista.

**Pregunta que nos queda:** ¿se repetiría la caída temporal al evaluar varios
periodos posteriores, y cuánto se explica por los tipos de pedidos frente a
cambios en el servicio? El contraste entre julio y agosto invita a revisar
esa cuestión, pero una sola partición no permite responderla.

**Regla propuesta para Ruta Verde, en una frase y sin tecnicismos:**  
Antes de confiar en una predicción, probémosla con pedidos que se parezcan a
los que llegarán mañana y sin mirar sus respuestas.
""",
}

for cell in nb.cells:
    tag = cell.metadata.get("respuesta")
    if tag in respuestas:
        cell.source = respuestas[tag].strip()

header = nb.cells[0].source
for old, new in [
    ("**PENDIENTE: nombre del integrante 1**", "GARCÍA ELÍAS JOSÉ ANDRÉS (1106423)"),
    ("**PENDIENTE: nombre del integrante 2**", "XIQUIN CUMES OSCAR DANIEL (1118423)"),
    ("**PENDIENTE: nombre del integrante 3**", "RUANO LAM JUAN LUIS DANIEL (1247723)"),
    ("| Integrante | Rol | Responsabilidad |", "| Integrante (carné) | Rol propuesto | Responsabilidad |"),
    ("**Estado:** análisis ejecutado; completar nombres y revisar el razonamiento en grupo antes de entregar.",
     "**Roles:** distribución propuesta en el orden recibido; revisar si corresponde al trabajo de cada integrante.\n\n"
     "**Estado:** análisis ejecutado; revisar el razonamiento en grupo antes de entregar."),
]:
    header = header.replace(old, new)
nb.cells[0].source = header
nbformat.validate(nb)
nbformat.write(nb, NB_PATH)

body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
(ROOT / "output" / "Laboratorio_9_Lectura.html").write_text(body, encoding="utf-8")

guide = f"""
# Guía para preparar la defensa oral

La defensa vale 30 puntos. Cada integrante debe poder explicar todas las partes.
Esta guía sirve para practicar; el notebook contiene los cálculos y las respuestas.

## Qué significa cada resultado

| Pregunta de uso | Evaluación | F1 M1 | F1 M0 |
|---|---|---:|---:|
| Describir variación del reparto actual | A, promedio de cinco semillas | {a.f1.mean():.4f} | 0.0000 |
| Comparar modelos con las clases representadas | B, promedio de cinco semillas | {b.f1.mean():.4f} | 0.0000 |
| Predecir pedidos posteriores | C, enero-junio → julio-agosto | {t['f1']:.4f} | 0.0000 |
| Atender restaurantes nuevos | D, dos negocios completos fuera | {g['f1']:.4f} | 0.0000 |
| Mostrar un promedio que conoce respuestas de test | E1, procedimiento con fuga | {e1['f1']:.4f} | 0.0000 |
| Promedio por negocio con respuestas de train | E2, mismo split que E1 | {e2['f1']:.4f} | 0.0000 |

## Preguntas para practicar

1. **¿Qué predice M1?** Si la nota será 1 o 2. No predice minutos en A-F.
2. **¿Por qué M0 tiene F1 cero?** La clase mayoritaria es no baja; M0 no predice
   positivos. No encuentra ninguna baja aunque acierte muchos pedidos normales.
3. **¿Qué mide F1?** Combina precisión y recall de la clase baja. La fórmula
   equivalente es 2 × TP / (2 × TP + FP + FN). No es exactitud.
4. **¿Qué cambió en A?** Solo las filas asignadas por cada semilla; el modelo
   y sus variables se conservaron.
5. **¿Qué hace stratify?** Mantiene proporciones aproximadas de clases:
   11 positivos en cada test, sin garantizar un mejor promedio de F1.
6. **¿Por qué C es relevante para septiembre?** Entrena con datos anteriores
   y prueba con posteriores. Septiembre todavía no se midió con este CSV.
7. **¿Cómo reconstruyo F1 de C?** TP=1, FP=39, FN=13:
   2/(2+39+13)=0.037037. Precision=1/40=0.025; recall=1/14=0.071429.
8. **¿La tasa baja subió de forma gradual?** No. Julio tiene 1.32% y agosto
   17.81%; train y test juntos tienen proporciones parecidas. No prueba una
   causa del deterioro.
9. **¿Por qué separar restaurantes si su nombre no es variable?** Sus pedidos
   comparten categoría, zona y otros patrones. Cafetería no aparece en train de D.
10. **¿Dónde está la fuga de E1?** El promedio usado como insumo conoce baja
    del test, incluida la respuesta del propio pedido evaluado.
11. **¿Por qué el empate E1/E2 no vuelve válida E1?** La validez depende de la
    información disponible al predecir, no de que suba una métrica particular.
12. **¿Qué evita Pipeline?** Aprende mediana, escala y categorías solo con
    train. No utiliza test para ajustar esos pasos.
13. **¿Qué número reporto al gerente?** Para futuros pedidos, C con M0 al lado,
    el tamaño de test y sus errores. F1 0.0370 no sostiene una recomendación
    favorable para detectar bajas.
14. **¿Qué problema de calidad hay?** 626 filas representan 620 IDs, hay seis
    duplicados exactos; en A42 dos IDs están a ambos lados. Esto limita una
    evaluación aleatoria. Se documenta y no se altera el ejercicio.
15. **¿Qué hipótesis falló?** E1 sería superior a E2: empataron. A fue menos
    estable de lo esperado. No se reescribieron las hipótesis.
16. **¿Qué aportó la IA?** Organización, hipótesis propuestas, código,
    verificación y borradores. El grupo debe poder reconstruir los cálculos
    y no atribuir a sí mismo una conversación que no ocurrió.

## Antes de entregar

- Revisar y confirmar los roles propuestos: José programa, Oscar registra,
  Juan Luis cuestiona; ajustarlos al trabajo real del grupo.
- Leer las 17 respuestas y discutir las hipótesis con las que discrepen.
- Mantener explícita la declaración de uso de IA.
- Hacer la conversación con otro grupo que pide la pregunta 4, si es posible,
  y añadir sus observaciones reales sin inventar nombres ni resultados.
- Entregar un notebook por trío, con sus salidas visibles.

## Material de lectura consultado

- [Validación y particiones, scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Fuga de información, scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html)
"""
(ROOT / "GUIA_DEFENSA_ORAL.md").write_text(guide.strip() + "\n", encoding="utf-8")

readme = """
# Laboratorio 9 - Estrategias de partición de datos

El entregable principal es Laboratorio_9_Particion_Datos.ipynb, ejecutado con
salidas y explicaciones. Contiene las partes A-F, las 17 preguntas,
predicciones anteriores a los entrenamientos, tabla comparativa y reflexión.
Los tres nombres y carnés ya están incluidos; los roles se proponen en el
orden recibido y deben ajustarse al trabajo real del grupo.

## Archivos

- Laboratorio_9_Particion_Datos.ipynb: archivo que pide el docente.
- output/Laboratorio_9_Lectura.html: versión para leer sin ejecutar Python.
- GUIA_DEFENSA_ORAL.md: práctica para los 30 puntos de defensa.
- HIPOTESIS_PREVIAS.md: registro de las hipótesis iniciales con apoyo de IA.
- data/pedidos_ruta_verde.csv: copia idéntica del archivo de Descargas.
- data/Enunciado_Laboratorio_9.pdf: copia del enunciado.
- output/: tablas CSV, tres gráficas y registro de resultados/particiones.
- pyproject.toml y uv.lock: dependencias y versiones fijadas con UV.

## Ejecutar con UV

Desde esta carpeta, abrir una terminal y ejecutar:

    uv sync --frozen
    uv run --frozen python scripts/preparar_entrega.py --ejecutar

La opción --ejecutar conserva las celdas de texto y vuelve a ejecutar el
notebook existente. Utiliza un kernel temporal con el Python del entorno de
UV; no instala kernels globales ni necesita Jupyter abierto. También actualiza
la versión HTML y las tablas. El notebook puede abrirse en Jupyter o VS Code
seleccionando el intérprete .venv/bin/python.

Si se cambia la base de datos, las celdas de texto con cifras deben revisarse.
No se recomienda cambiar el CSV para esta entrega.

## Reconstruir la entrega

Solo si se desea reconstruir desde las plantillas:

    uv run --frozen python scripts/preparar_entrega.py
    uv run --frozen python scripts/completar_entrega.py

Esta reconstrucción sustituye ediciones manuales del notebook. El segundo
paso está redactado para el CSV original y comprueba sus resultados antes de
crear el paquete; no debe utilizarse con otros datos.

## Criterios respetados

- M1 y M0 conservan los bloques del enunciado. E añade únicamente riesgo_rest
  en una fábrica separada, como pide esa parte de la guía.
- E1 y E2 usan idénticas filas, semilla 42 y estratificación.
- No se eliminan duplicados ni se normalizan categorías: se documentan para
  conservar el experimento indicado por el docente.
- reg se prepara con el filtro indicado; A-F evalúan clasificación con F1.
- Cada F1 M1 se acompaña de F1 M0.
- Las predicciones propuestas con IA se registraron antes del primer ajuste.
  No son una declaración ficticia de deliberación del grupo.
- No se inventó una conversación con otro grupo para la pregunta 4.

## Revisión humana final

Confirmar los roles reales, estudiar las explicaciones y agregar únicamente
observaciones auténticas de la discusión con otro grupo. La declaración de
uso de IA forma parte de las reglas del enunciado y ya está incluida.
"""
(ROOT / "README.md").write_text(readme.strip() + "\n", encoding="utf-8")


def verify():
    # Estos controles validan el trabajo solicitado, no cambian el experimento.
    from preparar_entrega import MODELO, BASELINE
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert any(c.source.strip() == MODELO.strip() for c in code_cells)
    assert any(c.source.strip() == BASELINE.strip() for c in code_cells)
    assert all(c.execution_count is not None for c in code_cells)
    errors = [o for c in code_cells for o in c.outputs if o.output_type == "error"]
    assert not errors, errors
    texts = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for question in range(1, 18):
        assert re.search(rf"Pregunta {question}\.", texts), f"Falta pregunta {question}"
    assert "PENDIENTE" not in texts
    assert all(name in texts for name in [
        "GARCÍA ELÍAS JOSÉ ANDRÉS", "XIQUIN CUMES OSCAR DANIEL",
        "RUANO LAM JUAN LUIS DANIEL", "1106423", "1118423", "1247723"])
    assert len(r["comparativa"]) == 7
    assert all(x["f1_baseline"] == 0 for x in r["a"] + r["b"])
    assert np.isclose(e1["f1"], e2["f1"])
    assert t["f1"] < a.f1.min()
    assert int(best.semilla) == 1 and int(worst.semilla) == 2026
    assert r["restaurantes_test_g"] == ["Burger Xelajú", "Café Chuwa"]
    assert r["filas"] == 626 and r["ids_unicos"] == 620
    assert r["mensual"][6]["bajas"] == 1 and r["mensual"][7]["bajas"] == 13
    import hashlib
    assert hashlib.sha256((ROOT / "data" / "pedidos_ruta_verde.csv").read_bytes()).hexdigest() == r["sha256_csv"]
    # Pares de semilla y orden de splits, sin cruzar filas.
    splits = pd.read_csv(ROOT / "output" / "particiones_auditables.csv")
    for _, split in splits.groupby("experimento"):
        assert split.fila_csv.nunique() == len(split) == r["n_cal"]
    v1 = splits[splits.experimento == "E_v1"].drop(columns="experimento").reset_index(drop=True)
    v2 = splits[splits.experimento == "E_v2"].drop(columns="experimento").reset_index(drop=True)
    pd.testing.assert_frame_equal(v1, v2)
    report = {
        "notebook_valido": True, "celdas_codigo_ejecutadas": len(code_cells),
        "errores_ejecucion": len(errors), "preguntas": 17,
        "filas_tabla_comparativa": 7, "modelo_original_verificado": True,
        "mismos_pedidos_E1_E2": True, "csv_original_sin_cambios": True,
        "roles": "propuestos; confirmar en el grupo",
        "discusion_otro_grupo": "no realizada por IA; pendiente de actividad humana",
    }
    (ROOT / "output" / "verificacion_entrega.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


verify()
archive = ROOT / "Entrega_Laboratorio_9.zip"
files = [NB_PATH, ROOT / "README.md", ROOT / "GUIA_DEFENSA_ORAL.md",
         ROOT / "HIPOTESIS_PREVIAS.md", ROOT / "pyproject.toml", ROOT / "uv.lock",
         ROOT / ".gitignore", *sorted((ROOT / "data").glob("*")),
         *sorted((ROOT / "output").rglob("*")), *sorted((ROOT / "scripts").glob("*.py"))]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
    for file in files:
        if file.is_file():
            package.write(file, Path("Laboratorio_9") / file.relative_to(ROOT))
with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
print(f"Paquete verificado: {archive.name}")
