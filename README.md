# Laboratorio 9 - Estrategias de partición de datos

El entregable principal es Laboratorio_9_Particion_Datos.ipynb, ejecutado con
salidas y explicaciones. Contiene las partes A-F, las 17 preguntas,
predicciones anteriores a los entrenamientos, tabla comparativa y reflexión.
Los tres nombres y carnés ya están incluidos; los roles se proponen en el
orden recibido y deben ajustarse al trabajo real del grupo.

## Qué se entrega al docente

El enunciado solicita **un notebook (.ipynb) por trío**. En ese mismo archivo
deben aparecer las partes A-F con sus predicciones previas y resultados, las
17 respuestas en celdas de texto, la tabla comparativa y la reflexión para la
clase. También deben incluirse los tres nombres, los roles reales y la
declaración de uso de IA. La defensa oral se realiza aparte y vale 30 puntos.

GitHub sirve para compartir y conservar el historial del trabajo; el enunciado
no exige entregarlo por GitHub. Subir el notebook por el medio que indique el
docente. Si acepta un enlace al repositorio privado, darle acceso previamente.

## Archivos

- Laboratorio_9_Particion_Datos.ipynb: archivo que pide el docente.
- output/Laboratorio_9_Lectura.html: versión para leer sin ejecutar Python.
- GUIA_DEFENSA_ORAL.md: práctica para los 30 puntos de defensa.
- HIPOTESIS_PREVIAS.md: registro de las hipótesis iniciales con apoyo de IA.
- data/pedidos_ruta_verde.csv: copia idéntica del archivo de Descargas.
- data/Enunciado_Laboratorio_9.pdf: copia local del enunciado, excluida de Git.
- output/: tablas CSV, tres gráficas y registro de resultados/particiones.
- pyproject.toml y uv.lock: dependencias y versiones fijadas con UV.

El repositorio incluye el notebook ejecutado, el CSV, las guías, los scripts,
las dependencias y los resultados. Se excluyen el entorno .venv, las cachés,
los archivos temporales, el gráfico interno de código y el ZIP duplicado.

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
