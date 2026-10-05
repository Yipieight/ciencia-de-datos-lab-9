# Hipótesis registradas antes de ejecutar los modelos

Registro inicial: 5 de octubre de 2026. Estas son hipótesis propuestas con ayuda
de Codex antes de la primera ejecución de los experimentos, no declaraciones
de una discusión realizada por el grupo. Se conservan aunque el resultado las
contradiga. Los integrantes deben revisarlas y explicar su razonamiento.

- **Pregunta 2 / M0:** esperamos F1 = 0 para la clase baja si el entrenamiento
  tiene como mayoría a los pedidos sin calificación baja. Al no predecir
  positivos, no encuentra ningún caso positivo verdadero.
- **Parte A:** esperamos F1 parecidos, pero no idénticos. Cambiar la semilla
  cambia los pedidos de entrenamiento y prueba y puede alterar la cantidad
  y dificultad de los positivos del test.
- **Parte B:** esperamos una proporción de positivos más estable y, posiblemente,
  una menor dispersión del F1. No esperamos una mejora garantizada del promedio.
- **Parte C:** esperamos peor F1 que con la partición aleatoria si el servicio
  cambia durante los últimos meses. Esto es una hipótesis; todavía no se ha
  medido la prevalencia mensual ni el desempeño temporal.
- **Parte D:** esperamos peor F1 al evaluar restaurantes nunca vistos, porque
  sus características y patrones pueden diferir de los del entrenamiento.
- **Parte E:** esperamos mayor F1 en la versión 1 porque su promedio incluye
  las respuestas del test. Una fuga puede volver optimista una evaluación,
  pero no garantiza un F1 mayor en cada partición concreta.

La predicción de la parte C se registra antes de observar su gráfica mensual;
el notebook presenta primero la gráfica y luego recuerda esta hipótesis antes
de entrenar. No se reescribe para acomodarla a los resultados.
