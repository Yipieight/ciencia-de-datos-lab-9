# Guía para preparar la defensa oral

La defensa vale 30 puntos. Cada integrante debe poder explicar todas las partes.
Esta guía sirve para practicar; el notebook contiene los cálculos y las respuestas.

## Qué significa cada resultado

| Pregunta de uso | Evaluación | F1 M1 | F1 M0 |
|---|---|---:|---:|
| Describir variación del reparto actual | A, promedio de cinco semillas | 0.1694 | 0.0000 |
| Comparar modelos con las clases representadas | B, promedio de cinco semillas | 0.1769 | 0.0000 |
| Predecir pedidos posteriores | C, enero-junio → julio-agosto | 0.0370 | 0.0000 |
| Atender restaurantes nuevos | D, dos negocios completos fuera | 0.1034 | 0.0000 |
| Mostrar un promedio que conoce respuestas de test | E1, procedimiento con fuga | 0.2295 | 0.0000 |
| Promedio por negocio con respuestas de train | E2, mismo split que E1 | 0.2295 | 0.0000 |

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
