# 📊 Redes Sociales & Productividad

**Proyecto Investigativo — Muestreo y Diseño de Experimentos**  
Universidad Santo Tomás · Facultad de Estadística · 2026-I

| | |
|---|---|
| **Autores** | Lina María Galvis Barragán · Julián Mateo Valderrama Tibaduiza |
| **Docente** | Javier Mauricio Sierra |
| **Asignatura** | Diseño de Experimentos · VI Semestre |
| **Entregable** | Dashboard interactivo (Streamlit) |

---

## 🎯 Pregunta de investigación

> ¿El nivel de uso diario de redes sociales se asocia con diferencias significativas en la productividad real, controlando por tipo de trabajo?

**Hallazgo principal:** No se evidencia efecto práctico. Medias de productividad real por nivel de redes: 4.950 (Bajo), 4.966 (Medio), 4.929 (Alto) — diferencia máxima de 0.037 pts en escala 0–10. ANOVA: p ≈ 0.39, η² ≈ 0.

---

## 📁 Estructura del proyecto

```
proyecto/
├── app.py                  # Dashboard principal (Streamlit)
├── requirements.txt        # Dependencias
├── README.md
└── data/
    └── social_media_vs_productivity.csv
```

---

## 🚀 Instalación y ejecución

```bash
# 1. Clonar o descomprimir el proyecto
cd proyecto/

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
streamlit run app.py
```

El dashboard quedará disponible en `http://localhost:8501`.

---

## 📦 Dependencias

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
altair>=5.3.0
scipy>=1.13.0
statsmodels>=0.14.2
```

---

## 📐 Fases del proyecto (rúbrica)

El dashboard está organizado en 7 fases que corresponden a los criterios de evaluación:

| Tab | Fase | Contenido | Peso |
|-----|------|-----------|------|
| ① | **Fase 1 · Muestreo Básico** | Estratificado proporcional, bietápico HT, estimadores de razón y regresión | 10% |
| ② | **Fase 2 · DOE Básico** | DBCA, ANOVA Tipo II, Tukey / Fisher LSD / Dunnett, supuestos, potencia | 7% |
| ③ | **Fase 3 · Factorial** | 2² con bloqueo, efectos principales, interacción, pareto de efectos | 12% |
| ④ | **Fase 4 · Diseños 2ᵏ** | Contrastes, SS, gráfica de probabilidad normal de efectos, D-optimalidad | 12% |
| ⑤ | **Fase 5 · Bloqueo** | 2³ con confusión ABC, esquema generador, ANOVA con bloque, comparación con/sin | 10% |
| ⑥ | **Fase 6 · P. Desiguales** | PPS, estimador Horvitz-Thompson, comparación MAS vs HT-PPS | 10% |
| ⑦ | **Fase 7 · Encuestas** | Diseño complejo ensamblado, raking IPF, deff, cuantiles ponderados | 10% |
| 📋 | **Conclusiones** | Síntesis, pruebas excluidas con justificación, declaración de uso de IA | — |

---

## 📊 Dataset

- **Fuente:** `social_media_vs_productivity.csv`
- **Observaciones:** 30 000
- **Variables:** 19 (demográficas, conductuales y de productividad)
- **Variable respuesta:** `actual_productivity_score` (escala 0–10)
- **Faltantes:** ~8% en variables clave — mecanismo MCAR verificado

### Variables principales

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `daily_social_media_time` | Numérica | Horas diarias en redes sociales |
| `actual_productivity_score` | Numérica | Productividad real autoreportada (0–10) |
| `perceived_productivity_score` | Numérica | Productividad percibida (0–10) |
| `job_type` | Categórica | Tipo de trabajo (6 niveles) — usado como bloque |
| `uses_focus_apps` | Binaria | Uso de apps de enfoque (Sí/No) |
| `has_digital_wellbeing_enabled` | Binaria | Bienestar digital activado (Sí/No) |
| `nivel_redes` | Ordinal | Nivel de uso de redes: Bajo / Medio / Alto (derivada) |

---

## 🔬 Métodos estadísticos implementados

### Manejo de datos faltantes
- Verificación MCAR mediante χ² entre pares de indicadores de missingness
- Calibración por **Raking IPF** (Iterative Proportional Fitting) sobre género × job_type

### Diseño muestral (Fases 1, 6, 7)
- Muestreo estratificado proporcional con afijación por job_type
- Muestreo bietápico con estimador **Horvitz-Thompson** (π⁻¹)
- Estimadores auxiliares de razón y regresión (auxiliar: productividad percibida, r ≈ 0.96)
- Muestreo PPS (Probability Proportional to Size)
- Efecto de diseño (deff) y cuantiles ponderados

### Diseño experimental (Fases 2–5)
- **DBCA** (Diseño en Bloques Completos Aleatorizados): tratamiento = nivel de redes, bloque = job_type
- **ANOVA Tipo II** con verificación de supuestos
- Comparaciones múltiples: **Tukey HSD**, **Fisher LSD**, **Dunnett** (vs control)
- Eficiencia relativa del bloqueo (RE)
- Análisis de **potencia** y tamaño de muestra (curvas de potencia)
- **Diseño factorial 2²**: A = nivel redes, B = focus apps
- **Diseño 2³ con confusión ABC** en 2 bloques (contraste generador L = ABC)

### Pruebas de supuestos
- Normalidad: **Anderson-Darling** + Q-Q plot (Shapiro-Wilk excluido por límite n ≤ 5 000)
- Homocedasticidad: **Levene** (centro = mediana)
- Independencia: **Durbin-Watson**

---

## ⚠️ Pruebas excluidas y justificación

| Prueba | Razón de exclusión | Alternativa |
|--------|-------------------|-------------|
| Shapiro-Wilk | Límite n ≤ 5 000; n grande detecta trivialidades | Anderson-Darling + Q-Q + TCL |
| Prueba de Little | Requiere `pyampute` / normalidad multivariada | χ² entre indicadores de missingness |
| Esfericidad de Mauchly | Solo aplica a medidas repetidas | Levene + Durbin-Watson |
| Cuadrado Latino | Requiere 2 bloques ortogonales; solo tenemos job_type | DBCA + 2³ confusión |
| BIBD | Bloques completos disponibles → DBCA preferible | DBCA balanceado |
| MICE | Sin sesgo bajo MCAR; ganancia marginal con n = 30 000 | Raking IPF |
| Welch ANOVA | Levene no rechaza homocedasticidad | ANOVA clásico Tipo II |

---

## 🤖 Declaración de uso de IA

Se utilizó **Claude (Anthropic)** como apoyo en: depuración de código Python/Streamlit, implementación de funciones estadísticas (raking IPF, estimador HT, Anderson-Darling, confusión 2³) y redacción de interpretaciones.

Errores identificados y corregidos por los investigadores:
- La IA propuso Shapiro-Wilk para n grande → corregido a Anderson-Darling
- Variable global `C` colisionaba con `C()` de patsy → eliminado `C()` de fórmulas OLS
- Confusión entre estimadores Hansen-Hurwitz y Horvitz-Thompson → corregido

Todos los resultados fueron verificados contra la literatura de referencia. El criterio estadístico y la interpretación final son de los investigadores.

---

## 📚 Referencias

- Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley. — Caps. 5, 6, 7.
- Lohr, S. L. (2022). *Sampling: Design and Analysis* (3rd ed.). CRC Press. — Caps. 6, 7.
- Box, G. E. P., Hunter, J. S. & Hunter, W. G. (2005). *Statistics for Experimenters* (2nd ed.). Wiley.
- Lumley, T. (2011). *Complex Surveys: A Guide to Analysis Using R*. Wiley.

---

*Diseño de Experimentos · Universidad Santo Tomás · 2026-I*
