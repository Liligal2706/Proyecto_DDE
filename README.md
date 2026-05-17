# 📊 Dashboard: Redes sociales vs productividad

Análisis completo de la relación entre uso de redes sociales y productividad mediante Diseño de Experimentos, con integración a **Google Stitch** para sincronización de datos.

## 🎯 Características

### Análisis Estadístico
- ✅ Análisis exploratorio (EDA)
- ✅ Muestreo bietápico con estimador Horvitz-Thompson
- ✅ Calibración por raking (IPF)
- ✅ Estimadores auxiliares (razón y regresión)
- ✅ Diseño en Bloques Completos Aleatorizados (DBCA)
- ✅ ANOVA y comparaciones múltiples (Tukey, Fisher LSD, Dunnett)
- ✅ Diseño factorial 2²
- ✅ Verificación de supuestos (normalidad, homocedasticidad, autocorrelación)
- ✅ Análisis de potencia estadística

### Integración de Datos
- 🌐 **Google Stitch** — Sincronización automática de fuentes de datos
- 📡 **MCP (Model Context Protocol)** — Conexión segura a servicios de Google
- 🔐 **Variables de entorno** — Manejo seguro de credenciales
- 📈 **Escalabilidad** — Soporte para grandes volúmenes de datos

## 🚀 Instalación y Setup

### 1. Clonar o descargar el proyecto
```bash
cd Proyecto_DDE
```

### 2. Crear e instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar Google Stitch (opcional)

#### Obtén tu API Key de Google Cloud

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea o selecciona tu proyecto
3. Habilita la API de Stitch
4. Crea una clave de API (restricción de HTTP referrer opcional)

#### Configura el archivo `.env`

Crea un archivo `.env` en la raíz del proyecto:

```env
GOOGLE_STITCH_API_KEY=tu_api_key_aqui
STITCH_MCP_URL=https://stitch.googleapis.com/mcp
```

⚠️ **IMPORTANTE:** El archivo `.env` está en `.gitignore` — NO se versionará.

### 4. Ejecutar el dashboard
```bash
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`

## 📑 Estructura de Pestañas

| Pestaña | Descripción |
|---------|-------------|
| 📋 **Resumen** | Resumen ejecutivo y metodología |
| 🔍 **Datos & Faltantes** | Análisis de missingness y raking |
| 📊 **Exploración** | EDA, distribuciones y correlaciones |
| 🧮 **Muestreo bietápico** | Diseño muestral y estimadores HT |
| 🧪 **DBCA** | Diseño en bloques y eficiencia |
| 📈 **ANOVA & Comparaciones** | Pruebas y contrastes múltiples |
| 🔢 **Diseño 2ᵏ** | Diseño factorial 2² |
| ✅ **Supuestos** | Verificación de normalidad y homocedasticidad |
| ⚡ **Potencia** | Análisis de potencia estadística |
| 🌐 **Integración Stitch** | Estado y fuentes de datos Stitch |
| 🔗 **Integración** | Síntesis de resultados |

## 📊 Datos

### Formato
- **Localización:** `data/` directory
- **Archivos:**
  - `social_media_vs_productivity.csv` — Base principal (~30,000 registros)
  - `datos_dbca_balanceado.csv` — Base para DBCA (opcional)

### Columnas principales
```
age, gender, job_type
daily_social_media_time, actual_productivity_score, perceived_productivity_score
stress_level, sleep_hours, screen_time_before_sleep, work_hours_per_day
number_of_notifications, uses_focus_apps, has_digital_wellbeing_enabled
...
```

## 🔧 Configuración

### `.streamlit/config.toml`
Controla temas, puertos y opciones de Streamlit:
```toml
[theme]
primaryColor = "#98C1D9"
backgroundColor = "#0B1220"
```

### Constantes en `app.py`
```python
CSV_PATH = Path("data/social_media_vs_productivity.csv")
ALPHA = 0.05                    # Nivel de significancia
SEED = 123                      # Para reproducibilidad
CLR = {...}                     # Paleta de colores
```

## 🌐 Integración Stitch

### ¿Qué es Stitch?
Google Stitch es una plataforma de integración que:
- Sincroniza datos automáticamente desde múltiples fuentes
- Realiza transformaciones ETL
- Se conecta a data warehouses (BigQuery, Postgres, etc.)
- Escala para volúmenes grandes

### Cómo usar en este proyecto

1. **Verifica el estado** → Pestaña "Integración Stitch"
2. **Configura fuentes** en Stitch Console
3. **Los datos se sincronizan** automáticamente
4. **Analiza en el dashboard** con datos frescos

### Módulo `stitch_integration.py`
```python
from stitch_integration import get_stitch_client, get_stitch_status

# Verificar estado
status = get_stitch_status()

# Obtener cliente
client = get_stitch_client()

# Listar integraciones
integrations = client.get_configured_integrations()
```

## 📝 Variables de Entorno

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `GOOGLE_STITCH_API_KEY` | API Key de Google Stitch | `AQ.Ab8RN...` |
| `STITCH_MCP_URL` | URL del MCP de Stitch | `https://stitch.googleapis.com/mcp` |

## 🔐 Seguridad

- ✅ Credenciales en `.env` (nunca en código)
- ✅ `.env` en `.gitignore`
- ✅ Variables de entorno con `python-dotenv`
- ✅ Headers seguros en requests
- ✅ Timeout de conexión configurable

## 📚 Dependencias Principales

- **streamlit** — Framework de UI
- **pandas** — Manipulación de datos
- **altair** — Visualizaciones
- **scipy/statsmodels** — Análisis estadístico
- **python-dotenv** — Variables de entorno
- **google-cloud-stitch** — Integración Stitch
- **requests** — HTTP client

## 🎨 Tema Visual

### Paleta de colores
```python
CLR = {
    "bg": "#0B1220",              # Fondo oscuro
    "panel": "#111827",           # Paneles
    "text": "#E5E7EB",            # Texto principal
    "blue": "#98C1D9",            # Azul principal
    "teal": "#7FBFBF",            # Teal
    "warn": "#F59E0B",            # Advertencia
    "ok": "#10B981",              # Éxito
    "err": "#EF4444",             # Error
}
```

## 📖 Autores

- Lina María Galvis Barragán
- Julián Mateo Valderrama Tibaduiza
- Docente: Javier Mauricio Sierra

**Universidad:** Santo Tomás  
**Curso:** Diseño de Experimentos  
**Período:** 2026-I

## 📄 Licencia

Este proyecto es educativo. Úsalo libremente respetando la autoría.

---

**Última actualización:** May 2026  
**Versión:** 3.0 (Con integración Stitch)