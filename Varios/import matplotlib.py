import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)

# Variables dependientes
outcomes = [
    "Lenguaje", "Matemáticas", "Competencia digital", "Uso creativo digital",
    "Adicción smartphone", "Pervasividad smartphone", "Uso interactivo redes", "Satisfacción con la vida"
]

# Efectos por edad (comparado con edad ≤10)
edad_11 = [0.291, 0.171, 0.228, 0.219, -0.178, -0.344, 0.236, 0.028]
edad_12 = [0.252, 0.168, 0.266, 0.281, -0.251, -0.470, 0.318, -0.031]
edad_13 = [0.243, 0.124, 0.245, 0.207, -0.397, -0.562, 0.417, -0.045]

sig_11 = ["***", "**", "***", "***", "***", "***", "***", ""]
sig_12 = ["***", "**", "***", "***", "***", "***", "***", ""]
sig_13 = ["***", "*",  "***", "***", "***", "***", "***", ""]

# Efectos de covariables (constantes para todas las edades)
genero_masc = [-0.113, 0.313, -0.006, -0.054, -0.203, -0.201, 0.030, 0.313]
educ_padres_alta = [0.249, 0.282, 0.273, 0.137, -0.112, -0.153, 0.124, 0.108]
origen_inmigrante = [-0.333, -0.199, -0.261, -0.243, 0.101, 0.136, -0.094, -0.262]

# Crear DataFrames
df_efectos = pd.DataFrame({
    "11 años": edad_11,
    "12 años": edad_12,
    "13 o más años": edad_13
}, index=outcomes)

df_signif = pd.DataFrame({
    "11 años": sig_11,
    "12 años": sig_12,
    "13 o más años": sig_13
}, index=outcomes)

df_covars = pd.DataFrame({
    "Género masculino (vs. femenino)": genero_masc,
    "Padres con educación alta (vs. baja)": educ_padres_alta,
    "Origen inmigrante (vs. nativo)": origen_inmigrante
}, index=outcomes)

# Colores para líneas
colores = {
    "11 años": "tab:blue",
    "12 años": "tab:orange",
    "13 o más años": "tab:red"
}

# Graficar
fig, ax = plt.subplots()
for edad, color in colores.items():
    ax.plot(df_efectos.index, df_efectos[edad], marker="o", label=f"Edad {edad}", color=color)
    for i, outcome in enumerate(df_efectos.index):
        y = df_efectos.loc[outcome, edad]
        s = df_signif.loc[outcome, edad]
        if s:
            ax.annotate(s, (outcome, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=10)

# Graficar covariables como líneas horizontales por categoría
offsets = {"Género masculino (vs. femenino)": -0.02, "Padres con educación alta (vs. baja)": -0.04, "Origen inmigrante (vs. nativo)": -0.06}
linestyles = {"Género masculino (vs. femenino)": ":", "Padres con educación alta (vs. baja)": "--", "Origen inmigrante (vs. nativo)": "-."}
colors_cov = {"Género masculino (vs. femenino)": "gray", "Padres con educación alta (vs. baja)": "green", "Origen inmigrante (vs. nativo)": "brown"}

for covar in df_covars.columns:
    for i, outcome in enumerate(df_covars.index):
        y = df_covars.loc[outcome, covar] + offsets[covar]
        ax.hlines(y, i - 0.3, i + 0.3, colors=colors_cov[covar], linestyles=linestyles[covar], linewidth=1)
    ax.plot([], [], color=colors_cov[covar], linestyle=linestyles[covar], label=covar)

# Ajustes
plt.xticks(rotation=45, ha='right')
ax.axhline(0, color="black", linestyle="--", linewidth=1)
plt.ylabel("Cambio en desviaciones estándar")
plt.title("Impacto de edad de adquisición del smartphone y variables sociodemográficas\nsobre rendimiento, competencias, uso y bienestar")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


###########################################################################################################3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Estilo visual
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)

# Ejes X: edades
edades = ["Hasta 10", "11", "12", "13 o más"]

# Datos desde Tabla 2 (efectos marginales comparados con categoría de referencia)
genero_male = [-0.029, -0.014, 0.015, 0.028]
signif_genero = ["*", "*", "*", "*"]

origen_other = [0.000, 0.000, 0.000, 0.000]
signif_origen = ["", "", "", ""]  # no significativo

padres_educ_alt = [-0.044, -0.018, 0.023, 0.038]
signif_educ = ["*", "**", "*", "**"]

# Crear DataFrames
df = pd.DataFrame({
    "Edad de adquisición": edades,
    "Género masculino (ref: femenino)": genero_male,
    "Origen inmigrante (ref: nativo)": origen_other,
    "Padres educación alta (ref: baja)": padres_educ_alt
})
df = df.set_index("Edad de adquisición")

signif = pd.DataFrame({
    "Género masculino (ref: femenino)": signif_genero,
    "Origen inmigrante (ref: nativo)": signif_origen,
    "Padres educación alta (ref: baja)": signif_educ
}, index=edades)

# Gráfico
fig, ax = plt.subplots()
colors = ["tab:blue", "tab:orange", "tab:green"]

for i, col in enumerate(df.columns):
    ax.plot(df.index, df[col], marker='o', label=col, color=colors[i])
    for j, edad in enumerate(df.index):
        y = df[col][edad]
        asterisco = signif[col][edad]
        if asterisco:
            ax.annotate(asterisco, (edad, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=11)

# Estética
ax.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.title("Efectos marginales en la probabilidad de recibir el smartphone\npor grupo socio-demográfico (comparado con su referencia)")
plt.ylabel("Diferencia en probabilidad")
plt.xlabel("Edad al recibir el smartphone")
plt.legend(title="Grupo (categoría de referencia entre paréntesis)")
plt.tight_layout()
plt.show()
