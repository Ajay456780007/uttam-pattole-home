import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Style Enhancements ------------------
sns.set_theme(style="darkgrid")            # Global theme
plt.rcParams["figure.figsize"] = (10, 5)   # Default figure size
# plt.rcParams["axes.edgecolor"] = "white"
# plt.rcParams["axes.labelcolor"] = "white"
# plt.rcParams["xtick.color"] = "#0F0F0F"
# plt.rcParams["ytick.color"] = "#0F0F0F"
# plt.rcParams["text.color"] = "#0F0F0F"
# plt.rcParams["axes.facecolor"] = "#0F0F0F"      # axis background
# plt.rcParams["figure.facecolor"] = "white"    # full figure background
# plt.rcParams["savefig.facecolor"] = "white"   # saved images background


# ------------------ Paths ------------------
DATASET_PATH = "Dataset/Rainfall_Dataset/archive/rainfall in india 1901-2015.csv"
SAVE_DIR = "ImageResults/Rainfall"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ Load Dataset ------------------
df = pd.read_csv(DATASET_PATH)

# ------------------ Filter ONE subdivision ------------------
df = df[df["SUBDIVISION"] == df["SUBDIVISION"].unique()[0]]

# ------------------ TIME SERIES PLOTS (5) ------------------

colors = ["#00E5FF", "#FF4081", "#69F0AE", "#FFAB40", "#7C4DFF"]

# 1️⃣ Annual Rainfall Trend
plt.plot(df["YEAR"], df["ANNUAL"], color=colors[0], linewidth=2.5, marker="o", markersize=3)
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.title("Annual Rainfall Trend")
plt.grid(True, alpha=0.3)
plt.savefig(f"{SAVE_DIR}/ts_annual.png", dpi=300)
plt.close()

# 2️⃣ January Rainfall Trend
plt.plot(df["YEAR"], df["JAN"], color=colors[1], linestyle="--", linewidth=2.2, marker="s", markersize=3)
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.title("January Rainfall Trend")
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/ts_jan.png", dpi=300)
plt.close()

# 3️⃣ April Rainfall Trend
plt.plot(df["YEAR"], df["APR"], color=colors[2], linestyle="-.", linewidth=2.2, marker="^", markersize=4)
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.title("April Rainfall Trend")
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/ts_apr.png", dpi=300)
plt.close()

# 4️⃣ July Rainfall Trend
plt.plot(df["YEAR"], df["JUL"], color=colors[3], linewidth=2.5, marker="d", markersize=4)
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.title("July Rainfall Trend")
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/ts_jul.png", dpi=300)
plt.close()

# 5️⃣ October Rainfall Trend
plt.plot(df["YEAR"], df["OCT"], color=colors[4], linewidth=2.5, linestyle=":", marker="o")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.title("October Rainfall Trend")
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/ts_oct.png", dpi=300)
plt.close()

# ------------------ SIMPLE PLOTS (5) ------------------

MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN",
          "JUL","AUG","SEP","OCT","NOV","DEC"]

# 6️⃣ Histogram of Annual Rainfall
plt.hist(df["ANNUAL"], bins=25, color="#FF6E40", edgecolor="black", alpha=0.8)
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.title("Annual Rainfall Distribution")
plt.grid(True, alpha=0.3)
plt.savefig(f"{SAVE_DIR}/hist_annual.png", dpi=300)
plt.close()

# 7️⃣ Boxplot of Monthly Rainfall
sns.boxplot(data=df[MONTHS], palette="coolwarm")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.title("Monthly Rainfall Boxplot")
plt.xticks(rotation=45)
plt.savefig(f"{SAVE_DIR}/boxplot_monthly.png", dpi=300)
plt.close()

# 8️⃣ Average Monthly Rainfall Bar Plot
monthly_avg = df[MONTHS].mean()
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette="viridis")
plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")
plt.title("Average Monthly Rainfall")
plt.xticks(rotation=45)
plt.savefig(f"{SAVE_DIR}/bar_monthly_avg.png", dpi=300)
plt.close()

# 9️⃣ Scatter Plot: Year vs Annual Rainfall
plt.scatter(df["YEAR"], df["ANNUAL"], color="#00E676", s=30, alpha=0.8, edgecolors="black")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.title("Year vs Annual Rainfall")
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/scatter_year_annual.png", dpi=300)
plt.close()

# 🔟 Correlation Heatmap
corr = df[MONTHS + ["ANNUAL"]].corr()
sns.heatmap(corr, annot=True, cmap="magma", fmt=".2f")
plt.title("Rainfall Feature Correlation")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/correlation_heatmap.png", dpi=300)
plt.close()

print("🎨✨ 10 styled rainfall plots saved in:", SAVE_DIR)




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ GLOBAL STYLE ------------------
sns.set_theme(style="darkgrid")
# plt.rcParams["figure.figsize"] = (10, 5)
# plt.rcParams["axes.labelcolor"] = "white"
# plt.rcParams["text.color"] = "white"
# plt.rcParams["xtick.color"] = "white"
# plt.rcParams["ytick.color"] = "white"

# Color palette
colors = sns.color_palette("viridis", 10)

# ------------------ CONFIG ------------------
DATA_PATH = "Dataset/Temperature_Dataset/archive/train_dataset.csv"
SAVE_DIR = "ImageResults/Temperature"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH)

print("Columns:", df.columns.tolist())
print(df.head())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", cat_cols)

# Helper function
def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, filename), dpi=320)
    plt.close(fig)

# -------------------------------------------------
# 1️⃣ Histogram of First Numeric Column (Styled)
# -------------------------------------------------
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, color=colors[0], ax=ax)
    ax.set_title(f"Histogram of {col}")
    save_plot(fig, f"hist_{col}.png")

# -------------------------------------------------
# 2️⃣ Boxplot of First Numeric Column
# -------------------------------------------------
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], color=colors[1], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    save_plot(fig, f"box_{col}.png")

# -------------------------------------------------
# 3️⃣ Histogram of Second Numeric Column
# -------------------------------------------------
if len(numeric_cols) > 1:
    col = numeric_cols[1]
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, color=colors[2], ax=ax)
    ax.set_title(f"Histogram of {col}")
    save_plot(fig, f"hist2_{col}.png")

# -------------------------------------------------
# 4️⃣ Boxplot of Second Numeric Column
# -------------------------------------------------
if len(numeric_cols) > 1:
    col = numeric_cols[1]
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], color=colors[3], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    save_plot(fig, f"box2_{col}.png")

# -------------------------------------------------
# 5️⃣ Correlation Heatmap (Styled)
# -------------------------------------------------
if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = df[numeric_cols].corr()

    sns.heatmap(
        corr, annot=True, cmap="magma", fmt=".2f",
        linewidths=0.5, ax=ax
    )

    ax.set_title("Correlation Heatmap (Numeric Columns)")
    save_plot(fig, "corr_heatmap.png")

# -------------------------------------------------
# 6️⃣ Scatter Plot (Good Colors)
# -------------------------------------------------
if len(numeric_cols) >= 2:
    xcol, ycol = numeric_cols[0], numeric_cols[1]
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[xcol], y=df[ycol], color=colors[4], s=40, edgecolor="black", ax=ax)
    ax.set_title(f"Scatter: {xcol} vs {ycol}")
    save_plot(fig, f"scatter_{xcol}_vs_{ycol}.png")

# -------------------------------------------------
# 7️⃣ Count Plot of First Categorical Column
# -------------------------------------------------
if len(cat_cols) > 0:
    col = cat_cols[0]
    top_vals = df[col].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=top_vals.index, y=top_vals.values, ax=ax, palette="coolwarm")
    ax.set_xticklabels(top_vals.index, rotation=90)
    ax.set_title(f"Count Plot: {col}")
    save_plot(fig, f"count_{col}.png")

# -------------------------------------------------
# 8️⃣ Numeric vs Categorical Boxplot
# -------------------------------------------------
if len(numeric_cols) > 0 and len(cat_cols) > 0:
    num = numeric_cols[0]
    cat = cat_cols[0]
    top = df[cat].value_counts().index[:10]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(x=df[cat], y=df[num], order=top, palette="viridis", ax=ax)
    ax.set_xticklabels(top, rotation=45)
    ax.set_title(f"{num} Distribution by {cat}")
    save_plot(fig, f"box_num_by_cat.png")

# -------------------------------------------------
# 9️⃣ Pairwise Scatter matrix (Styled)
# -------------------------------------------------
if len(numeric_cols) >= 2:
    use_cols = numeric_cols[:4]

    sns_plot = sns.pairplot(df[use_cols], diag_kind="kde", palette="viridis")
    sns_plot.fig.suptitle("Pairwise Scatter Matrix", y=1.02)

    sns_plot.savefig(os.path.join(SAVE_DIR, "pairwise_scatter_matrix.png"), dpi=320)
    plt.close()

# -------------------------------------------------
# 🔟 Missing Values Bar Plot
# -------------------------------------------------
missing_counts = df.isna().sum()
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(x=missing_counts.index, y=missing_counts.values, palette="rocket", ax=ax)
ax.set_xticklabels(missing_counts.index, rotation=90)
ax.set_title("Missing Values Per Column")
save_plot(fig, "missing_values.png")

print(f"🎨✨ Styled Temperature Plots saved in: {SAVE_DIR}")





import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # ------------------ GLOBAL STYLE ------------------
sns.set_theme(style="darkgrid")
# plt.rcParams["figure.figsize"] = (10, 5)
# plt.rcParams["axes.labelcolor"] = "white"
# plt.rcParams["text.color"] = "white"
# plt.rcParams["xtick.color"] = "white"
# plt.rcParams["ytick.color"] = "white"

# Color palettes
pal = sns.color_palette("viridis", 10)
cool = sns.color_palette("coolwarm", 10)

# ------------------ CONFIG ------------------
DATA_PATH = "Dataset/Wind_Dataset/archive/wind_dataset.csv"
SAVE_DIR = "ImageResults/Wind"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ------------------ HELPER FUNCTION ------------------
def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, name), dpi=320)
    plt.close(fig)

# ==================================================
#                   PLOTS (10)
# ==================================================

# 1️⃣ Stylish Histogram + KDE
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=30, kde=True, color=pal[0], ax=ax)
    ax.set_title(f"Histogram of {col}")
    save_plot(fig, f"wind_01_hist_{col}.png")

# 2️⃣ Stylish Boxplot
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], color=pal[1], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    save_plot(fig, f"wind_02_box_{col}.png")

# 3️⃣ Second Numeric Histogram
if len(numeric_cols) > 1:
    col = numeric_cols[1]
    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=30, kde=True, color=pal[2], ax=ax)
    ax.set_title(f"Histogram of {col}")
    save_plot(fig, f"wind_03_hist_{col}.png")

# 4️⃣ Second Numeric Boxplot
if len(numeric_cols) > 1:
    col = numeric_cols[1]
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], color=pal[3], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    save_plot(fig, f"wind_04_box_{col}.png")

# 5️⃣ Correlation Heatmap (Beautiful)
if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="magma", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap (Wind Dataset)")
    save_plot(fig, "wind_05_correlation_heatmap.png")

# 6️⃣ Scatter Plot (Modern look)
if len(numeric_cols) >= 2:
    xcol, ycol = numeric_cols[0], numeric_cols[1]
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[xcol], y=df[ycol], s=40, color=pal[4], edgecolor="black", ax=ax)
    ax.set_title(f"Scatter: {xcol} vs {ycol}")
    save_plot(fig, f"wind_06_scatter_{xcol}_vs_{ycol}.png")

# 7️⃣ Category Count with Coolwarm Palette
if len(categorical_cols) > 0:
    col = categorical_cols[0]
    top = df[col].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=top.index.astype(str), y=top.values, palette="coolwarm", ax=ax)
    ax.set_xticklabels(top.index.astype(str), rotation=90)
    ax.set_title(f"Category Count: {col}")
    save_plot(fig, f"wind_07_bar_{col}.png")

# 8️⃣ Boxplot: Numeric vs Categorical
if len(numeric_cols) > 0 and len(categorical_cols) > 0:
    num = numeric_cols[0]
    cat = categorical_cols[0]
    top_cats = df[cat].value_counts().index[:10]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df[df[cat].isin(top_cats)], x=cat, y=num, palette="viridis", ax=ax)
    ax.set_title(f"{num} by {cat}")
    ax.set_xticklabels(top_cats, rotation=45)
    save_plot(fig, f"wind_08_box_{num}_by_{cat}.png")

# 9️⃣ Pairwise Scatter (3 stylish pairs)
from itertools import combinations
pairs = list(combinations(numeric_cols[:4], 2))[:3]

for i, (xcol, ycol) in enumerate(pairs, start=1):
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[xcol], y=df[ycol], color=pal[i], s=50, edgecolor="black", ax=ax)
    ax.set_title(f"Scatter: {xcol} vs {ycol}")
    save_plot(fig, f"wind_09_pair_{i}_{xcol}_vs_{ycol}.png")

# 🔟 Missing Value Bar Plot
missing = df.isna().sum()
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(x=missing.index, y=missing.values, palette="rocket", ax=ax)
ax.set_xticklabels(missing.index, rotation=90)
ax.set_title("Missing Values per Column")
save_plot(fig, "wind_10_missing_values.png")

print("🎨✨ Styled Wind dataset visualizations saved in:", SAVE_DIR)
