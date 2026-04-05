import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4d',
    'axes.labelcolor':  '#e0e0e0',
    'axes.titlecolor':  '#ffffff',
    'xtick.color':      '#a0a0b0',
    'ytick.color':      '#a0a0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3a',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

PALETTE   = ['#6c63ff', '#ff6584', '#43e97b', '#f9c74f', '#4cc9f0']
ACCENT    = '#6c63ff'
ACCENT2   = '#ff6584'
POSITIVE  = '#43e97b'
WARNING   = '#f9c74f'

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/titanic.csv')
print(f"Dataset shape: {df.shape}")
print(df.dtypes)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & Missing Values
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

# Title
fig.text(0.5, 0.97, '🚢  Titanic — Exploratory Data Analysis',
         ha='center', va='top', fontsize=22, fontweight='bold', color='white')
fig.text(0.5, 0.935, f'891 passengers  ·  12 features  ·  38.4 % survived',
         ha='center', va='top', fontsize=13, color='#a0a0b0')

# ── 1. Survival count ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
surv_counts = df['survived'].value_counts().sort_index()
bars = ax1.bar(['Did Not\nSurvive', 'Survived'], surv_counts.values,
               color=[ACCENT2, POSITIVE], width=0.5, zorder=3)
for b in bars:
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 8,
             f'{int(b.get_height())}', ha='center', fontsize=12, color='white', fontweight='bold')
ax1.set_title('Survival Count', fontsize=13, pad=10)
ax1.set_ylim(0, 620)
ax1.grid(axis='y', zorder=0)
ax1.set_yticks([])

# ── 2. Survival % donut ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sizes  = [surv_counts[0], surv_counts[1]]
colors = [ACCENT2, POSITIVE]
wedges, _ = ax2.pie(sizes, colors=colors, startangle=90,
                    wedgeprops=dict(width=0.55, edgecolor='#0f1117', linewidth=2))
ax2.text(0, 0, f"{surv_counts[1]/len(df)*100:.1f}%\nsurvived",
         ha='center', va='center', fontsize=13, color='white', fontweight='bold')
ax2.set_title('Survival Rate', fontsize=13)
ax2.legend(['Did not survive', 'Survived'], loc='lower center',
           bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=9, ncol=2)

# ── 3. Missing values heatmap ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
key_cols  = ['age', 'embarked', 'fare', 'sex', 'pclass', 'sibsp', 'parch', 'survived']
miss_pct  = df[key_cols].isnull().mean() * 100
colors_mv = [WARNING if v > 20 else (ACCENT2 if v > 0 else POSITIVE) for v in miss_pct.values]
bars3 = ax3.barh(key_cols, miss_pct.values, color=colors_mv, height=0.6, zorder=3)
for b, v in zip(bars3, miss_pct.values):
    ax3.text(v + 0.5, b.get_y() + b.get_height()/2,
             f'{v:.1f}%', va='center', fontsize=9, color='white')
ax3.set_xlim(0, 85)
ax3.set_title('Missing Values (%)', fontsize=13)
ax3.grid(axis='x', zorder=0)

# ── 4. Age distribution ───────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
surv0 = df[df['survived'] == 0]['age'].dropna()
surv1 = df[df['survived'] == 1]['age'].dropna()
ax4.hist(surv0, bins=28, alpha=0.7, color=ACCENT2, label='Did not survive', zorder=3)
ax4.hist(surv1, bins=28, alpha=0.7, color=POSITIVE, label='Survived',        zorder=3)
ax4.axvline(df['age'].median(), color=WARNING, lw=1.5, ls='--', label=f'Median {df["age"].median():.0f}')
ax4.set_title('Age Distribution by Survival', fontsize=13)
ax4.set_xlabel('Age'); ax4.grid(axis='y', zorder=0)
ax4.legend(fontsize=8)

# ── 5. Fare distribution (log) ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
fare_data = df['fare'].dropna()
ax5.hist(np.log1p(fare_data), bins=35, color=ACCENT, alpha=0.85, zorder=3)
ax5.axvline(np.log1p(fare_data.median()), color=WARNING, lw=1.5, ls='--',
            label=f'Median £{fare_data.median():.1f}')
ax5.set_title('Fare Distribution (log scale)', fontsize=13)
ax5.set_xlabel('log(Fare + 1)'); ax5.grid(axis='y', zorder=0)
ax5.legend(fontsize=9)

# ── 6. Passenger class survival ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
pclass_surv = df.groupby(['pclass', 'survived']).size().unstack(fill_value=0)
pclass_surv_pct = pclass_surv.div(pclass_surv.sum(axis=1), axis=0) * 100
bottom = np.zeros(3)
for i, (col, clr) in enumerate(zip([0, 1], [ACCENT2, POSITIVE])):
    vals = pclass_surv_pct[col].values
    ax6.bar(['1st', '2nd', '3rd'], vals, bottom=bottom, color=clr, label=['Not survived', 'Survived'][i], width=0.5)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 6:
            ax6.text(j, b + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    bottom += vals
ax6.set_title('Survival Rate by Class', fontsize=13)
ax6.set_ylabel('Passengers (%)')
ax6.legend(fontsize=9); ax6.grid(axis='y', zorder=0)

# ── 7. Sex vs Survival ────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
sex_surv = df.groupby(['sex', 'survived']).size().unstack(fill_value=0)
sex_surv_pct = sex_surv.div(sex_surv.sum(axis=1), axis=0) * 100
bottom = np.zeros(2)
labels = sex_surv_pct.index.tolist()
for col, clr in zip([0, 1], [ACCENT2, POSITIVE]):
    vals = sex_surv_pct[col].values
    ax7.bar(labels, vals, bottom=bottom, color=clr, width=0.4)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax7.text(j, b + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    bottom += vals
ax7.set_title('Survival Rate by Sex', fontsize=13)
ax7.legend(['Not survived', 'Survived'], fontsize=9)
ax7.grid(axis='y')

# ── 8. Embarkation ────────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
emb_surv = df.groupby(['embarked', 'survived']).size().unstack(fill_value=0)
emb_surv_pct = emb_surv.div(emb_surv.sum(axis=1), axis=0) * 100
ports = emb_surv_pct.index.tolist()
bottom = np.zeros(len(ports))
for col, clr in zip([0, 1], [ACCENT2, POSITIVE]):
    vals = emb_surv_pct[col].values
    ax8.bar(ports, vals, bottom=bottom, color=clr, width=0.4)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax8.text(j, b + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    bottom += vals
ax8.set_title('Survival by Embarkation Port\n(C=Cherbourg, Q=Queenstown, S=Southampton)', fontsize=11)
ax8.legend(['Not survived', 'Survived'], fontsize=9)
ax8.grid(axis='y')

# ── 9. Family size ────────────────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
df['family_size'] = df['sibsp'].astype(int) + df['parch'].astype(int) + 1
fs_surv = df.groupby('family_size')['survived'].mean() * 100
ax9.bar(fs_surv.index, fs_surv.values, color=ACCENT, alpha=0.85, zorder=3)
ax9.axhline(50, color=WARNING, lw=1.2, ls='--', label='50% line')
ax9.set_title('Survival Rate by Family Size', fontsize=13)
ax9.set_xlabel('Family Size'); ax9.set_ylabel('Survival %')
ax9.grid(axis='y', zorder=0); ax9.legend(fontsize=9)

plt.savefig('/mnt/user-data/outputs/titanic_eda_page1.png', dpi=160, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("Page 1 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Correlations, Outliers & Multivariate
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 14))
fig2.patch.set_facecolor('#0f1117')
gs2  = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.55, wspace=0.42)

fig2.text(0.5, 0.97, '🚢  Titanic EDA — Correlations, Outliers & Multivariate',
          ha='center', va='top', fontsize=20, fontweight='bold', color='white')

# ── 10. Correlation heatmap ───────────────────────────────────────────────────
ax10 = fig2.add_subplot(gs2[0, :2])
num_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(10, 240, as_cmap=True)
sns.heatmap(corr, ax=ax10, annot=True, fmt='.2f', cmap=cmap, center=0,
            linewidths=0.5, linecolor='#0f1117', annot_kws={'size': 10, 'color': 'white'},
            cbar_kws={'shrink': 0.7})
ax10.set_title('Correlation Matrix', fontsize=14, pad=10)
ax10.tick_params(labelsize=10)

# ── 11. Survival correlation bar ─────────────────────────────────────────────
ax11 = fig2.add_subplot(gs2[0, 2])
surv_corr = corr['survived'].drop('survived').sort_values()
colors_c  = [POSITIVE if v > 0 else ACCENT2 for v in surv_corr.values]
ax11.barh(surv_corr.index, surv_corr.values, color=colors_c, height=0.6, zorder=3)
ax11.axvline(0, color='white', lw=0.8)
ax11.set_title('Correlation with\nSurvival', fontsize=13)
ax11.grid(axis='x', zorder=0)
for i, (v, name) in enumerate(zip(surv_corr.values, surv_corr.index)):
    ax11.text(v + (0.01 if v >= 0 else -0.01), i,
              f'{v:.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=9, color='white')

# ── 12. Age boxplot by class ──────────────────────────────────────────────────
ax12 = fig2.add_subplot(gs2[1, 0])
groups = [df[df['pclass'] == c]['age'].dropna() for c in [1, 2, 3]]
bp = ax12.boxplot(groups, patch_artist=True, notch=False,
                  medianprops=dict(color=WARNING, lw=2.5),
                  whiskerprops=dict(color='#a0a0b0'),
                  capprops=dict(color='#a0a0b0'),
                  flierprops=dict(marker='o', markerfacecolor=ACCENT2, markersize=4, alpha=0.6))
for patch, clr in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(clr); patch.set_alpha(0.75)
ax12.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
ax12.set_title('Age Distribution by Class', fontsize=13)
ax12.set_ylabel('Age'); ax12.grid(axis='y')

# ── 13. Fare boxplot by class ─────────────────────────────────────────────────
ax13 = fig2.add_subplot(gs2[1, 1])
groups_f = [df[df['pclass'] == c]['fare'].dropna() for c in [1, 2, 3]]
bp2 = ax13.boxplot(groups_f, patch_artist=True,
                   medianprops=dict(color=WARNING, lw=2.5),
                   whiskerprops=dict(color='#a0a0b0'),
                   capprops=dict(color='#a0a0b0'),
                   flierprops=dict(marker='o', markerfacecolor=ACCENT2, markersize=4, alpha=0.6))
for patch, clr in zip(bp2['boxes'], PALETTE):
    patch.set_facecolor(clr); patch.set_alpha(0.75)
ax13.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
ax13.set_title('Fare Distribution by Class\n(outliers visible in 1st)', fontsize=12)
ax13.set_ylabel('Fare (£)'); ax13.grid(axis='y')

# ── 14. Age vs Fare scatter ───────────────────────────────────────────────────
ax14 = fig2.add_subplot(gs2[1, 2])
for s, clr, lbl in zip([0, 1], [ACCENT2, POSITIVE], ['Not survived', 'Survived']):
    sub = df[df['survived'] == s]
    ax14.scatter(sub['age'], sub['fare'], c=clr, alpha=0.4, s=18, label=lbl, zorder=3)
ax14.set_title('Age vs Fare', fontsize=13)
ax14.set_xlabel('Age'); ax14.set_ylabel('Fare (£)')
ax14.legend(fontsize=9); ax14.grid(zorder=0)

# ── 15. Pclass × Sex survival heatmap ────────────────────────────────────────
ax15 = fig2.add_subplot(gs2[2, 0])
pivot = df.pivot_table('survived', index='sex', columns='pclass', aggfunc='mean') * 100
sns.heatmap(pivot, ax=ax15, annot=True, fmt='.0f', cmap='RdYlGn',
            linewidths=0.5, linecolor='#0f1117', annot_kws={'size': 13, 'fontweight': 'bold'},
            vmin=0, vmax=100, cbar_kws={'label': 'Survival %', 'shrink': 0.8})
ax15.set_title('Survival % — Sex × Class', fontsize=13)
ax15.set_xlabel('Passenger Class'); ax15.set_ylabel('')

# ── 16. Age group survival ────────────────────────────────────────────────────
ax16 = fig2.add_subplot(gs2[2, 1])
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child\n(0-12)', 'Teen\n(13-18)', 'Adult\n(19-35)',
                                 'Middle\n(36-60)', 'Senior\n(60+)'])
ag_surv = df.groupby('age_group', observed=False)['survived'].mean() * 100
ax16.bar(ag_surv.index, ag_surv.values, color=PALETTE, alpha=0.85, zorder=3)
ax16.axhline(38.4, color=WARNING, lw=1.5, ls='--', label='Overall avg 38.4%')
ax16.set_title('Survival Rate by Age Group', fontsize=13)
ax16.set_ylabel('Survival %'); ax16.set_ylim(0, 75)
ax16.grid(axis='y', zorder=0); ax16.legend(fontsize=9)
for i, v in enumerate(ag_surv.values):
    ax16.text(i, v + 1.5, f'{v:.0f}%', ha='center', fontsize=10, color='white', fontweight='bold')

# ── 17. Outlier summary — IQR ─────────────────────────────────────────────────
ax17 = fig2.add_subplot(gs2[2, 2])
outlier_cols = ['age', 'fare', 'sibsp', 'parch', 'family_size']
outlier_pcts = []
for col in outlier_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
    outlier_pcts.append(n_out / df[col].notna().sum() * 100)

colors_o = [WARNING if v > 5 else ACCENT for v in outlier_pcts]
ax17.barh(outlier_cols, outlier_pcts, color=colors_o, height=0.55, zorder=3)
ax17.axvline(5, color=ACCENT2, lw=1.2, ls='--', label='5% threshold')
ax17.set_title('Outlier % by Feature\n(IQR method)', fontsize=13)
ax17.set_xlabel('% Outliers'); ax17.grid(axis='x', zorder=0)
for i, v in enumerate(outlier_pcts):
    ax17.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=9, color='white')
ax17.legend(fontsize=9)

plt.savefig('/mnt/user-data/outputs/titanic_eda_page2.png', dpi=160, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("Page 2 saved.")
print("Done!")
