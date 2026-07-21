import matplotlib.pyplot as plt
import numpy as np

# Modeller (Düşük FPS'ten yükseğe sıralı)
models = ['DIFRINT', 'DUT Model', 'StabNet', 'NNDVS', 'L1 Optimal', 'Proposed Method']

# Aydınlık ve Karanlık Ortam FPS Değerleri
fps_well_lit = [1.45, 1.54, 2.20, 8.82, 20.75, 121.79]
fps_low_light = [2.64, 5.93, 11.34, 35.24, 24.08, 121.79]

x = np.arange(len(models))  # Etiket konumları
width = 0.35  # Çubuk genişliği

# Akademik stil ayarları
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 11

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

# Çubukların çizimi
rects1 = ax.bar(x - width/2, fps_well_lit, width, label='Well-Lit (Aydınlık)', color='#4a7c59', edgecolor='#2c3e50', linewidth=0.7)
rects2 = ax.bar(x + width/2, fps_low_light, width, label='Low-Light (Karanlık)', color='#68809a', edgecolor='#2c3e50', linewidth=0.7)

# Geniş aralıktaki verileri dengeli göstermek için logaritmik Y ekseni
ax.set_yscale('log')

# Eksen ve Başlıklar
ax.set_ylabel('Throughput (FPS) [Log Scale]', fontweight='bold', labelpad=10)
ax.set_xlabel('Stabilization Models', fontweight='bold', labelpad=10)
ax.set_title('Throughput Comparison Under Varying Illumination Conditions', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend(frameon=True, facecolor='white', edgecolor='none')

# Çubuk üstü etiketler
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Grid ve Çerçeve
ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Limit esnetme
ax.set_ylim(top=max(fps_low_light) * 5)

plt.tight_layout()

# Kaydetme
plt.savefig('illumination_comparison_fps_all.pdf', format='pdf', bbox_inches='tight')
plt.savefig('illumination_comparison_fps_all.png', format='png', bbox_inches='tight')

print("Tüm modelleri (L1 dahil) içeren aydınlık vs. karanlık grafiği oluşturuldu.")