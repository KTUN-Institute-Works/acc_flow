import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from analyzer import MetricAnalyzer


def compute_warp_flowmap(original_video_path, stabilized_video_path):
    """
    Orijinal video ile stabilize edilmiş video arasındaki
    yoğun optik akışı (Farneback) hesaplayarak warp matrisini üretir.
    (Boyut uyuşmazlıklarına karşı otomatik yeniden boyutlandırma koruması eklendi)
    """
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_stab = cv2.VideoCapture(stabilized_video_path)

    flowmap_array = []

    ok1, prev_orig = cap_orig.read()
    ok2, prev_stab = cap_stab.read()

    if not (ok1 and ok2):
        print(f"HATA: Videolar okunamadı! -> {original_video_path}")
        return None

    prev_orig_gray = cv2.cvtColor(prev_orig, cv2.COLOR_BGR2GRAY)
    prev_stab_gray = cv2.cvtColor(prev_stab, cv2.COLOR_BGR2GRAY)

    # GÜVENLİK KONTROLÜ: Boyutlar farklıysa, stabilize videoyu orijinalin boyutuna zorla
    if prev_orig_gray.shape != prev_stab_gray.shape:
        prev_stab_gray = cv2.resize(prev_stab_gray, (prev_orig_gray.shape[1], prev_orig_gray.shape[0]))

    # Optik akış hesaplama döngüsü
    while True:
        ok1, curr_orig = cap_orig.read()
        ok2, curr_stab = cap_stab.read()

        if not (ok1 and ok2):
            break

        curr_orig_gray = cv2.cvtColor(curr_orig, cv2.COLOR_BGR2GRAY)
        curr_stab_gray = cv2.cvtColor(curr_stab, cv2.COLOR_BGR2GRAY)

        # GÜVENLİK KONTROLÜ: Boyutlar farklıysa yeniden boyutlandır
        if curr_orig_gray.shape != curr_stab_gray.shape:
            curr_stab_gray = cv2.resize(curr_stab_gray, (curr_orig_gray.shape[1], curr_orig_gray.shape[0]))

        # Farneback Yoğun Optik Akış
        flow = cv2.calcOpticalFlowFarneback(curr_orig_gray, curr_stab_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flowmap_array.append(flow)

    cap_orig.release()
    cap_stab.release()

    return np.array(flowmap_array, dtype=np.float32)


def plot_academic_metrics(results_dict):
    """
    Hesaplanan metrikleri akademik bir bar grafiği (grouped bar chart) olarak çizer.
    """
    models = list(results_dict.keys())
    crop_ratios = [results_dict[m]['Crop Ratio'] for m in models]
    distortions = [results_dict[m]['Distortion'] for m in models]
    stabilities = [results_dict[m]['Stability Score'] for m in models]

    x = np.arange(len(models))
    width = 0.25  # Bar genişliği

    fig, ax = plt.subplots(figsize=(12, 6))

    # Barları oluşturuyoruz
    rects1 = ax.bar(x - width, crop_ratios, width, label='Crop Ratio (↑ Yüksek İyidir)', color='#4C72B0')
    rects2 = ax.bar(x, distortions, width, label='Distortion (↓ 1.0\'a Yakın İyidir)', color='#DD8452')
    rects3 = ax.bar(x + width, stabilities, width, label='Stability Score (↑ Yüksek İyidir)', color='#55A868')

    # Eksen etiketleri ve başlık
    ax.set_ylabel('Skor Değerleri', fontsize=12)
    ax.set_title('Video Stabilizasyon Modellerinin Karşılaştırmalı Performans Analizi', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Değerleri barların üzerine yazdırma
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 point dikey ofset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    # Grafiği kaydet ve göster
    plt.savefig('model_kiyaslama_grafigi.png', dpi=300, bbox_inches='tight')
    print("\nGrafik 'model_kiyaslama_grafigi.png' olarak kaydedildi.")
    plt.show()


def main():
    # 1. Proje ana dizinini dinamik olarak bulma (evaluation klasöründen bir üst klasöre çıkıyoruz)
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    # Orijinal videonuzun kesin yolu
    original_video = os.path.join(project_root, "data", "videos", "input.mp4")

    # Kıyaslanacak modellerin çıktı videolarının kesin yolları
    # DİKKAT: Buradaki dosya isimlerini (örn: l1_optimal_output.mp4) kendi kaydettiğiniz
    # gerçek dosya isimleriyle değiştirmelisiniz!
    models_to_evaluate = {
        "L1 Optimal": os.path.join(project_root, "results", "night", "l1_optimal_output.mp4"),
        "DUTCode": os.path.join(project_root, "results", "night", "dutcode_output.mp4"),
        "NNDVS": os.path.join(project_root, "results", "night", "nndvs_output.mp4"),
        "DIFRINT": os.path.join(project_root, "results", "night", "difrint_output.mp4"),
        "StabNet": os.path.join(project_root, "results", "night", "stabnet_output.mp4"),
        "Önerilen Model": os.path.join(project_root, "results", "night", "proposed_model_output.mp4")
    }

    # Çözünürlüğü ilk videodan otomatik alıyoruz
    cap = cv2.VideoCapture(original_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    all_results = {}

    # 2. Aşama: Tüm modeller için döngüye girip metrikleri hesaplama
    print("Metrik hesaplamaları başlıyor. Bu işlem video uzunluğuna göre biraz sürebilir...\n")

    for model_name, stab_video_path in models_to_evaluate.items():
        if not os.path.exists(stab_video_path):
            print(f"[{model_name}] Videosu bulunamadı, atlanıyor...")
            continue

        print(f"İşleniyor: {model_name}...")

        # A) Orijinal ve Stabilize video arasındaki warp matrisini (Optik Akış) hesapla
        flowmap_array = compute_warp_flowmap(original_video, stab_video_path)

        # B) Metrik Analizini Başlat
        # scale_factor=1 yapıyoruz çünkü doğrudan orijinal çözünürlükteki videoları karşılaştırıyoruz.
        analyzer = MetricAnalyzer(frame_width=frame_width, frame_height=frame_height, scale_factor=1, start=1)

        # C) Metrikleri hesapla
        crop_ratio, distortion, stab_score = analyzer.run(flowmap_array, stab_video_path)

        all_results[model_name] = {
            "Crop Ratio": crop_ratio,
            "Distortion": distortion,
            "Stability Score": stab_score
        }
        print(f"[{model_name}] Tamamlandı -> CR: {crop_ratio:.3f}, Dist: {distortion:.3f}, Stab: {stab_score:.3f}\n")

    # 3. Aşama: Sonuçları görselleştirme
    if all_results:
        plot_academic_metrics(all_results)


if __name__ == "__main__":
    main()