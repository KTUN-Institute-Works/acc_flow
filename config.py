"""
ACCFlow — çoklu video eğitim/test hattı için merkezi ayarlar.
Central configuration for the multi-video training / testing pipeline.

Tüm yollar proje köküne göre hesaplanır; script nereden çalıştırılırsa çalıştırılsın doğru çalışır.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Veri seti / Dataset
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "IDA_split")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

# Her klip için hesaplanan IMU özellikleri + RAFT tabanlı ground truth önbelleği
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

# ---------------------------------------------------------------------------
# RAFT (ground-truth üretimi)
# ---------------------------------------------------------------------------
RAFT_DIR = os.path.join(PROJECT_ROOT, "RAFT")
RAFT_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "raft-things.pth")
FLOW_WIDTH = 320          # RAFT bu genişlikte çalışır (stage3 ile aynı). Model çıktısı dx,dy bu ölçektedir.
RAFT_ITERS = 10           # stage3 ile aynı
TRAJ_SIGMA = 15           # stage4: Gaussian yörünge yumuşatma sigması
# Hedef hareket: "similarity" = RAFT akışına RANSAC ile oturtulan öteleme + dönme -> model çıktısı dx, dy, dθ, α
#                "median"     = eski davranış, yalnızca medyan öteleme -> dx, dy, α (dönmeyi göremez)
TARGET = "similarity"
# Modelin tahmin ettiği (ve düzeltilen) hareket bileşenleri — "gözlenebilen düzeltme":
# yalnızca ivmeölçer (jiroskop yok) yerçekimi yönünden roll (-> dθ) ve pitch (-> dy) açısını görür. Yatay kayma dx
# çoğunlukla yaw dönmesinden gelir ve yerçekimini değiştirmez -> IMU'dan gözlenemez (RAFT ile korelasyon ~0).
# dx tahmin edilmez ve düzeltilmez (0). Tümü için: ("dx", "dy", "dtheta").
MOTION_COMPONENTS = ("dy", "dtheta")
# Benzerlik oturtulurken RAFT akışının örneklendiği noktalar: "corners" (Shi-Tomasi köşeleri; su/gökyüzünü dışlar)
# veya "gradient" (gradyanı yüksek ızgara noktaları)
RAFT_FIT_POINTS = "corners"
MAX_ROT_DEG = 5.0         # güvenlik: tahmin edilen dönme düzeltmesi bu açıyla sınırlanır
ALPHA_EPS = 1e-6

# Alpha hedefi (tez): "pan"   = kasıtlı hareket algılama. Kamera sabitken alpha -> 1, kasıtlı pan/dönüşte -> 0:
#                              alpha = 1 / (1 + (v / ALPHA_PAN_V0)^ALPHA_PAN_POWER), v = yumuşak yolun hızı
#                     "ratio" = ilk tanım: |jitter| / (|jitter| + |yumuşak hareket|)
#                     "optimal" = öğrenilmiş düzeltme güveni (varsayılan): her kare için kalan sarsıntıyı en aza
#                              indiren düzeltme oranı. Modelin (çapraz doğrulamayla, görmediği kliplerde yapılan)
#                              hareket tahmini p ve RAFT hedefi g ile, zamanda Gaussian pencereli en küçük kareler:
#                              alpha* = Σw g·p / Σw p·p  (0..1'e kırpılır; tahmin küçükken 1'e çekilir).
#                              Tahmin güvenilir -> alpha* ~ 1, güvenilmez/yanlış yönlü -> alpha* ~ 0.
ALPHA_MODE = "optimal"
ALPHA_OPT_FOLDS = 5       # alpha* için çapraz doğrulama katı (her klip, onu görmeyen modelle tahmin edilir)
ALPHA_OPT_EPOCHS = 30     # kat modellerinin epoch sayısı (yalnızca hareket; birkaç saniye)
ALPHA_OPT_SHRINK = 0.05   # tahmin çok küçükken alpha*'ı 1'e çeken düzenleme (ortalama Σp·p'nin oranı)
# v0: bu hızda alpha = 0.5. Veriye göre ayarlandı: IDA eğitim verisinde yumuşak yol hızının yüzdelikleri
# 50/75/90/99 = 0.14/0.34/0.52/0.71 px/kare (320 px ölçek). v0=1.0 iken karelerin yalnızca %0.2'si "pan"
# sayılıyordu (alpha neredeyse sabit 0.93). v0 ≈ 75. yüzdelik -> hızın en yüksek ~%25'lik dilimi kasıtlı hareket.
ALPHA_PAN_V0 = 0.35
ALPHA_PAN_POWER = 2.0     # geçişin keskinliği
ALPHA_SMOOTH_SIGMA = 10   # kare; alpha'nın zamanda titrememesi için
ALPHA_MIN = 0.0

# ---------------------------------------------------------------------------
# IMU ön işleme (stage1 + stage2)
# ---------------------------------------------------------------------------
# IMU sarsıntı bandı: "detrend" = IMU - Gaussian(IMU, TRAJ_SIGMA)  (görsel hedefle aynı bant, önerilen)
#                     "highpass" = eski stage2 (Butterworth > IMU_CUTOFF_HZ). DİKKAT: IDA'da görsel sarsıntı
#                     ~0.6-0.8 Hz'de; 2 Hz high-pass bu bilgiyi siler ve model öğrenemez.
#   "detrend_tilt" = yukarıdaki 3 kanal + yerçekiminden eğim açıları roll=atan2(ax,ay), pitch=atan2(az,ay)
#                    (aynı bantta) -> 5 kanal. Roll, görüntüdeki dönmeyle r≈0.89 korelasyonlu (varsayılan).
#   "tilt_motion"  = "detrend_tilt" + kasıtlı hareket bandı: yavaş roll/pitch değişim hızı ve yavaş doğrusal
#                    ivme -> 8 kanal (alpha'nın pan'i algılayabilmesi için; varsayılan)
IMU_FEATURE = "tilt_motion"
# Telefon yönü: "auto" = yerçekimi hangi eksendeyse ona göre IMU eksenleri dikey (portrait: yerçekimi +y) düzene
# çevrilir. IDA dikey çekildiği için değişmez; yatay (landscape) videolarda x' = -y, y' = x (ya da tersi).
IMU_ORIENTATION = "auto"
# IMU-video zaman kayması (sn). RAFT hedefiyle çapraz korelasyon: IMU, videodan neredeyse her klipte 6-9 kare
# (~0.25 sn) geride (IDA, araba, vapur, yürüyüş). Cihaza bağlı sabit bir gecikme olduğundan testte de (RAFT'sız)
# uygulanabilir: IMU zaman damgaları bu kadar geri çekilir. 0 = kapalı.
IMU_TIME_OFFSET_S = 0.25
IMU_CUTOFF_HZ = 2.0       # yalnızca IMU_FEATURE = "highpass" için
IMU_FILTER_ORDER = 4

# ---------------------------------------------------------------------------
# Eğitim / Training
# ---------------------------------------------------------------------------
WINDOW_SIZE = 120         # 120 kare ≈ 4 sn (alıcı alan 61 kare olduğu için pencere daha uzun tutuldu)
STRIDE = 10
# Model alıcı alanı: (1,1,1,1) = orijinal (17 kare ≈ 0.6 sn), (1,2,4,8) = 61 kare ≈ 2 sn.
# IDA'daki sarsıntı ~0.6-0.8 Hz (periyot ~1.3 sn) olduğundan geniş alıcı alan gerekli.
MODEL_DILATIONS = (1, 2, 4, 8)
MODEL_DROPOUT = 0.3       # aşırı öğrenmeye karşı (eski modelde yoktu; 0.0 = kapalı)
VAL_FRACTION = 0.15       # train kliplerinin bu kadarı doğrulama (klip bazında ayrılır)
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
EARLY_STOP_PATIENCE = 30
MIN_EPOCHS = 20           # erken durdurma bu epoch'tan önce devreye girmez (α öğrenmesi ~15. epoch'tan sonra başlıyor)
# Doğrulama klipleri alan bazında (klip bazında ayrılır). Yalnızca IDA'dan seçmek, modeli sarsıntılı alanlara
# bakmadan seçtiriyordu. Ek alanlarda her klip değerli olduğundan REFIT_ALL ile en son hepsiyle yeniden eğitilir.
VAL_CLIPS_PER_DOMAIN = {"ida": 3, "araba": 1, "vapur": 1}
# Model seçim ölçütü: her alanda naif referansa göre GÖRELİ hata (hareket: 0 tahmini, α: eğitim ortalaması),
# alanların ortalaması -> LAMBDA_MOTION * göreli hareket + LAMBDA_ALPHA * göreli α. Böylece büyük genlikli
# alanlar (araba) ölçütü domine etmez ve α'daki iyileşme de seçimde görünür.
# REFIT_ALL: seçilen epoch sayısıyla (aynı öğrenme oranı takvimiyle) tüm eğitim klipleri (doğrulama dahil) ile
# sıfırdan yeniden eğitip son modeli onunla kaydeder (eğitim birkaç saniye sürdüğü için maliyetsiz).
REFIT_ALL = True
LAMBDA_MOTION = 1.0
LAMBDA_ALPHA = 0.5
SEED = 42

# Çok alanlı eğitim (IDA + araba + vapur + yürüyüş). Eski yalnızca-IDA modeli/sonuçları
# (models/imu_adaptive_multivideo.pth, results/ida_split/) karşılaştırma için olduğu gibi bırakıldı.
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "imu_adaptive_multidomain.pth")
TRAIN_RUN_DIR = os.path.join(PROJECT_ROOT, "results", "multidomain", "training")

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
# Boş bırakılırsa otomatik seçim yapılır; elle vermek için clip_id listesi yazın.
TEST_CLIPS = [                      # IDA_split/test içindeki 8 klibin tamamı (hiçbiri eğitimde yok)
    "video_1778671955586",          # tamamen görülmemiş kayıt (hiçbir parçası eğitimde yok)
    "video_1778672350011_part04",
    "video_1778672350011_part08",
    "video_1778672696776_part04",
    "video_1778672696776_part06",
    "video_1778673028077_part02",
    "video_1778673028077_part03",
    "video_1783343528738_part02",   # sallantının belirgin olduğu anlar var
    "araba_1778511656348",          # araba (eğitimde yok)
    "vapur_1789302036248",          # vapur (eğitimde yok)
]
NUM_TEST_CLIPS = 10

# Karşılaştırılan durumlar: sabit alpha (0, 0.5, 1), kontrol deneyi ve adaptif alpha (tez)
#   "mean" = sabit alpha, değeri adaptif alpha'nın o klipteki ortalaması. Adaptif ile aynı ortalama
#            düzeltme miktarını uygular ama zamanda değişmez -> adaptif kazancın alpha'nın zamanla
#            değişmesinden mi yoksa yalnızca ortalamasının 1'den küçük olmasından mı geldiğini ayırır.
MODES = [
    ("alpha_0.0", 0.0),
    ("alpha_0.5", 0.5),
    ("alpha_1.0", 1.0),
    ("alpha_mean", "mean"),
    ("adaptive", None),
]

# RAFT referansı (modelin hedefi olan RAFT jitter'ı ile stabilizasyon = ulaşılabilecek üst sınır)
#   raft_gt       : RAFT dx,dy, alpha = 1
#   raft_gt_alpha : RAFT dx,dy, RAFT'tan hesaplanan alpha_gt (modelin alpha hedefi)
#   raft_gt_obs   : RAFT jitter'ının yalnızca modelin düzelttiği bileşenleri (MOTION_COMPONENTS), alpha = 1
#                   -> IMU tahmini kusursuz olsaydı ulaşılabilecek üst sınır (dx düzeltilmeden)
#   imu_alpha_opt : IMU tahmini + RAFT'tan hesaplanan ideal alpha* (bu modelin tahminine göre) -> alpha başının
#                   ulaşabileceği üst sınır (hareket tahmini aynı, yalnızca alpha kusursuz)
REFERENCE_MODES = [
    ("raft_gt", 1.0),
    ("raft_gt_obs", "obs"),
    ("imu_alpha_opt", "opt"),
]
RAFT_REFERENCE_DIR = os.path.join(PROJECT_ROOT, "results", "ida_split", "raft_reference")

# Ek alanlar (IDA dışı; Veri Seti/<klasör>/video_<id>.mp4 + sensors_<id>.csv). Klip adı: <önek>_<id>.
# IDA ve IDA_split_video_level_backup klasörleri KULLANILMAZ: IDA_split test kliplerinin aynısını içerirler (sızıntı).
VERI_SETI_DIR = os.path.expanduser("~/Veri Seti")
EXTRA_DOMAINS = {"araba": "Araba", "vapur": "Vapur", "walk": "Yürüyüş"}
# Her alandan bir video teste ayrılır (eğitimde hiç görülmez); kalanlar eğitime girer. Yürüyüş tek video -> eğitim.
EXTRA_TEST_CLIPS = ["araba_1778511656348", "vapur_1789302036248"]
# Alan dengesi: IDA ~15 dk, diğer alanlar toplam ~2 dk. Eğitimde her alanın pencere payı n^(1-DOMAIN_BALANCE) ile
# orantılı örneklenir: 0 = doğal oran (IDA ~%93), 1 = tüm alanlar eşit (az veride aşırı tekrar), 0.5 = arası.
DOMAIN_BALANCE = 0.5

MAX_SHIFT_RATIO = 0.25    # Aşırı tahminlere karşı güvenlik: kaydırma, kare boyutunun %25'iyle sınırlanır
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "multidomain", "test")

# DUTCode / NNDVS MetricAnalyzer ayarı: homografi için akış haritası kaç pikselde bir örneklenir
METRIC_GRID_STEP = 40     # NNDVS/DUTCode varsayılanı (scale_factor=40)
