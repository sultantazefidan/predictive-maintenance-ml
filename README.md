# 🔧 Sensör Verileri ile Makine Öğrenmesi Tabanlı Kestirimci Bakım

Bu projede, **AI4I 2020 Predictive Maintenance** veri seti kullanılarak sensör verilerine dayalı **ikili arıza sınıflandırma problemi** ele alınmıştır.  
Amaç, sistemin **normal** ya da **arıza** durumunda olup olmadığını makine öğrenmesi yöntemleriyle **doğru, güvenilir ve pratik** biçimde tahmin edebilen bir **karar destek modeli** geliştirmektir.

Çalışma, farklı modelleme stratejilerinin karşılaştırıldığı **deneysel ve uygulamaya dönük** bir yaklaşımla tasarlanmıştır.

---

##  Problem Tanımı

Endüstri 4.0 ortamlarında makinelerde meydana gelen beklenmeyen arızalar;  
plansız duruşlara, üretim kayıplarına ve artan bakım maliyetlerine yol açmaktadır.

Geleneksel bakım yaklaşımları:

- **Reaktif bakım:** Arıza oluştuktan sonra müdahale  
- **Periyodik bakım:** Sabit zaman aralıklarında bakım  

Bu yaklaşımlar çoğu zaman **verimsiz ve maliyetlidir**.  
Bu proje, sensör verilerinden faydalanarak **arıza durumunu önceden tahmin etmeyi** hedefleyen **kestirimci bakım yaklaşımını** ele almaktadır.

---

##  Veri Seti

- **Veri seti:** AI4I 2020 Predictive Maintenance  
- **Toplam örnek:** ~10.000  
- **Özellik sayısı:** 13 sayısal sensör değişkeni  

**Hedef değişken:**
- `0` → Normal  
- `1` → Arıza  

Veri seti üzerinde uygulanan işlemler:

- Eksik değer kontrolü  
- Gereksiz kimlik sütunlarının kaldırılması  
- One-Hot Encoding (Type değişkeni)  
- Standardizasyon (StandardScaler)  

---

##  Kullanılan Yöntemler

### Makine Öğrenmesi Modelleri
- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- XGBoost  

### Deneysel Senaryolar
- 5-Katlı Çapraz Doğrulama (**Stratified K-Fold**)  
- %80 – %20 Eğitim / Test Bölmesi  
- Feature Selection (Random Forest Feature Importance)  
- Keşifsel Veri Analizi (EDA)  
- Denetimsiz Kümeleme (K-Means + PCA)  

---

##  Performans Değerlendirme Metrikleri

Modeller aşağıdaki metrikler ile değerlendirilmiştir:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- AUC-ROC  
- Specificity  
- Eğitim Süresi  

Ayrıca aşağıdaki görsellerle analiz yapılmıştır:

- Confusion Matrix  
- ROC eğrileri  
- Precision–Recall eğrileri  

---

##  Keşifsel Veri Analizi (EDA) ve Kümeleme

Bu çalışmada modelleme aşamasına geçilmeden önce veri setinin genel yapısını anlamak amacıyla **keşifsel veri analizi (EDA)** gerçekleştirilmiştir. Bu kapsamda aşağıdaki görseller üretilmiştir:

- Özellikler arası korelasyon ısı haritası (*correlation_heatmap*)
- Sensör değişkenlerine ait histogramlar (*hist_air_temperature, hist_tool_wear* vb.)

EDA süreci, veri setinin dağılım karakteristiklerini incelemek, olası ilişkileri gözlemlemek ve modelleme öncesinde veri hakkında genel bir farkındalık kazanmak amacıyla uygulanmıştır.

Ayrıca veri seti **etiketli ve denetimli bir sınıflandırma problemi** olmasına rağmen, veri dağılımının doğal yapısını incelemek amacıyla **K-Means kümeleme analizi** gerçekleştirilmiştir. Bu analiz, performans karşılaştırması amacıyla değil; yalnızca veri noktalarının özellik uzayında nasıl dağıldığını gözlemlemek için yapılmıştır.

Kümeleme sonuçları **PCA tabanlı görselleştirme** ile sunulmuş olup, veri setinde belirgin ve ayrık doğal kümeler bulunmadığını göstermektedir. Bu durum, problemin doğası gereği **denetimli öğrenme yöntemlerinin gerekli olduğunu** desteklemektedir.

---
##  Öne Çıkan Bulgular (Pratik Değerlendirme)

- **5-Fold çapraz doğrulama**, model genellenebilirliğini değerlendirmek için en güvenilir yaklaşım olmuştur.  
- **%80–20 eğitim–test bölmesi**, 5-Fold ile benzer performans sunarken **daha kısa eğitim süresi** sağlamış; bu yönüyle **pratik ve zaman kısıtlı senaryolar için uygun** bir alternatif olarak değerlendirilmiştir.  
- **Logistic Regression**, %80–20 senaryosunda **en kısa eğitim süresi** ile birlikte yüksek accuracy, precision, recall ve AUC-ROC değerleri sunarak **gerçek hayatta uygulanabilirlik açısından en dengeli ve verimli model** olarak öne çıkmıştır.  
- **XGBoost**, tüm senaryolarda **en yüksek ve en kararlı performansı** sergilemiş; ancak daha yüksek hesaplama maliyeti nedeniyle **performans odaklı uygulamalar** için daha uygun bir seçenek olmuştur.  
- Feature selection sonrası:
  - Random Forest ve XGBoost modelleri yüksek performanslarını korumuş,  
  - Decision Tree modelinde ise precision ve F1-score metriklerinde belirgin performans düşüşü gözlemlenmiştir.  
- **K-Means kümeleme analizi**, veri setinde belirgin bir doğal küme yapısı olmadığını göstermiş; bu durum problemin doğası gereği **denetimli öğrenme yaklaşımlarının zorunlu olduğunu** ortaya koymuştur.  

---

##  Gerçek Hayat (Deployment) Senaryosu

Bu projede geliştirilen model, doğrudan Python kodu olarak değil;  
**eğitilmiş model ve ön işleme adımlarının entegre edilmesiyle** canlı sistemlerde kullanılabilecek şekilde tasarlanmıştır.

### Öngörülen kullanım senaryoları:

- **Edge / gömülü sistem:**  
  Sensör verilerinin yerel olarak analiz edilmesiyle düşük gecikmeli arıza tahmini  

- **REST API tabanlı sistem:**  
  Sensör verilerinin API üzerinden gönderilmesi ve sonuçların dashboard üzerinden izlenmesi  

Model, fiziksel sistemleri doğrudan kontrol etmek yerine **erken uyarı ve karar destek mekanizması** olarak görev yapmaktadır.

---

##  Proje Yapısı

├── code/
│   ├── 01_data_overview.py
│   ├── 02_data_cleaning_1.py
│   ├── 03_data_cleaning_2.py
│   ├── 04_data_validation.py
│   ├── 05_eda_analysis.py
│   ├── 06_kmeans_clustering.py
│   ├── 07_feature_selection.py
│   ├── 08_modeling_5fold.py
│   └── 09_modeling_80_20.py
│
├── results/
│   ├── eda/
│   ├── clustering/
│   ├── feature_selection/
│   ├── modeling_5fold/
│   └── modeling_80_20/
│
├── data/
│   └── ai4i2020_clean_step2.csv
│
└── README.md

##  Kullanılan Teknolojiler

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  

##  Amaç

Bu proje, staj ve iş başvurularında portföy amaçlı hazırlanmış olup:

- Makine öğrenmesi  
- Veri ön işleme  
- Model karşılaştırma  
- Performans analizi  
- Gerçek hayata uyarlanabilirlik  

konularındaki yetkinliği göstermeyi hedeflemektedir.
