import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Veri setini okuma
file_path = "data/data.xlsx"
df = pd.read_excel(file_path)

# 2. Takım ID ile isim eşleşmesi için bir sözlük oluşturma
team_mapping = dict(zip(df["Home Team ID"], df["Home Team Name"]))

# 3. Sonuç sütunlarını çıkartma (Fulltime Result, Halftime Result, Secondhalf Result, Home Team Name, Away Team Name)
df_cleaned = df.drop(columns=["Fulltime Result", "Halftime Result", "Secondhalf Result", "Home Team Name", "Away Team Name"])

# 4. Hedef değişkeni tanımlama
target = df["Fulltime Result"]

# 5. Hedef değişkenin sınıf dağılımını kontrol etme
print("\nHedef değişkenin sınıf dağılımı:")
print(target.value_counts())

# 6. Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(df_cleaned, target, test_size=0.2, random_state=42, stratify=target)

# 7. SMOTE ile oversampling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 8. Dengelenmiş sınıf dağılımını kontrol etme
print("\nDengelenmiş eğitim seti sınıf dağılımı:")
print(y_train_balanced.value_counts())

# 9. XGBoost sınıflandırıcı oluşturma (Hiperparametrelerle)
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    learning_rate=0.05,  # Öğrenme oranı
    n_estimators=200,    # Ağaç sayısı
    max_depth=5          # Maksimum derinlik
)

# 10. Modeli eğitim verileriyle eğitme
xgb_model.fit(X_train_balanced, y_train_balanced)

# 11. Test setiyle tahmin yapma
y_pred = xgb_model.predict(X_test)

# 12. Performans değerlendirme
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 13. Yeni maç verilerini tahmin etme
# 'next.xlsx' dosyasını okuma
next_file_path = "data/next.xlsx"
next_matches = pd.read_excel(next_file_path)

# Tahmin için modelin kullandığı özelliklerle uyumlu olması için düzenleme
next_matches_cleaned = next_matches[df_cleaned.columns]  # Sadece eğitimde kullanılan sütunları alıyoruz

# Tahmin yapma
predictions = xgb_model.predict(next_matches_cleaned)

# Tahminleri anlamlı hale getirme
results_map = {0: "Beraberlik", 1: "Ev sahibi galibiyeti", 2: "Deplasman galibiyeti"}
next_matches["Tahmin"] = [results_map[pred] for pred in predictions]

# Takım isimlerini ID'lerden dönüştürme
next_matches["Home Team Name"] = next_matches["Home Team ID"].map(team_mapping)
next_matches["Away Team Name"] = next_matches["Away Team ID"].map(team_mapping)

# Kullanıcıya takım isimleriyle sonuçları gösterme
print("\nYeni maçların tahmin sonuçları:")
print(next_matches[["Home Team Name", "Away Team Name", "Tahmin"]])

# Sonuçları bir dosyaya kaydetme (isteğe bağlı)
next_matches.to_excel("data/next_predictions_with_names.xlsx", index=False)
print("\nTahminler 'data/next_predictions_with_names.xlsx' dosyasına kaydedildi.")
