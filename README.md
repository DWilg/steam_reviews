# Analiza recenzji gier Steam z wykorzystaniem PySpark

## 1. Cel projektu
Celem projektu jest demonstracja zintegrowanego podejścia do analizy danych tekstowych i liczbowych dotyczących recenzji gier na platformie Steam. Zakres obejmuje: wstępne przygotowanie danych, przetwarzanie w paradygmacie MapReduce (RDD), agregacje na poziomie API DataFrame (Spark SQL), eksploracyjną analizę danych (EDA) oraz podstawową wizualizację wyników. Projekt może służyć jako baza do dalszych rozszerzeń (np. analiza sentymentu z użyciem NLP).

## 2. Format danych wejściowych
Oczekiwany plik CSV: `dataset.csv` umieszczony w katalogu głównym projektu, zawierający nagłówek:
```
Game ID,app_name,Game Name,review_text,review_score,review_votes
```
Opis wybranych pól:
- `review_score` – liczbowy wynik oceny (przyjęto domyślnie, że wartości >= 4 oznaczają rekomendację; próg można zmienić w kodzie).
- `review_text` – pełny tekst recenzji użytkownika.

Alternatywnie ścieżkę do innego pliku można wskazać parametrem `--path` podczas uruchamiania skryptu.

## 3. Zakres funkcjonalny
W projekcie zaimplementowano następujące elementy:
1. Czyszczenie danych: usuwanie rekordów z pustym lub nullowym polem `review_text`, deduplikacja.
2. Operacje MapReduce (RDD): zliczanie recenzji, obliczanie średnich ocen, agregacja prostego sentymentu.
3. Agregacje DataFrame: analogiczne metryki w celu porównania ergonomii i wydajności.
4. EDA: identyfikacja najczęściej recenzowanych gier, rozkład ocen, średnia ocena według `app_name`.
5. Wizualizacje: wykresy słupkowe oraz histogramy eksportowane do plików graficznych (PNG).
6. Rejestrowanie czasu wykonania kluczowych etapów (podstawowa profilacja).

## 4. Struktura katalogów
```
steam_reviews/
  requirements.txt
  README.md
  src/
    steam_reviews_analysis.py
  output/  # katalog wynikowy (generowany automatycznie)
```

## 5. Wymagania wstępne
- Python w wersji co najmniej 3.9., ale nie wyżej niż 3.11, 3.12 jest nie wspierane przez PySpark.
- Zainstalowane środowisko Java (JDK 8+), wymagane do działania Spark.

## 6. Instalacja środowiska
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
W przypadku problemów z PySpark (brak JVM) należy zainstalować JDK oraz ustawić zmienną środowiskową `JAVA_HOME`.

## 7. Uruchamianie analizy
Standardowe wywołanie (ze wskazaniem ścieżki):
```pwsh
python .\src\steam_reviews_analysis.py --path .\dataset.csv
```
Automatyczne wyszukiwanie pliku (bez jawnego parametru):
```pwsh
python .\src\steam_reviews_analysis.py
```

## 8. Generowane wyniki (katalog `output/`)
- RDD (CSV): `rdd_review_counts.csv`, `rdd_avg_scores.csv`, `rdd_sentiment_counts.csv`
- DataFrame (foldery CSV): `df_review_counts/`, `df_avg_scores/`, `df_sentiment_counts/`
- EDA (CSV): `top_games.csv`, `score_distribution.csv`, `app_avg_scores.csv`
- Wizualizacje (PNG): `top_games_review_count.png`, `review_score_distribution.png`, `average_review_score_top.png`

## 9. Parametry konfiguracyjne
- Próg pozytywnej oceny: stała `POSITIVE_THRESHOLD` w pliku `steam_reviews_analysis.py`.
- Liczba gier w zestawieniu TOP: stała `TOP_N`.

## 10. Rozwiązywanie typowych problemów
| Problem | Możliwa przyczyna | Zalecane działanie |
|---------|------------------|--------------------|
| Błędna schema | Niezgodny nagłówek CSV | Zweryfikować kolejność i nazwy kolumn |
| Błąd JVM | Brak lub zła instalacja Javy | Sprawdzić `JAVA_HOME`, wersję JDK |
| Wysokie zużycie pamięci | Duży plik wejściowy | Zredukować dane lub skonfigurować parametry Spark |

## 11. Możliwe rozszerzenia
- Analiza sentymentu z wykorzystaniem bibliotek NLP.
- Porównanie wydajności z przetwarzaniem wyłącznie w Pandas.
- Dodanie metryk jakości danych (np. długość tekstu, wykrywanie spamu).

## 12. Licencja
Projekt nie zawiera formalnej licencji. Może zostać dostosowany do wymogów zajęć akademickich lub pracy badawczej.

## 13. Podsumowanie
Projekt ilustruje podstawowy przepływ analityczny: wczytanie → oczyszczanie → agregacja (RDD/DataFrame) → eksploracja → wizualizacja. Stanowi punkt wyjścia do bardziej zaawansowanych analiz i eksperymentów z ekosystemem Apache Spark.

---
