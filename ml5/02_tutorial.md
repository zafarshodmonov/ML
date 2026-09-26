# Tutorial: Decision Trees va Ensemble metodlar (CART, Random Forest, GBDT)

Bu tutorial ML5 loyihasidagi barcha konsepsiyalarni — nazariya, pseudocode, murakkablik tahlili (complexity analysis) va implementatsiya darajasida — chuqur tushuntirib beradi. Maqsad — loyihani "andishasiz copy-paste" qilmasdan, har bir formula va algoritmning **nega** shunday ishlashini tushunib yechish.

---

## 1. Boshlang'ich tushunchalar: Impurity o'lchovlari

Decision tree qurishning yuragi — "qaysi split eng yaxshi?" degan savolga javob berish. Buning uchun **impurity** (aralashganlik/nopoklik) o'lchovlari kerak.

### 1.1 Gini impurity (classification uchun)

Tugunda $K$ ta klass bor, $p_k$ — $k$-klassga tegishli namunalar ulushi bo'lsin:

$$Gini(t) = 1 - \sum_{k=1}^{K} p_k^2 = \sum_{k \ne k'} p_k p_{k'}$$

**Intuitsiya:** agar tugundan tasodifiy ikkita namuna olsak (qaytarib qo'yish bilan) va ularni tugundagi klass taqsimoti bo'yicha tasodifiy belgilasak, Gini — bu ikkalasi turli klassga tushish ehtimoli. Tugun toza (pure), ya'ni faqat bitta klass bo'lsa, $Gini = 0$. Binary classification'da klasslar teng taqsimlangan bo'lsa ($p=0.5$), $Gini = 0.5$ — maksimal qiymat.

### 1.2 Entropiya (muqobil o'lchov)

$$H(t) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

Gini va entropiya deyarli bir xil xatti-harakatga ega (ikkalasi ham toza tugunlarda 0, aralashgan tugunlarda maksimal), lekin Gini hisoblash tezroq (logarifm yo'q), shuning uchun ko'pchilik implementatsiyalar (jumladan sklearn'ning default'i va bu loyihaning talabi) Gini'ni ishlatadi.

### 1.3 Regression uchun: variance / standard deviation

Regression tree uchun target uzluksiz (continuous), shuning uchun impurity o'rniga tugun ichidagi **dispersiya (variance)** yoki **standart og'ish (standard deviation)** ishlatiladi:

$$Var(t) = \frac{1}{n_t}\sum_{i \in t}(y_i - \bar{y}_t)^2$$

bu yerda $\bar{y}_t$ — tugundagi target'larning o'rtachasi. Bu MSE (mean squared error) bilan bevosita bog'liq: agar tugunning bashorati $\bar{y}_t$ bo'lsa, $Var(t)$ aynan shu tugundagi MSE'dir.

### 1.4 Split sifatini baholash: Impurity reduction

$D$ tugun $t$ ni ikkiga — $t_L$ (chap, $n_L$ ta namuna) va $t_R$ (o'ng, $n_R$ ta namuna) ga bo'lsin, $n = n_L + n_R$. Split sifati:

$$\Delta = Impurity(t) - \left(\frac{n_L}{n} Impurity(t_L) + \frac{n_R}{n} Impurity(t_R)\right)$$

Bu — **weighted impurity reduction**. Algoritm har bir feature va har bir threshold uchun $\Delta$ ni hisoblab, uni maksimallashtiradigan split'ni tanlaydi.

---

## 2. CART: Best split qidirish algoritmi

### 2.1 Pseudocode

```
function FIND_BEST_SPLIT(X, y):
    best_gain = -infinity
    best_feature, best_threshold = None, None
    parent_impurity = IMPURITY(y)

    for feature in range(n_features):
        sorted_values = SORT_UNIQUE(X[:, feature])
        for i in range(len(sorted_values) - 1):
            threshold = (sorted_values[i] + sorted_values[i+1]) / 2
            left_mask  = X[:, feature] <= threshold
            right_mask = ~left_mask
            if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                continue
            gain = WEIGHTED_IMPURITY_REDUCTION(y, y[left_mask], y[right_mask], parent_impurity)
            if gain > best_gain:
                best_gain = gain
                best_feature, best_threshold = feature, threshold

    return best_feature, best_threshold, best_gain
```

```
function BUILD_TREE(X, y, depth):
    if depth == max_depth or IMPURITY(y) < eps or len(y) < min_samples_split:
        return LEAF(value = AGGREGATE(y))   # klassifikatsiyada: klass ehtimolliklari; regressiyada: o'rtacha

    feature, threshold, gain = FIND_BEST_SPLIT(X, y)
    if gain <= 0:
        return LEAF(value = AGGREGATE(y))

    left_idx, right_idx = SPLIT(X, feature, threshold)
    left_child  = BUILD_TREE(X[left_idx],  y[left_idx],  depth + 1)
    right_child = BUILD_TREE(X[right_idx], y[right_idx], depth + 1)
    return NODE(feature, threshold, left_child, right_child)
```

### 2.2 Extra Randomized split (ExtraTrees uchun)

Oddiy CART barcha mumkin bo'lgan threshold'larni to'liq qidiradi (exhaustive search). **Extra Randomized Tree** esa:

```
function FIND_EXTRA_RANDOM_SPLIT(X, y, K):
    candidate_features = RANDOM_SUBSET(features, size=K)
    best_gain = -infinity
    for feature in candidate_features:
        min_v, max_v = MIN(X[:, feature]), MAX(X[:, feature])
        threshold = UNIFORM_RANDOM(min_v, max_v)     # <-- tasodifiy tanlanadi, qidirilmaydi!
        gain = WEIGHTED_IMPURITY_REDUCTION(...)
        if gain > best_gain:
            best_gain = gain
            best_feature, best_threshold = feature, threshold
    return best_feature, best_threshold, best_gain
```

Farq: har bir feature uchun **eng yaxshi threshold qidirilmaydi**, balki **bitta tasodifiy threshold tanlanadi**, so'ng shu tasodifiy kandidatlar orasidan eng yaxshi gain beruvchisi tanlanadi. Bu tree qurishni tezlashtiradi va variance'ni kamaytiradi (chunki har bir tree kamroq "greedy optimal"), lekin bias biroz oshadi.

### 2.3 Murakkablik tahlili (Complexity analysis)

$n$ — namunalar soni, $d$ — feature'lar soni, $D$ — tree chuqurligi bo'lsin.

- Bitta tugunda best split qidirish: har bir feature bo'yicha sort ($O(n \log n)$) + chiziqli o'tish ($O(n)$) → $O(d \cdot n \log n)$ (yoki oldindan sort qilingan bo'lsa $O(d \cdot n)$).
- Bir darajadagi (level) barcha tugunlar birgalikda $n$ ta namunani qamrab oladi → bir daraja uchun $O(d \cdot n \log n)$.
- Tree $D$ chuqurlikka ega → umumiy: $O(D \cdot d \cdot n \log n)$.
- Extra Randomized split uchun sort kerak emas (threshold tasodifiy) → bir daraja $O(d \cdot n)$, umumiy $O(D \cdot d \cdot n)$ — tezroq.

**Bashorat (predict) vaqti:** bitta namuna uchun $O(D)$ — ildizdan bargga bitta yo'l bosib o'tiladi.

---

## 3. Overfitting va uni cheklash

CART greedy va to'liq o'sganda trening ma'lumotini deyarli mukammal yodlab oladi (har bir barg — bitta yoki bir nechta namuna) → **overfitting**. Standart cheklovlar:

| Parametr | Ma'nosi |
|---|---|
| `max_depth` | Tree'ning maksimal chuqurligi |
| `min_samples_split` | Tugunni bo'lish uchun kerakli minimal namunalar soni |
| `min_samples_leaf` | Har bir bargda bo'lishi shart bo'lgan minimal namunalar soni |
| `min_impurity_decrease` | Split qabul qilinishi uchun kerakli minimal $\Delta$ |

Loyihada `max_depth` — class parametri sifatida talab qilingan, chunki bu overfitting'ni boshqarishning eng oddiy va samarali usuli.

---

## 4. Bagging va Random Forest

### 4.1 Nega o'rtachalash ishlaydi

Agar $B$ ta bir-biriga bog'liq bo'lmagan (independent), variance'i $\sigma^2$ bo'lgan bashoratchilarni (predictor) o'rtachalasak, natijaviy variance $\sigma^2/B$ ga tushadi. Ammo real tree'lar bir xil trening ma'lumotidan o'qitilgani uchun to'liq mustaqil emas — orasidagi korrelyatsiya $\rho$ bo'lsa:

$$Var\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Demak, $B \to \infty$ da variance $\rho\sigma^2$ ga yaqinlashadi (nolga emas!). Shuning uchun Random Forest tree'lar orasidagi korrelyatsiyani kamaytirishga harakat qiladi: **feature subsampling** ($M$ ta ustundan tasodifiy kichik qismini har bir split'da ko'rib chiqish) aynan shu maqsadda qo'llaniladi.

### 4.2 Pseudocode

```
function RANDOM_FOREST_FIT(X, y, n_trees, max_features, max_depth, seed):
    trees = []
    for b in range(n_trees):
        X_b, y_b = BOOTSTRAP_SAMPLE(X, y, seed + b)      # qaytarib qo'yish bilan tasodifiy tanlash, ~63% unikal
        tree = DecisionTree(max_depth, max_features=max_features)  # har bir split'da faqat max_features ta ustun ko'riladi
        tree.fit(X_b, y_b)
        trees.append(tree)
    return trees

function RANDOM_FOREST_PREDICT_PROBA(trees, X):
    return MEAN([tree.predict_proba(X) for tree in trees])
```

**Bootstrap sample**: $n$ ta namunani qaytarib qo'yish bilan tasodifiy tanlash. Bitta namunaning bitta bootstrap'ga tushmaslik ehtimoli $(1-1/n)^n \to e^{-1} \approx 0.368$, demak har bir tree o'rtacha trening ma'lumotining $\approx 63\%$ ida o'qiydi ("in-bag"), qolgan $\approx 37\%$ esa "out-of-bag" (OOB) — bu tabiiy validation sifatida ishlatilishi mumkin.

### 4.3 Murakkablik

$n_{trees}$ ta tree, har biri $O(D \cdot m \cdot n \log n)$ ($m$ — `max_features`) → umumiy $O(n_{trees} \cdot D \cdot m \cdot n \log n)$. Tree'lar bir-biridan mustaqil o'qitilgani uchun **parallellashtirish oson** — bu Random Forest'ning amaliy afzalliklaridan biri.

---

## 5. Gradient Boosting Decision Trees (GBDT)

### 5.1 Additive model va funksional gradient descent

GBDT $M$ ta tree'ning yig'indisi ko'rinishidagi modelni quradi:

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu \cdot h_m(x)$$

bu yerda $F_0$ — boshlang'ich bashorat (masalan, o'rtacha yoki log-odds), $h_m$ — $m$-tree, $\nu$ — **learning rate / shrinkage** (odatda 0.01–0.3).

Fikr: $F(x)$ ni funksiya sifatida qarab, uni loss'ning kamayish yo'nalishida "qadam bosib" yangilaymiz — xuddi parametrlar bo'yicha gradient descent kabi, faqat bu safar **funksiya fazosida**.

Har bir $m$-qadamda, har bir namuna uchun **pseudo-residual** (antigradient) hisoblanadi:

$$r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

Keyin yangi regression tree $h_m$ ushbu $r_i^{(m)}$ larni bashorat qilish uchun o'qitiladi (MSE loss bilan — shuning uchun GBDT ichida har doim **regression tree** ishlatiladi, hatto classification uchun ham!).

### 5.2 Regression uchun (MSE loss)

$$L(y, F) = \frac{1}{2}(y - F)^2 \quad \Rightarrow \quad r_i = y_i - F(x_i)$$

Bu holatda antigradient — oddiy **residual** (qoldiq xato). Intuitiv jihatdan: har bir keyingi tree oldingi bashoratning "qolgan xatosini" bashorat qiladi.

### 5.3 Binary classification uchun (Binary Cross-Entropy / Log Loss)

Model log-odds'ni bashorat qiladi: $F(x) = \log\frac{p}{1-p}$, ehtimollik esa sigmoid orqali olinadi: $p = \sigma(F(x)) = \frac{1}{1+e^{-F(x)}}$.

Loss (binary cross-entropy, $y \in \{0,1\}$):

$$L(y, F) = -\left[y \log p + (1-y)\log(1-p)\right]$$

Antigradientni hisoblaymiz ($p = \sigma(F)$, $\frac{dp}{dF} = p(1-p)$ ekanidan foydalanib):

$$r_i = -\frac{\partial L}{\partial F_i} = y_i - p_i = y_i - \sigma(F(x_i))$$

**Natija juda chiroyli:** antigradient — bu shunchaki "haqiqiy label minus bashorat qilingan ehtimollik". Har bir tree "hozirgi model qayerda ko'proq xato qilyapti (ehtimollik va haqiqiy label orasidagi farq katta)" ni bashorat qiladi.

### 5.4 To'liq pseudocode

```
function GBDT_FIT(X, y, n_trees, max_depth, max_features, learning_rate):
    F0 = LOG_ODDS(mean(y))            # boshlang'ich bashorat (klassifikatsiya uchun)
    F = [F0] * len(y)
    trees = []

    for m in range(n_trees):
        p = SIGMOID(F)
        residuals = y - p                              # antigradient (BCE uchun)
        tree_m = DecisionTreeRegressor(max_depth, max_features)
        tree_m.fit(X, residuals)                       # MSE bilan residual'larni bashorat qiladi
        update = tree_m.predict(X)
        F = F + learning_rate * update                 # incremental yangilash
        trees.append(tree_m)

    return trees, F0, learning_rate

function GBDT_PREDICT_PROBA(trees, F0, learning_rate, X):
    F = F0
    for tree in trees:
        F = F + learning_rate * tree.predict(X)
    return SIGMOID(F)
```

**Muhim:** bu yerda `max_features` bilan feature subsampling (Random Forest'dagidek) va `subsample` bilan qator subsampling qo'shish (**stochastic gradient boosting**) — bu GBDT'ga Random Forest'ning ba'zi "hiylalarini" olib kirish, variance'ni kamaytirish uchun qo'llaniladi (loyiha matnida ham qayd etilgan).

### 5.5 Murakkablik

$M$ ta tree, har biri $O(D \cdot m \cdot n \log n)$ → umumiy $O(M \cdot D \cdot m \cdot n \log n)$. Random Forest'dan farqli o'laroq, tree'lar **ketma-ket (sequential)** o'qitiladi — har biri oldingisiga bog'liq, shuning uchun bu jarayonni tree bo'yicha parallellashtirib bo'lmaydi (faqat bitta tree ichidagi split qidiruvini parallellashtirish mumkin).

### 5.6 Random Forest vs GBDT — asosiy farq

| | Random Forest | GBDT |
|---|---|---|
| Tree'lar orasidagi bog'liqlik | Mustaqil (parallel) | Ketma-ket, oldingi xatoni tuzatadi |
| Har bir tree | Chuqur, past bias / yuqori variance | Sayoz (shallow), yuqori bias / past variance |
| Asosiy maqsad | Variance'ni kamaytirish (averaging) | Bias'ni kamaytirish (additive correction) |
| Overfitting xavfi | Past (tree ko'paytirish deyarli xavfsiz) | Yuqori (juda ko'p tree yoki katta learning rate overfit qiladi) |
| Asosiy hyperparametr | `n_trees`, `max_depth`, `max_features` | `n_trees`, `learning_rate`, `max_depth` (odatda 3-8) |

---

## 6. XGBoost, LightGBM, CatBoost — nima farq qiladi

Loyiha talab qilgan qiyosiy tahlil uchun asosiy tushunchalar:

### 6.1 Tree o'sish strategiyasi

- **Depthwise (XGBoost, level-wise)**: tree daraja-daraja o'stiriladi, bir darajadagi barcha tugunlar bo'linadi, so'ng keyingi darajaga o'tiladi. Balandroq, muvozanatli (balanced) tree.
- **Leafwise (LightGBM)**: har safar **eng katta gain** beradigan bargni bo'ladi, tree balandligidan qat'iy nazar. Ko'proq gain'ga erishadi, lekin kichik dataset'larda overfitting xavfi yuqoriroq — shuning uchun `max_depth`/`num_leaves` bilan cheklanishi kerak.
- **Oblivious trees (Catboost, symmetric trees)**: bir darajadagi **barcha tugunlar bir xil feature va threshold** bo'yicha bo'linadi. Bu tree'ni juda tez baholashga (barcha bargga bitta bit-mask bilan yetish mumkin) va regularizatsiyaga (kamroq erkinlik darajasi) imkon beradi.

### 6.2 Kategorik feature'lar bilan ishlash

- **XGBoost**: an'anaviy holda oldindan encoding (one-hot/label) talab qiladi (yangi versiyalarda cheklangan native support bor).
- **LightGBM**: kategorik ustunlarni to'g'ridan-to'g'ri qabul qiladi, split sifatida `==` / `!=` (yoki kategoriyalar bo'linishi) ishlatadi — one-hot'siz.
- **CatBoost**: eng ilg'or — **ordered target encoding** ishlatadi: har bir namuna uchun target statistikasi faqat "o'zidan oldingi" (ma'lumotlar tasodifiy tartiblangan permutatsiyada) namunalar asosida hisoblanadi, bu target leakage'ning oldini oladi (naive target encoding'dagi kabi emas).

### 6.3 DART rejimi (XGBoost, LightGBM)

**DART = Dropout meets Additive Regression Trees.** Oddiy GBDT'da oxirgi qo'shilgan tree'lar dastlabki tree'larga nisbatan haddan tashqari katta ta'sirga ega bo'lib qolishi mumkin ("over-specialization"). DART har bir iteratsiyada **allaqachon qurilgan tree'larning tasodifiy qismini "tashlab yuboradi" (drop)**, yangi tree qolgan tree'lar ustiga qurilib, keyin tashlangan tree'larning og'irligi qayta normallashtiriladi. Bu neural network'lardagi dropout'ga o'xshash regularizatsiya effekti beradi.

### 6.4 Qisqacha jadval

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| O'sish | Depthwise | Leafwise | Oblivious (symmetric) |
| Kategorik feature | Encoding kerak | Native (`==`/`!=`) | Native (ordered target encoding) |
| Tezlik (katta dataset) | O'rtacha | Juda tez | O'rtacha-sekin |
| DART | Bor | Bor | Yo'q |
| Kuchli tomoni | Ishonchli, keng qo'llaniladigan default | Tezlik, xotira samaradorligi | Kategorik feature'lar ko'p bo'lgan ma'lumotlar |

---

## 7. Metrikani tushunish: Gini coefficient

Loyihada baholash metrikasi sifatida **Gini score** talab qilinadi (bu tugun impurity'sidagi Gini bilan **bir xil narsa emas!**). Bu — ranking sifatini o'lchovchi metrika:

$$Gini = 2 \cdot AUC - 1$$

bu yerda AUC — ROC egri chizig'i ostidagi yuza (Area Under the ROC Curve). $Gini \in [-1, 1]$: $1$ — mukammal ranking, $0$ — tasodifiy bashoratchi darajasida. sklearn'da: `2 * roc_auc_score(y_true, y_proba) - 1`.

---

## 8. Train/Valid/Test split — nega vaqt bo'yicha?

Loyiha `PurchDate` bo'yicha **xronologik split** talab qiladi (random split emas!). Sabab: real hayotda model kelajakdagi ma'lumotni bashorat qiladi, o'tmishdagi ma'lumot bilan o'qitilgan. Agar random split qilsak, model "kelajakdan" ma'lumot ko'rib o'qishi mumkin (temporal leakage) — bu validation/test natijalarini sun'iy ravishda yaxshi ko'rsatadi, lekin production'da model yomon ishlaydi. Shuning uchun:

```
train.PurchDate < valid.PurchDate < test.PurchDate
```

Xuddi shu sababdan, encoder'lar (LabelEncoder/OneHotEncoder/CountEncoder) **faqat train'ga fit qilinishi**, so'ng valid/test'ga faqat `transform` qilinishi kerak — aks holda bu ham data leakage bo'ladi.

---

## 9. Overfitting'ni aniqlash (train vs valid vs test Gini)

Loyihaning 8-bandi uchun asosiy mezon:
- Agar $Gini_{train} \gg Gini_{valid} \approx Gini_{test}$ — model overfit qilgan, lekin generalizatsiyasi barqaror.
- Agar $Gini_{valid} \gg Gini_{test}$ — bu **temporal drift** yoki valid/test o'rtasidagi taqsimot farqi belgisi (yoki hyperparametrlar valid'ga "moslashtirib" tanlangan, ya'ni valid'ga leakage/overfit qilingan).
- Sog'lom model uchun uchala qiymat bir-biriga yaqin bo'lishi, ozgina pasayish (train > valid > test) esa tabiiy hisoblanadi.

---

## 10. Xulosa: qanday ketma-ketlikda ishlash tavsiya etiladi

1. Ma'lumotni yuklang, `PurchDate` bo'yicha xronologik 33/33/33 split qiling.
2. Kategorik ustunlarni faqat train'ga fit qilingan encoder bilan kodlang (yangi kategoriyalarga tayyor bo'ling).
3. `Node` va `DecisionTreeClassifier`/`Regressor` class'larini Gini impurity bilan implement qiling, `fit`/`predict`/`predict_proba` metodlari bilan.
4. Extra randomized split funksiyasini qo'shing (ExtraTree uchun).
5. O'z DecisionTree'ingizni valid'da $Gini \ge 0.1$ ga yetkazing.
6. sklearn `DecisionTreeClassifier` bilan solishtiring (odatda tezroq va ko'proq optimallashtirilgan bo'ladi — C-implementatsiya, Cython, ko'proq split kriteriylari).
7. `RandomForestClassifier`'ni implement qiling ($Gini \ge 0.15$).
8. GBDT classifier'ni implement qiling (BCE gradient bilan, incremental).
9. LightGBM, XGBoost, CatBoost'ni sinab ko'ring, farqlarini tahlil qiling.
10. Eng yaxshi modelni tanlang, uni test'da baholang, train/valid/test Gini'larini solishtirib overfitting haqida xulosa chiqaring.
11. (Bonus) ExtraTreesClassifier'ni implement qiling ($Gini \ge 0.12$).

Keyingi (3-) fayl — shu ketma-ketlikni amalga oshiruvchi to'liq ishlaydigan kod (solution).
