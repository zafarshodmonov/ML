# Darslik: Klassifikatsiya algoritmlari (Logistic Regression, Naive Bayes, KNN, SVM) va metrikalar

Bu darslik ML4 loyihasidagi barcha tushunchalarni chuqur tushunish uchun tuzilgan: nazariya, matematik chiqarish, pseudocode, murakkablik tahlili va Python implementatsiyasi.

---

## 1. Klassifikatsiya masalasi nima?

Regressiyada biz uzluksiz qiymatni ($y \in \mathbb{R}$) bashorat qilamiz. Klassifikatsiyada esa $y$ chekli to'plamdan ($\{1, \dots, C\}$) tanlanadi.

**Nega bu "boshqacha ta'm"ga ega?** Chunki:
- Xatolikni o'lchash usuli boshqacha (MSE emas, logloss yoki accuracy kabi metrikalar);
- Chiqish ehtimol sifatida talqin qilinadi: $p(y=c \mid x)$;
- Qaror chegarasi (decision boundary) tushunchasi paydo bo'ladi — fazoni klasslarga ajratuvchi sirt.

**Binar klassifikatsiya** — $C=2$ holat, $y \in \{0, 1\}$. Bu darslikning asosiy predmeti.

### Misol bilan tushuntirish

Aytaylik, bankka kredit so'rovi keldi. Bizda ikkita belgi bor: `daromad` (ming so'mda) va `qarz_nisbati` (0 dan 1 gacha). Maqsad — mijoz kreditni to'lay oladimi (`y=1`) yoki yo'qmi (`y=0`) bashorat qilish.

| daromad | qarz_nisbati | y (to'laydi=1) |
|---|---|---|
| 5000 | 0.2 | 1 |
| 1200 | 0.8 | 0 |
| 3000 | 0.4 | 1 |

Model vazifasi — shu ikki belgi asosida yangi mijoz uchun $p(y=1)$ ni bashorat qilish.

---

## 2. Logistic Regression

### 2.1. Model tuzilishi

$$z = \mathbf{w}^T\mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_D x_D + b$$
$$p(y=1\mid x) = \sigma(z) = \frac{1}{1+e^{-z}}$$

**Sigmoid funksiya** — $z \to -\infty$ da 0 ga, $z \to +\infty$ da 1 ga intiladi, $z=0$ da 0.5 ni beradi. Grafigi "S" harfiga o'xshaydi.

**Nega chiziqli kombinatsiyani to'g'ridan-to'g'ri ehtimol sifatida ishlatmaymiz?** Chunki $\mathbf{w}^T\mathbf{x}+b$ har qanday haqiqiy son bo'lishi mumkin ($-\infty$ dan $+\infty$ gacha), ehtimol esa $[0,1]$ oralig'ida bo'lishi shart. Sigmoid aynan shu "siqish" vazifasini bajaradi.

### 2.2. O'qitish: Maximum Likelihood Estimation (MLE)

Barcha $N$ ta namuna mustaqil deb faraz qilinadi. Ularning birgalikdagi ehtimoli (likelihood):

$$L(\boldsymbol\theta) = \prod_{i=1}^N p(y_i \mid x_i, \boldsymbol\theta) = \prod_{i=1}^N \sigma(z_i)^{y_i}\,(1-\sigma(z_i))^{1-y_i}$$

Bu — Bernoulli taqsimotining ehtimollik funksiyasi (chunki $y_i \in \{0,1\}$).

Logarifm olib, manfiy ishora bilan **Negative Log Likelihood (NLL)** hosil qilamiz:

$$\text{NLL}(\boldsymbol\theta) = -\sum_{i=1}^N \Big[y_i \log \sigma(z_i) + (1-y_i)\log(1-\sigma(z_i))\Big]$$

Bu ifoda **binary cross-entropy (logloss)** deb ham ataladi. MLE — NLL'ni minimallashtirish bilan bir xil.

### 2.3. Gradientni qo'lda chiqarish

Bitta namuna uchun loss: $\ell_i = -\big[y_i\log\sigma(z_i) + (1-y_i)\log(1-\sigma(z_i))\big]$, bu yerda $z_i = \mathbf{w}^Tx_i+b$.

**1-qadam.** $\sigma(z)$ ning hosilasi:
$$\sigma'(z) = \sigma(z)(1-\sigma(z))$$

**2-qadam.** Zanjir qoidasi (chain rule) bo'yicha $\ell_i$ ning $z_i$ bo'yicha hosilasi:
$$\frac{\partial \ell_i}{\partial z_i} = -\left[\frac{y_i}{\sigma(z_i)} - \frac{1-y_i}{1-\sigma(z_i)}\right]\sigma(z_i)(1-\sigma(z_i)) = \sigma(z_i) - y_i$$

Bu — juda chiroyli natija: gradient shunchaki **bashorat va haqiqiy qiymat orasidagi farq**.

**3-qadam.** $z_i$ ning $\mathbf{w}$ va $b$ bo'yicha hosilasi: $\partial z_i/\partial\mathbf{w} = x_i$, $\partial z_i/\partial b = 1$.

**Yakuniy gradientlar** (barcha namunalar bo'yicha):
$$\frac{\partial \text{NLL}}{\partial \mathbf{w}} = \sum_{i=1}^N (\sigma(z_i)-y_i)\,x_i, \qquad \frac{\partial \text{NLL}}{\partial b} = \sum_{i=1}^N (\sigma(z_i)-y_i)$$

### 2.4. Stochastic Gradient Descent (SGD)

To'liq gradient barcha $N$ ta namunani ko'rib chiqishni talab qiladi — millionlab qatorlar uchun sekin. SGD esa har qadamda faqat bitta (yoki kichik `batch`) namuna asosida yangilaydi:

```
w, b ni tasodifiy (yoki nol) qiymatlar bilan boshlash
lr (learning rate) ni tanlash, masalan 0.01
epochs ta marta takrorlash:
    ma'lumotlarni aralashtirish
    har bir namuna (x_i, y_i) uchun:
        z = w^T x_i + b
        p = sigmoid(z)
        grad_w = (p - y_i) * x_i
        grad_b = (p - y_i)
        w = w - lr * grad_w
        b = b - lr * grad_b
```

**Murakkablik:** Har bir epoch $O(N \cdot D)$ vaqt talab qiladi ($D$ — belgilar soni). Nyuton usuli esa Hessianni hisoblash va teskarisini olish uchun har qadamda $O(D^3)$ talab qiladi — shuning uchun katta $D$ da SGD ancha qulayroq.

### 2.5. Kuchli va zaif tomonlari (xulosa)

| Jihat | Baho |
|---|---|
| Talqin qilinuvchanlik | Yuqori (har bir belgi og'irligi tushunarli) |
| Aniqlik | O'rtacha (chiziqli qaror chegarasi) |
| Tezlik | Tez o'qitiladi va bashorat qiladi |
| Overfitting'ga moyillik | Past (kam parametr) |
| Nochiziqli bog'liqlik | Faqat belgi muhandisligi (feature engineering) orqali |

---

## 3. Naive Bayes

### 3.1. Bayes teoremasi

$$p(y=c\mid x) = \frac{p(y=c)\,p(x\mid y=c)}{p(x)}$$

- $p(y=c)$ — **prior**: klass haqidagi oldindan bilim (masalan, klassning ma'lumotlardagi ulushi);
- $p(x\mid y=c)$ — **likelihood**: shu klassda bunday belgilar uchrash ehtimoli;
- $p(x)$ — **evidence**: normallashtiruvchi konstanta (barcha klasslar uchun bir xil).

Bashorat qilishda faqat $\arg\max_c$ kerak bo'lgani uchun, $p(x)$ ni hisoblash shart emas:

$$\hat{y} = \arg\max_c\ p(y=c)\,p(x\mid y=c)$$

### 3.2. "Naive" taxmin nima uchun kerak?

$p(x\mid y=c) = p(x_1, x_2, \dots, x_D \mid y=c)$ ni to'g'ridan-to'g'ri baholash uchun barcha belgilarning **birgalikdagi** taqsimotini bilish kerak — bu eksponensial ko'p parametr talab qiladi ($D$ ta binar belgi uchun $2^D$ ta kombinatsiya).

Naive taxmin: belgilar klass berilganda **shartli mustaqil**:

$$p(x\mid y=c) = \prod_{d=1}^D p(x_d \mid y=c)$$

Endi faqat $D$ ta bitta o'lchovli taqsimotni baholash kifoya — bu chiziqli miqdorda parametr degani.

### 3.3. O'qitish (parametrlarni baholash)

**Prior:** $\hat{p}(y=c) = \dfrac{\text{c-klassdagi namunalar soni}}{N}$ (shunchaki sanoq).

**Binar belgi** uchun (Bernoulli): $\hat{p}(x_d=1\mid y=c) = \dfrac{\text{c-klassda } x_d=1\text{ bo'lgan namunalar soni}}{\text{c-klassdagi namunalar soni}}$

**Uzluksiz belgi** uchun (Gaussian NB): har bir klass $c$ va belgi $d$ uchun o'rtacha $\mu_{c,d}$ va dispersiya $\sigma^2_{c,d}$ hisoblanadi:

$$p(x_d \mid y=c) = \frac{1}{\sqrt{2\pi\sigma^2_{c,d}}}\exp\left(-\frac{(x_d-\mu_{c,d})^2}{2\sigma^2_{c,d}}\right)$$

### 3.4. Pseudocode

```
Fit(X, y):
    har bir klass c uchun:
        prior[c] = count(y==c) / N
        har bir belgi d uchun:
            agar uzluksiz bo'lsa: mu[c][d], sigma2[c][d] ni hisoblash
            agar diskret bo'lsa: har bir qiymat uchun chastotani hisoblash

Predict(x):
    har bir klass c uchun:
        score[c] = log(prior[c])
        har bir belgi d uchun:
            score[c] += log(p(x_d | y=c))     # logarifm ishlatiladi, ko'paytmalar juda kichik bo'lib ketmasligi uchun
    qaytarish: argmax_c score[c]
```

**Nega logarifm ishlatiladi?** $D$ katta bo'lganda, ko'plab kichik ehtimollarni ko'paytirish sonli to'lib ketish (underflow)ga olib kelishi mumkin. Logarifm ko'paytmani yig'indiga aylantiradi va bu muammoni hal qiladi.

**Murakkablik:** O'qitish — $O(N\cdot D)$ (bir marta o'tish), bashorat — bitta namuna uchun $O(D\cdot C)$. Bu KNN va hatto Logistic Regression'ga qaraganda ham tezroq bo'lishi mumkin.

### 3.5. Kuchli va zaif tomonlari

- ✅ Juda tez o'qitiladi (yopiq formula, optimallashtirish shart emas);
- ✅ Yo'qolgan qiymatlarni tabiiy tarzda qayta ishlaydi (shunchaki hisoblashda tashlab ketish mumkin);
- ✅ Kam ma'lumotda ham ishlaydi;
- ❌ "Mustaqillik" taxmini deyarli hech qachon to'g'ri kelmaydi;
- ❌ Belgilar orasidagi korrelyatsiyani hisobga ololmaydi.

---

## 4. KNN (K-Nearest Neighbors)

### 4.1. G'oya

KNN — **noparametrik** model: u "o'qitish" bosqichida hech narsani optimallashtirmaydi, shunchaki barcha ma'lumotni xotirada saqlab qoladi (**lazy learning**).

Bashorat qilish uchun:
1. Yangi $x$ nuqtadan o'qitish to'plamidagi barcha nuqtalargacha masofani hisoblash;
2. Eng yaqin $K$ ta nuqtani tanlash;
3. Klassifikatsiya uchun: shu $K$ ta qo'shnining yorliqlari orasida ko'pchilik ovoz (majority vote); regressiya uchun: o'rtachasini olish.

### 4.2. Masofa metrikalari

$$L2\ (\text{Euclid}):\quad d(x,x') = \sqrt{\sum_{d=1}^D (x_d-x'_d)^2}$$
$$L1\ (\text{Manhattan}):\quad d(x,x') = \sum_{d=1}^D |x_d-x'_d|$$

### 4.3. Pseudocode

```
Predict(x, X_train, y_train, K):
    masofalar = []
    har bir (x_i, y_i) juftlik uchun X_train, y_train dan:
        d_i = masofa(x, x_i)
        masofalar.append((d_i, y_i))
    masofalar ni d_i bo'yicha o'sish tartibida saralash
    K ta eng yaqin qo'shnini olish: masofalar[0:K]
    qaytarish: shu K ta orasida eng ko'p uchragan y_i (klassifikatsiya)
              yoki o'rtacha y_i (regressiya)
```

### 4.4. Murakkablik va "o'lchamlar la'nati"

- **O'qitish:** $O(1)$ — shunchaki ma'lumotni saqlash;
- **Bashorat (naiv):** bitta namuna uchun $O(N\cdot D)$ (barcha $N$ ta o'qitish namunasigacha masofa) + saralash $O(N\log N)$;
- **Xotira:** $O(N\cdot D)$ — butun train to'plami saqlanishi kerak.

**Curse of dimensionality (o'lchamlar la'nati):** $D$ ortishi bilan barcha nuqtalar bir-biridan "bir xil darajada uzoq" bo'lib qoladi — L2/L1 masofa o'zining informativligini yo'qota boshlaydi. Masalan, $D=2$ da yaqin qo'shnilar aniq ko'rinadi, lekin $D=1000$ da deyarli barcha nuqtalar orasidagi masofa taxminan bir xil bo'lib qoladi.

### 4.5. $K$ ni tanlash

- Kichik $K$ (masalan, $K=1$) — shovqinga juda sezgir, overfitting;
- Katta $K$ — model "silliqlashadi", underfitting xavfi;
- $K$ odatda cross-validation orqali tanlanadi (masalan, $K \in \{3,5,7,\dots,31\}$).

---

## 5. SVM (qisqacha)

SVM klasslarni ajratuvchi giperteklislikni **maksimal margin** (eng yaqin nuqtalargacha bo'lgan masofani maksimallashtiruvchi) tamoyili bilan quradi. Faqat "chegara yaqinidagi" nuqtalar (**support vectors**) qaror chegarasiga ta'sir qiladi. Nochiziqli SVM **kernel trick** yordamida belgilar fazosini yuqori o'lchamga proyeksiya qiladi (masalan, RBF kernel), bu yerda klasslar chiziqli ajraladigan bo'lib qoladi — hisoblashda esa aslida yuqori o'lchamli fazoga o'tish shart emas, faqat kernel funksiyasi (skalyar ko'paytma) hisoblanadi.

---

## 6. Train/Validation/Test bo'lish (vaqt bo'yicha)

Odatiy tasodifiy bo'lishdan farqli o'laroq, bu loyihada **vaqt bo'yicha** bo'lish talab qilinadi, chunki `PurchDate` bor va real hayotda kelajakni bashorat qilamiz, o'tmishni emas:

```
barcha sanalarni saralash (o'sish tartibida)
N = umumiy namunalar soni
train  = birinchi N/3 (eng erta sanalar)
valid  = o'rtadagi N/3
test   = oxirgi N/3 (eng kech sanalar)
tekshirish: max(train.PurchDate) < min(valid.PurchDate) < min(test.PurchDate) shartiga yaqin
```

**Nega bu muhim?** Agar tasodifiy bo'lsak, model "kelajakdagi" ma'lumotdan o'rganib, "o'tmishni" bashorat qilishi mumkin — bu real hayotda mumkin emas va **data leakage**ning bir turi hisoblanadi. Vaqt bo'yicha bo'lish modelning haqiqiy amaliy sifatini ko'rsatadi.

---

## 7. Kategorial belgilarni kodlash

| Usul | Qanday ishlaydi | Xavf |
|---|---|---|
| **LabelEncoder** | Har bir kategoriyaga butun son beradi (0,1,2,...) | Model tartibni "mavjud" deb noto'g'ri tushunishi mumkin (masalan LR uchun) |
| **OneHotEncoder** | Har bir kategoriya uchun alohida binar ustun | Kategoriya soni ko'p bo'lsa, o'lchamlar portlaydi |
| **Count/Frequency Encoding** | Kategoriyani uning chastotasi bilan almashtiradi | Train'da ko'rilmagan yangi kategoriyalarni ham qayta ishlay oladi |

**Data leakage'dan saqlanish qoidasi:** Encoder faqat **train** to'plamiga `fit()` qilinishi, so'ng **validation** va **test**'ga faqat `transform()` qilinishi kerak. Aks holda, validation/test haqidagi ma'lumot (masalan, kategoriya chastotalari) o'qitishga "sizib kiradi" va baho haqiqiydan yaxshiroq ko'rinadi.

Agar validation/test'da train'da bo'lmagan yangi kategoriya uchrasa — Count Encoding kabi usullar (yoki "noma'lum" maxsus kategoriya belgilash) yordam beradi.

---

## 8. Feature Engineering (nochiziqli belgilar)

Chiziqli model (LR) faqat chiziqli qaror chegarasini chiza oladi. Murakkab bog'liqliklarni "chiziqli qilib ko'rsatish" uchun:

- **Nisbatlar:** `feature1 / feature2` (masalan, `qarz / daromad`);
- **Guruh statistikalari (target/feature encoding):**
  ```python
  df['kategoriya_orta'] = df['kategoriya'].map(
      df.groupby('kategoriya')['uzluksiz_belgi'].mean()
  )
  ```
  Bu — kategorial belgini shu kategoriyadagi boshqa uzluksiz belgining o'rtachasi bilan almashtirish;
- **Logarifm:** `log(1 + x)` — "yog'li dum"li taqsimotlarni silliqlashtiradi;
- **Binlash:** uzluksiz belgini intervallarga bo'lish (masalan, yosh: 0-18, 18-30, 30-50, 50+).

**Ehtiyot bo'ling:** guruh statistikalarini hisoblaganda ham faqat **train** ma'lumotidan foydalaning, aks holda yana data leakage yuzaga keladi.

---

## 9. Metrikalar — chuqur tahlil

### 9.1. Confusion Matrix (Chalkashlik matritsasi)

|  | Bashorat: 0 | Bashorat: 1 |
|---|---|---|
| **Haqiqiy: 0** | TN (to'g'ri manfiy) | FP (yolg'on musbat) |
| **Haqiqiy: 1** | FN (yolg'on manfiy) | TP (to'g'ri musbat) |

### 9.2. Precision, Recall, F1, Fbeta

$$Precision = \frac{TP}{TP+FP} \qquad Recall = \frac{TP}{TP+FN}$$

- Precision past → model ko'p **yolg'on hushyorlik** (false alarm) beradi;
- Recall past → model ko'p haqiqiy holatlarni **o'tkazib yuboradi**.

$$F1 = 2\cdot\frac{P\cdot R}{P+R}, \qquad F_\beta = (1+\beta^2)\cdot\frac{P\cdot R}{\beta^2 P + R}$$

**Savol (README'dan):** *Qaysi $\beta$ qiymati Recall'ga ko'proq og'irlik beradi?* — Javob: $\beta > 1$ (masalan, $\beta=2$) Recall'ga ko'proq og'irlik beradi, chunki formulada $\beta^2$ maxrajning Precision hadini "kichraytiradi", ya'ni Recall natijaga ko'proq ta'sir qiladi. $\beta < 1$ (masalan, $\beta=0.5$) esa Precision'ga ko'proq og'irlik beradi.

### 9.3. ROC va AUC ROC

**ROC egri chizig'i** — turli chegaralar (threshold) uchun **True Positive Rate** ($TPR=Recall$) va **False Positive Rate** ($FPR = FP/(FP+TN)$) o'rtasidagi bog'liqlikni chizadi.

**AUC ROC** — bu egri chiziq ostidagi maydon. Talqini: *tasodifiy tanlangan 1-klass namunasining bashorat qilingan ehtimoli, tasodifiy tanlangan 0-klass namunasining ehtimolidan katta bo'lish ehtimoli.*

**Pseudocode (AUC ROC ni hisoblash — rank asosida):**

```
AUC_ROC(y_true, y_scores):
    pos = y_scores[y_true == 1]   # musbat klass bashoratlari
    neg = y_scores[y_true == 0]   # manfiy klass bashoratlari
    juftliklar_soni = len(pos) * len(neg)
    yutgan_juftliklar = 0
    har bir p in pos, n in neg uchun:
        agar p > n:      yutgan_juftliklar += 1
        agar p == n:     yutgan_juftliklar += 0.5     # durrang holat
    qaytarish: yutgan_juftliklar / juftliklar_soni
```

**Murakkablik:** to'g'ridan-to'g'ri (ikki tsikl) $O(N_{pos}\cdot N_{neg})$ — katta ma'lumotlar uchun sekin. Amalda **saralash orqali** (rank statistikasi, Mann-Whitney U testiga o'xshash) $O(N\log N)$ vaqtda hisoblanadi: barcha bashoratlarni saralab, har bir musbat namunaning "rank"ini yig'ib chiqish yetarli.

### 9.4. AUC PR va Gini

**AUC PR** — Precision-Recall egri chizig'i ostidagi maydon, barcha chegaralar bo'yicha. Noldan katta bo'lgan sinflar juda kam uchraydigan (imbalanced) ma'lumotlarda AUC ROC'ga qaraganda ko'proq informativ bo'lishi mumkin.

**Gini koeffitsienti:**
$$Gini = |2\cdot AUC\_ROC - 1|$$

AUC ROC = 0.5 (tasodifiy model) → Gini = 0. AUC ROC = 1.0 (mukammal model) → Gini = 1.0.

---

## 10. Regularizatsiya (L1) va belgi tanlash

L1 regularizatsiya loss funksiyasiga $\lambda\sum_d |w_d|$ hadini qo'shadi. Bu ba'zi $w_d$ larni aynan **0** ga tenglashtirishga moyillik qiladi (L2'dan farqli o'laroq), shuning uchun L1 tabiiy ravishda **belgi tanlash (feature selection)** vazifasini ham bajaradi — koeffitsienti 0 bo'lgan belgilar modelga "kerak emas" deb hisoblanadi.

---

## 11. Umumiy xulosa jadvali

| Model | Turi | O'qitish tezligi | Talqin qilinuvchanlik | Nochiziqlilik |
|---|---|---|---|---|
| Logistic Regression | Parametrik | Tez (SGD) | Yuqori | Yo'q (feature eng. bilan qo'shiladi) |
| Naive Bayes | Parametrik | Juda tez (yopiq formula) | O'rtacha | Yo'q |
| KNN | Noparametrik | Bashoratda sekin | Past | Ha (tabiiy ravishda) |
| SVM (kernel) | Yarim-parametrik | O'rtacha/sekin | Past | Ha (kernel orqali) |

---

Keyingi faylda (`03_yechim.ipynb`) shu barcha tushunchalar amaliy kodga aylantirilgan — loyihadagi barcha 12 ta vazifa bosqichma-bosqich yechilgan.
