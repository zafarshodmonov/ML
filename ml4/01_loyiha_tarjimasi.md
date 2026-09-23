# O'qituvchili o'qitish (Supervised Learning). Klassifikatsiya

**Qisqacha mazmun:** bu loyiha klassifikatsiya masalalari va ular bilan bog'liq ML algoritmlariga kirish hisoblanadi.

## I bob. Kirish so'zi (Preamble)

Siz allaqachon regressiya masalalarini o'rgangansiz — masalan, odamning yoshini, aksiyalar narxini yoki ertangi haroratni bashorat qilish. Lekin agar biz **kategorial** (toifali) belgilarni bashorat qilmoqchi bo'lsak-chi? Masalan: bemorda ma'lum bir kasallik bormi yoki yo'qmi, yoki yo'l belgisidagi raqam nima? Bunday masalalar **klassifikatsiya masalalari** deb ataladi. Ular regressiya masalalariga o'xshaydi, lekin boshqacha "ta'mga" ega.

Ushbu loyihada 3 turdagi klassifikatsiya modeli ko'rib chiqiladi:
- **Logistic Regression** — Chiziqli regressiyaning o'ziga xos "o'zgartirilgan" ko'rinishi;
- **Naive Bayes Classifier** — maxsus klassifikatsiya uchun mo'ljallangan model;
- **KNN (K eng yaqin qo'shnilar)** — ham regressiya, ham klassifikatsiyaga qo'llash mumkin bo'lgan model.

## II bob. Kirish (Introduction)

Klassifikatsiya masalalari regressiya masalalariga o'xshash. Klassifikatsiyada chiqish maydoni — bu **C** ta tartiblanmagan va **bir-birini istisno qiluvchi** yorliqlar (klasslar) to'plami. Bizga quyidagi funksiyani topish kerak:

$$f: X \rightarrow Y,\quad Y \in \{1, 2, \dots, C\},\quad X - \text{o'qitish to'plami}$$

Agar bizda faqat ikkita klass bo'lsa ($Y \in \{0, 1\}$), bu masala **binary classification (binar klassifikatsiya)** deb ataladi. Ushbu darslikda biz asosan binar masalalarga e'tibor qaratamiz.

**Binar klassifikatsiyaga hayotiy misollar:**
- Bemorda ma'lum kasallik bor-yo'qligi;
- Tranzaksiya firibgarlik (fraud) ekan-emasligi;
- Foydalanuvchi mobil ilovadagi reklama bannerini bosadimi yoki yo'qmi;
- Aksiya narxi keyingi bir necha soniyada ko'tariladimi yoki tushadimi (treyding uchun foydali);
- Mijoz kreditni qaytaradimi yoki yo'qmi (default ehtimoli — PD; bankingda foydali).

Ehtimollik nuqtai nazaridan, binar klassifikatsiya masalasini — berilgan x namunaning 1-yorliqqa ega bo'lish ehtimolini baholash sifatida ko'rish mumkin:

$$p(y=1\mid \mathbf{x}, \boldsymbol\theta)$$

bu yerda $\mathbf{x}$ — namuna, $\boldsymbol\theta$ — modelning parametrlari.

Binar masalalar klassifikatsiyaning xususiy holati (agar **C > 2** bo'lsa, bu **multiclass (ko'p klassli) masala** deb ataladi). Binar yondashuvlardan ko'p klassli masalalarni yechishda ham foydalanish mumkin: **one-vs-all** usuli — ***C*** ta alohida binar model o'qitiladi, har bir *i* modeli uchun maqsad (target) 1 deb belgilanadi agar y = i bo'lsa, aks holda 0. Bashorat qilish uchun ***C*** ta olingan bahodan eng kattasini (argmax) tanlaymiz:

$$\arg\max_{i \in \{1,\dots,C\}} p(y=i\mid \mathbf{x}, \boldsymbol\theta)$$

Ba'zida (biznes masalalarini yechishda) regressiyaga o'xshash vazifalarni binar masalaga aylantirish qulay bo'ladi: masalan, insonning daromadi mamlakat bo'yicha o'rtacha daromaddan yuqori yoki yo'qligini bashorat qilish. Va aksincha — klassifikatsiya masalasini, yorliqlarni tartiblab, regressiya masalasiga aylantirish mumkin.

### Logistic Regression

#### Formulasi

Logistic Regression — bu quyidagi ko'rinishdagi model:

$$p(y=1\mid \mathbf{x}, \boldsymbol\theta) = \sigma(a) = \frac{1}{1+e^{-a}}, \quad a = \mathbf{w}^T\mathbf{x} + b,\ \ \boldsymbol\theta=(\mathbf{w}, b)$$

$\sigma$ funksiyasi **sigmoid** deb ataladi. Logistic Regression — ***X*** matritsasining belgilarini chiziqli birlashtirib, ustiga ***b*** siljish (bias) hadini qo'shishdan iborat. Natijani sigmoid funksiyasi orqali "siqib", 0 va 1 oralig'iga tushiramiz. "Regression" so'zi nomida bo'lsa-da, bu algoritm regressiya masalasiga hech qanday aloqasi yo'q.

Logistic Regression aniqlik (accuracy) bo'yicha eng yaxshi algoritm emas, lekin u **talqin qilinadigan (interpretable)** algoritm: uni har qanday menejer yoki foydalanuvchiga tushuntirish oson.

Parametrlarni qanday tanlaymiz? Buning uchun **Maksimal Ehtimollik Bahosi (Maximum Likelihood Estimation, MLE)** qo'llaniladi — bizga berilgan ***X*** va ***Y*** ma'lumotlar to'plamini olish ehtimolini maksimallashtiruvchi parametrlarni topish kerak. Barcha namunalar mustaqil deb hisoblanadi, shuning uchun ehtimollik zichlik funksiyalarining ko'paytmasi olinadi, so'ng logarifm qo'llaniladi. Manfiy ishora bilan olingan natija **Negative Log Likelihood (NLL)** deb ataladi. MLE'dan foydalanish — bu NLL'ni minimallashtirish demakdir.

NLL'ni minimallashtirish uchun har qanday gradientga asoslangan optimallashtirish algoritmidan foydalanish mumkin — masalan, **Stochastic Gradient Descent (SGD)** yoki ikkinchi tartibli usullar, masalan **Nyuton usuli**. Minglab yoki millionlab namunalarda o'qitishda SGD Nyuton usuliga qaraganda qulayroq. SGD nafaqat LR modellarini o'qitishda, balki **Deep Learning** modellarining parametrlarini optimallashtirishning asosiy usuli hisoblanadi.

LR modellari uchun odatda Linear Regression uchun ishlatiladigan xuddi shu preprocessing (oldindan ishlov berish) usullari qo'llaniladi: masshtablash (scaling), kategorial belgilar uchun one-hot-encoding va h.k. NaN qiymatlar uchun o'rtacha/moda bilan to'ldirish (imputation) va qo'shimcha "NaN yoki yo'q" flag-belgisini qo'shish mumkin.

#### Kuchli va zaif tomonlari

Logistic Regression talqin qilinuvchanligi tufayli qattiq tartibga solinadigan sohalarda (masalan, banking) yoki to'liq tushuntirilishi lozim bo'lgan ilmiy sohalarda jozibador. Modelning oddiy shakli tushunarlilikni ta'minlaydi, lekin aniqlikni pasaytiradi. LR modeli quyidagi miqdorda belgiga ega:

*son-belgilar-soni + kategorial-belgilardagi-barcha-kategoriyalar-soni + 1*

shuning uchun bu algoritm murakkabroq algoritmlarga qaraganda overfitting (qayta o'rganib ketish)ga kamroq moyil.

Modelni yaxshilash uchun kirish ***X***ni **nochiziqli transformatsiyalar** (φ funksiyasi) orqali o'zgartirish mumkin. Masalan:
- Belgilar logarifmi (ayniqsa "yog'li dum"li belgilar uchun);
- Polinomial belgilar;
- Belgilar bo'linmasi: belgi1/belgi2;
- Davriy komponentlarni ushlash uchun **sin(x)** yoki **cos(x)**;
- Daraxt-chiqishga o'xshash belgilar: "belgi #1 > k sonidanmi" kabi taqqoslashlar.

Belgi qo'shish kamchiliklarga ham ega: o'rtacha hisobda ko'proq belgilar ko'proq overfitting'ga olib kelishi mumkin. Murakkab belgilardan foydalanish modelni kamroq talqin qilinadigan qiladi, lekin ba'zida tushunarlilik va aniqlik o'rtasida muvozanat topish mumkin (masalan, kredit skoring kartalarida uzluksiz belgilarni **binlash** — guruhlarga bo'lish).

### Naive Bayes

#### Formulasi

Klassifikatsiya masalalariga yana bir oddiy yondashuv — **Naive Bayes (NB) algoritmi**. NB'ning asosi quyidagi taxmin: barcha belgilar klass yorlig'i berilgan holda **shartli mustaqil** deb qabul qilinadi (shuning uchun "Naive" — "sodda/soddalashtirilgan" deb ataladi). Bu taxmin real hayotda deyarli to'g'ri kelmaydi, lekin NB klassifikatorlari ko'pincha bu taxmin buzilgan bo'lsa ham yaxshi natija beradi.

Usulning asosiy g'oyasi — Bayes formulasi orqali posterior ehtimollikni hisoblash:

$$\text{posterior} = \frac{\text{prior}\cdot\text{likelihood}}{\text{evidence}}$$

$$p(y=c\mid\mathbf{x}) = \frac{p(y=c)\,p(\mathbf{x}\mid y=c)}{p(\mathbf{x})} = \frac{p(y=c)\,p(\mathbf{x}\mid y=c)}{\sum_{c'} p(y=c')\,p(\mathbf{x}\mid y=c')}$$

Maqsad — **Maksimal A Posteriori (MAP)** bahosini topish. Maxrajdagi qiymat ma'lumotlar to'plami uchun doim bir xil (konstanta) bo'lgani sabab, uni e'tiborsiz qoldirish mumkin. NB taxmini bajarilsa, belgilar vektorini olish ehtimolini quyidagicha yozish mumkin:

$$p(\mathbf{x}\mid y=c) = \prod_{d=1}^{D} p(x_d\mid y=c)$$

Demak, Posterior'ni maksimallashtirish masalasi quyidagiga teng:

$$p(y=c)\,p(\mathbf{x}\mid y=c) \rightarrow \max$$

Klassifikatorni o'qitish uchun:
1. Prior taqsimot shaklini belgilash kerak;
2. Belgilarning taqsimot shaklini belgilash kerak;
3. Ikkalasini ham baholash kerak.

Prior odatda Bernoulli yoki multinomial taqsimot orqali modellashtiriladi (klass ulushi asosida). Belgilar uchun:
- **Binar belgilar** — Bernoulli taqsimoti;
- **Uzluksiz belgilar** — normal taqsimot (yoki ularni binarlashtirib Bernoulli qo'llash mumkin);
- **Kategorial belgilar** — Categorical (Multinomial) taqsimoti.

NB modeli parametrlarini MLE orqali hisoblash mumkin — bu faqat oddiy hisoblash: belgilar sanog'i (binar/kategorial holatda) yoki o'rtachadan kvadratik og'ish (uzluksiz holatda). Ya'ni murakkab optimallashtirish protsedurasi kerak emas — faqat tez va sodda hisob-kitoblar.

#### Kuchli va zaif tomonlari

NB taxmini — asosiy cheklov: ko'p real vaziyatlarda NB sifat jihatidan yomon natija beradi. Uning asosiy afzalliklari — o'qitishning soddaligi va yo'qolgan qiymatlarni tabiiy tarzda qayta ishlash qobiliyati.

### KNN

#### Formulasi

LR va NB modellari **parametrlarga** ega. O'qitish bosqichida biz parametrlarni baholaymiz, so'ng ma'lumotlar to'plamini "tashlab yuborishimiz" mumkin. **Noparametrik modellar**da esa parametrlarning aniq to'plami yo'q — modelning effektiv parametrlar soni ma'lumotlar hajmi bilan birga o'sib boradi. KNN — shunday modelga misol.

G'oya sodda: yangi ***x*** kirishni klassifikatsiya qilish uchun, o'qitish to'plamidan ***x***ga eng yaqin ***K*** ta namunani topamiz (L2 yoki L1 masofa kabi biror metrika yordamida), so'ng olingan ***K*** ta qo'shnining yorliqlarini o'rtachalashtirib bashoratni olamiz. Bu yerda optimallashtirish yo'q — biz shunchaki eng yaqin qo'shnilarni topib, ularni xotirada saqlaymiz.

#### Kuchli va zaif tomonlari

KNN'ning asosiy qiyinchiligi — masofa metrikasini tanlash. L1/L2 bir turdagi va uzluksiz ma'lumotlar uchun mos, lekin boshqa holatlarda namunalar orasidagi masofa unchalik informativ bo'lmasligi mumkin. Yana bir muammo — **o'lchamlar la'nati (curse of dimensionality)**: belgilar soni ko'p bo'lganda namunalar bir-biridan juda uzoqlashib ketadi. KNN'ni o'qitish ko'p xotira talab qiladi, chunki xulosa chiqarish uchun barcha o'qitish ma'lumotlarini saqlab qolish kerak. Bugungi kunda KNN boshqa ML algoritmlari (masalan, tavsiya tizimlari) uchun asosiy blok sifatida ishlatiladi.

### SVM

Klassifikatsiya uchun yana bir model — **Support Vector Machines (SVM)**. SVM chiziqli yoki nochiziqli (ya'ni nochiziqli yadroga — kernel'ga ega) bo'lishi mumkin. Chiziqli holatda SVM klasslarni giperteklislik (hyperplane) bilan ajratishga harakat qiladi, lekin bu giperteklislik biroz boshqacha usulda — **marginlar** yordamida quriladi. Nochiziqli holatda esa giperteklislik qurilishidan oldin belgilar fazosi **kernel**lar yordamida transformatsiya qilinadi, shunda klasslar chiziqli ajratiladigan bo'lib qoladi. SVM optimallashtirish — bu cheklovlar bilan optimallashtirish masalasi. Chiziqli SVM nochiziqli bog'liqliklar bilan qiyinchilik chekadi, nochiziqli SVM'lar esa neyron tarmoqlarning xususiy holati sifatida ko'rilishi mumkin.

#### Metrikalar

Klassifikatsiya masalalarida eng ko'p ishlatiladigan yo'qotish (loss) funksiyasi — **logloss**. Lekin bu funksiya biznes ilovalari uchun eng yaxshi sifat metrikasi emas.

Keng qo'llaniladigan metrikalar:
1. Precision/Recall;
2. F1 score;
3. Fbeta score;
4. AUC PR;
5. AUC ROC.

AUC metrikalari binar masalalar uchun hisoblanadi, Precision/Recall/F1/Fbeta esa multiklass uchun ham umumlashtirilishi mumkin.

##### Precision/Recall

Precision va Recall — **qattiq yorliqlar (hard labels)** uchun metrikalar, ya'ni ehtimollar emas, balki klass bashoratlari. Qattiq yorliqlar ehtimollarni chegara (threshold) bilan solishtirish orqali olinadi:

$$\mathbb{I}(p(y=1\mid x) > \text{threshold})$$

bu yerda $\mathbb{I}$ — indikator funksiyasi (ifoda to'g'ri bo'lsa 1, aks holda 0).

Qattiq yorliqlar yordamida **chalkashlik matritsasi (confusion matrix)** tuziladi:

$$Precision = \frac{tp}{tp+fp}, \qquad Recall = \frac{tp}{tp+fn}$$

Talqin:
- **Precision** — bashorat qilingan "1"lardan qanchasi haqiqatan ham "1" ekanligi;
- **Recall** — haqiqiy "1"lardan qanchasi model tomonidan topilgani.

Ba'zi vazifalarda Recall Precision'dan muhimroq:
- Tibbiyot (qo'shimcha tekshiruv arzon va xavfsiz bo'lgan hollarda);
- Firibgarlikni aniqlash (fraud detection);
- Mos kontent (kurslar, videolar) tavsiya qilish.

Boshqa hollarda Precision ustunroq:
- Tibbiyot (davolash/tekshiruv qimmat yoki zararli bo'lganda);
- VIP mijozlarga tavsiyalar (eng yaxshi mijozlarni keraksiz tavsiyalar bilan bezovta qilmaslik kerak);
- Aeroportlardagi xavfsizlik tizimlari.

##### F1 Score

Precision va Recall'ning garmonik o'rtachasi — **F1 Score**:

$$F1 = 2\cdot\frac{precision\cdot recall}{precision+recall}$$

##### Fbeta score

Recall yoki Precision'ga ko'proq e'tibor berish kerak bo'lsa, F1'ning umumlashtirilgan varianti — **Fbeta score** ishlatiladi:

$$F_\beta = (1+\beta^2)\cdot\frac{precision\cdot recall}{\beta^2\cdot precision + recall}$$

##### AUC PR

Fbeta score'ni optimallashtirishda ham optimal chegarani (threshold) topish kerak bo'ladi. **Average Precision (AP)** — barcha mumkin bo'lgan chegaralar uchun Precision va Recall'ni umumlashtirish usuli sifatida ko'rilishi mumkin.

##### AUC ROC

**AUC ROC** quyidagi hodisaning ehtimoli sifatida talqin qilinishi mumkin:

*[0-klassdan tasodifiy tanlangan namunaning bashorat qilingan ehtimoli < 1-klassdan tasodifiy tanlangan namunaning bashorat qilingan ehtimoli]*

Diqqat: AUC ROC metrikasining bazaviy qiymati (tasodifiy shovqin uchun) — **0.5**, 0 emas!

**Gini koeffitsienti** AUC ROC'ning "yo'nalishsiz" varianti sifatida qaralishi mumkin va quyidagi formula orqali hisoblanadi: **|2·ROC AUC − 1|**

## III bob. Maqsad

Ushbu vazifaning maqsadi — asosiy klassifikatsiya modellari (asosan Logistic Regression, NB va KNN) haqida chuqur tushuncha hosil qilish.

## IV bob. Ko'rsatmalar

**"School 21"da qanday o'qish kerak:**
- Bu yerda sizga katta erkinlik beriluvchi noyob ta'lim tajribasi kutmoqda. Sizga vazifa beriladi va uni yechish yo'lini o'zingiz — internet yoki AI vositalari (masalan, GigaChat) yordamida topasiz. Faqat ma'lumot sifatiga e'tibor bering: tekshiring, tanqidiy fikrlang, tahlil qiling va solishtiring.
- **P2P (peer-to-peer) o'qitish** — tengdoshlar bilan bilim va tajriba almashish, bunda har kim ham ustoz, ham talaba rolini o'ynaydi. Bu yondashuv materialni bir-biridan o'rganish orqali chuqurroq tushunishga yordam beradi.
- Yordam so'rashdan tortinmang: atrofingizda xuddi shu yo'lni birinchi marta bosib o'tayotgan tengdoshlaringiz bor. O'z tajribangiz va g'oyalaringizni ulashing. Jamoat e'lonlaridan xabardor bo'lish uchun Rocket.Chat'ga qo'shiling.
- Boshqalarning yechimini shunchaki nusxa ko'chirish — o'qishning ma'nosini yo'qotadi. Yordam olganda ham, yechimning "nega", "qanday" va "nima maqsadda" ekanini albatta to'liq tushunib oling. Xato qilishdan qo'rqmang.
- Vazifa imkonsizdek tuyulyaptimi? Tanaffus qiling, sof havodan nafas oling — bu ko'plarga yordam bergan. Balki shundan keyin yechim o'z-o'zidan kelib qoladi.
- O'qish jarayoni natija kabi muhim — bu shunchaki vazifani bajarish emas, balki uni QANDAY yechishni tushunish haqida.

**Loyiha bilan ishlash bo'yicha:**
- Bu loyiha faqat odamlar tomonidan baholanadi. Fayllaringizni istalgan tarzda tashkil qilish va nomlash mumkin.
- Bu yerda va bundan keyin faqat Python 3 qo'llaniladi (yagona to'g'ri versiya).
- Deep learning algoritmlarini o'qitish uchun [Google Colab](https://colab.research.google.com)dan foydalanish mumkin — u bepul GPU kernellarini taklif etadi, bu esa CPU'ga qaraganda tezroq.
- Bu loyihaga standart (code style) qo'llanilmaydi, lekin kodingizni tushunarli va tuzilgan qilib yozish so'raladi.
- Ma'lumotlar to'plamlarini `data` papkasida saqlang.

## V bob. Vazifa

Vazifalar ro'yxati (12 ta band) quyida "Loyiha yechimi" faylida har biri alohida tushuntirish va kod bilan bajarilgan:

1. [Don't Get Kicked](https://www.kaggle.com/c/DontGetKicked) Kaggle musobaqasidan ma'lumotlarni yuklab olish.
2. Train/validation/test bo'linishini loyihalash. `PurchDate` maydoni bo'yicha bo'lish kerak: test > validation > train (vaqt bo'yicha). Sanalarning birinchi 1/3 qismi — train, oxirgi 1/3 — test, o'rtadagi 1/3 — validation. *Test ma'lumotlaridan охиригача foydalanmang!*
3. Kategorial o'zgaruvchilarni sklearn'dagi `LabelEncoder` yoki `OneHotEncoder` bilan qayta ishlash. Ma'lumot sizib chiqishi (data leakage)ga ehtiyot bo'ling — Encoder'ni faqat train'ga moslashtiring (fit), so'ng validation va test'ga qo'llang (transform). Agar validation/test'da train'da ko'rilmagan yangi kategoriyalar uchrasa, boshqa kodlash yondashuvini (masalan, count encoding) ko'rib chiqing.
4. Train to'plamida sklearn'dan `LogisticRegression`, `GaussianNB`, `KNN` ni o'qitish va validation to'plamida sifatini tekshirish (`IsBadBuy` — binar bog'liq o'zgaruvchi). Modellarni o'qitishdan oldin ma'lumotlarni normallashtirishni unutmang. Kamida **0.15 Gini score** (uch algoritmning eng yaxshisi) olishingiz kerak. Qaysi algoritm yaxshiroq ishlaydi va nega?
5. Gini score hisoblashni implementatsiya qilish (2·ROC AUC − 1 yondashuvi orqali — buning uchun ROC AUC hisoblashni ham o'zingiz yozishingiz kerak). Natijangiz `abs(2*sklearn.metrics.roc_auc_score - 1)` ga taxminan teng ekanini tekshiring.
6. LogisticRegression, KNN va NaiveBayes'ning o'z versiyalarini implementatsiya qilish. LogisticRegression uchun yo'qotish funksiyasi gradientini hisoblab, stochastic gradient descent qo'llang. 4-band natijalarini takrorlay olasizmi? Model *fit*, *predict* (0.5 chegara bilan *predict_proba*), *predict_proba* metodlariga ega klass ko'rinishida bo'lishi kerak.
7. Nochiziqli belgilar yaratishga harakat qilish (masalan: belgi1/belgi2 nisbati, `groupby` orqali guruh statistikalari). Yangi belgilarni pipeline'ga qo'shib, 4-bandni takrorlang. Gini score'ni oshira oldingizmi (oshirishingiz kerak)?
8. Logistic model koeffitsientlari yordamida eng yaxshi belgilarni aniqlash. Keraksiz belgilarni qo'lda va L1 regularizatsiya orqali olib tashlashga harakat qiling. Qaysi yondashuv Gini score bo'yicha yaxshiroq?
9. Eng yaxshi model (algoritm + belgilar to'plami)ni tanlab, validation to'plamida Gini score'ni oshirish uchun giperparametrlarni sozlash. Qaysi giperparametrlar eng katta ta'sirga ega?
10. Eng yaxshi modelingiz uchun uchala to'plamda (train, valid, test) Gini score'larni tekshiring. Valid va test sifati orasida pasayish bormi? Modelingiz overfit bo'lganmi yoki yo'qmi? Tushuntiring.
11. Recall, Precision, F1 score va AUC PR metrikalarini implementatsiya qilish. Algoritmlaringizni test to'plamida AUC PR metrikasi bo'yicha solishtiring.
12. "Lemon" (nuqsonli) mashinalarni aniqlash vazifasi uchun qaysi qattiq yorliq (hard label) metrikasini afzal ko'rasiz?
