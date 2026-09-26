# Supervised Learning. Decision Trees and Ensembles — O'zbekcha tarjima

**Qisqacha mazmun:** Bu loyiha decision tree'lar (CART) va ensemble metodlar (Random Forest, GBDT) bilan tanishtiradi.

## Mundarija

- I bob. Kirish so'zi
- II bob. Umumiy tushuncha
  - Decision Tree
  - CART ning afzalliklari
  - CART ning kamchiliklari
  - Ensemble metodlar
  - Gradient Boosting
  - Zamonaviy Gradient Boosting implementatsiyalari
  - Manbalar
- III bob. Maqsad
- IV bob. Ko'rsatmalar
- V bob. Vazifa
- VI bob. Bonus qism

## I bob. Kirish so'zi

Biz allaqachon classification/regression masalalari uchun bir nechta algoritmlarni ko'rib chiqdik. Ammo tabular (jadval ko'rinishidagi) ma'lumotlar uchun eng kuchli ML algoritmlari **decision tree**'larga (yoki **classification and regression trees, CART**) asoslangan. CART modellari tushunish oson, yuqori darajada izohlanadigan (explainable) modellar bo'lib, inson qaror qabul qilish jarayonini taqlid qiladi (yoki avtomatlashtiradi). CART'lar ko'pchilik hollarda yaxshi aniqlik (accuracy) bermaydi, lekin **ensembling** texnikalari bizga o'qitilgan CART modellari to'plami asosida mustahkam va aniq modellar qurish imkonini beradi.

## II bob. Umumiy tushuncha

Classification va regression tree'lar kirish fazosini (feature space) rekursiv ravishda bo'laklarga bo'lish va har bir bo'lakda (region) lokal model aniqlash orqali ishlaydi. Tree'ning har bir tugunida (node) keyingi bo'lish (partitioning) taklifini topishga harakat qilamiz: qanday bo'lish keyingi tugunlardagi entropiyani yoki og'ishni kamaytiradi? Umumiy model (binary) tree ko'rinishida ifodalanadi, har bir region uchun bitta leaf (barg) mavjud.

### Decision Tree

Quyidagi rasm (hayvonlarni klassifikatsiya qilish) decision tree qanday ishlashini umumiy ko'rsatib beradi: biz kirish ma'lumotini rekursiv ravishda sub-fazolarga bo'lamiz; har bir bo'lish bizni to'g'ri javobga bir qadam yaqinlashtiradi. 0-darajada (ildiz tugun, root node) biz hayvon turini bilmaymiz, 1-darajada sariq bo'lmagan hayvonlar o'ng shoxga o'tadi, endi hayvonlarni yanada ko'proq ajratishimiz mumkin (bo'yi, vazniga qarab va h.k.).

*Manba: [x] https://www.simplilearn.com/tutorials/machine-learning-tutorial/decision-tree-in-python*

Fil (elephant) kirish sifatida berilganda decision tree quyidagicha ishlaydi:
1. Fil sariq emas, shuning uchun o'ng shoxga o'tadi.
2. Fil juda katta, shuning uchun yana o'ng shoxga o'tadi.

Keyingi rasm klassik Iris dataset uchun boshqa decision tree misolini ko'rsatadi (manba: Murphy). (a) rasm Iris klassifikatsiya masalasi uchun decision tree misolini, (b) rasm esa (a) tree tomonidan keltirib chiqarilgan qaror qoidasini (decision rule) ko'rsatadi. CART kirish fazosini o'qqa parallel (axis-parallel) kesimlar yordamida bo'lganini ko'ramiz. Har bir region doimiy qiymatli bashoratga (constant value prediction) mos keladi: masalan, to'q sariq (orange) regiondagi har bir input instansiya "setosa" deb belgilanadi. CART'ni yuqori darajada nolinear indikator feature'lar (har bir partition — bu solishtirishlar kombinatsiyasi) ustidagi lineer model deb qarash mumkin.

Tree'ning diskret strukturasi bizning masalamiz uchun differensiallanmaydigan (nondifferentiable) loss funksiyasiga olib keladi. Ma'lumotlarning optimal bo'linishini topish NP-complete masaladir. Standart amaliyot — **greedy** protsedura ishlatish: biz tree'ni bir vaqtning o'zida bitta tugun qo'shib, har bir qadamda eng yaxshi mumkin bo'lgan bo'lishni tanlab, iterativ ravishda o'stiramiz.

O'qitish protsedurasi quyidagi qadamlardan iborat:
1. Har bir qadamda, har bir mumkin bo'lgan split (bo'linish) uchun entropiyaning kamayishini (classification masalasi uchun) yoki standart og'ishning kamayishini (regression masalasi uchun) hisoblaymiz. Biz yangi 2 ta tugunga tushadigan inputlar target o'zgaruvchi bo'yicha bir xilroq (homogeneous) bo'lishini xohlaymiz.
   Masalan: vazifa hayvonni 3 ta feature (rang, vazn, bo'y) asosida fil yoki fil-emas deb klassifikatsiya qilish bo'lsin. Mumkin bo'lgan split'lar: har qanday rang, barcha vaznlar va h.k. Rang bo'yicha bo'lish informativ (fillar kulrang, tulkilar esa yo'q), lekin eng yaxshisi emas: kulrang rangli juda ko'p turli xil turlar mavjud; chap subtree'da (kulrang bo'lmagan hayvonlar) fillar bo'lmaydi, lekin o'ng subtree'da (kulrang hayvonlar) faqat fillar emas, balki boshqa ko'plab turlar ham bo'ladi.
   Bu vazifada **[vazn > 1000 kg]** eng yaxshi mumkin bo'lgan split hisoblanadi: fillar quruqlikdagi eng og'ir hayvonlardir. Bu split 2 ta yangi tugunda noaniqlikning maksimal kamayishini beradi: **[vazn >= 1000 kg]** faqat fillarni o'z ichiga oladi, **[vazn < 1000 kg]** esa umuman fil bo'lmaydi (yoki juda kichik, yengil fillar bo'lishi mumkin).
2. Ikkita yangi tugun kiritilgandan so'ng, ularda split qilish jarayonini rekursiv ravishda boshlaymiz.
3. Ba'zi to'xtash mezonlari (stopping criteria) bajarilgach, bo'lishni to'xtatamiz (masalan, oldindan belgilangan maksimal tree chuqurligi, keyingi split'larda noaniqlik kamayishi juda past bo'lishi va h.k.).

### CART ning afzalliklari

Decision tree'larning kuchli tomonlari:
* CART modellari izohlanadigan (interpretable) model kerak bo'lganda yaxshi tanlov: tree modelining bashoratini tushuntirish — tree'ning split'lari bo'ylab yurishni anglatadi.
* CART aralash taqsimotli (mixed distribution) datasetlar bilan ishlay oladi. CART'lar monoton transformatsiyalarga deyarli sezgir emas. CART ishlatilganda normalization, scaling, binning kabi preprocessing texnikalari odatda qo'llanilmaydi.
* Feature'lardagi outlier'lar ham CART ishlashiga kuchli ta'sir qilmaydi.

### CART ning kamchiliklari

Real hayotdagi masalalarda CART kamdan-kam holda alohida (stand-alone) model sifatida ishlatiladi, chunki uning jiddiy kamchiliklari bor:
* CART modellari tree'ning greedy o'qitish jarayoni tufayli overfitting'ga moyil.
* Tree'lar beqaror (unstable): kirish ma'lumotidagi kichik o'zgarishlar tree strukturasiga katta ta'sir ko'rsatishi mumkin, chunki tree o'sish jarayoni ierarxik (hierarchical) xarakterga ega — tepadagi xatolar tree'ning qolgan qismiga ta'sir qiladi [Murphy].

### Ensemble metodlar

CART modellarini overfit qilish oson. Ammo CART'lar boshqa murakkabroq va aniqroq tree-asosidagi modellar uchun qurilish bloki (building block) sifatida faol ishlatiladi. CART'ning beqarorligini bir xil dataset ustida o'qitilgan ko'plab turli CART modellarini o'rtachalash (averaging) orqali yumshatish mumkin.

**Random Forest** modeli bu g'oyaning misolidir:
1. Trening ma'lumotining tasodifiy subsample'laridan foydalanib bir talay (odatda 100-1000 ta) CART modellarini yaratamiz. Odatda datasetdan tasodifiy qatorlarning bir qismini (masalan, ⅔) va M ta tasodifiy ustunlarni subsample qilamiz hamda juda chuqur tree'lardan foydalanamiz. Natijada potentsial overfitting qilgan va past aniqlikka ega o'qitilgan tree'lar to'plamiga ("**bag**") ega bo'lamiz.
2. Random forest modelining natijasi — *har bir o'qitilgan tree'ning o'rtacha bashorati*. Har bir tree trening ma'lumotining qandaydir qismiga overfit qilingan, lekin ularning o'rtachasi ancha barqaror va aniq bashorat beradi.

Turli tasodifiy tanlangan ma'lumot versiyalariga turli base modellarni moslashtirish **bagging** deb ataladi. Bagging bir xil o'qitish algoritmini ma'lumotning turli subsetlarida qayta ishga tushirish yetarlicha farqli underlying modellarga olib keladi degan taxminga asoslanadi. Random Forest modeli bagging ishlatilishining misolidir.

Har bir tree kirish fazosini box'larga bo'ladi. Tree'lar to'plamini o'rtachalash silliqroq qaror chegarasiga (decision boundary) olib keladi. Bu tamoyilni quyidagi rasm illyustratsiya qiladi: (a) rasm bitta tree uchun qaror chegarasini ko'rsatadi, (b) 10 ta tree'dan iborat bag'ni o'rtachalashga mos keladi, (c) 50 ta tree'li Random Forest modeliga mos keladi. E'tibor bering, (c) da (a) dagi chegara artifaktlari yo'q.

**ExtraTrees** — boshqa turdagi tree-asosidagi model [ET paper]. Bu asosan qatorlar bo'yicha subsampling'siz Random Forest'dir (faqat feature'lar subsample qilinadi), CART o'rniga **extra randomized tree**'lardan foydalaniladi.
Extra randomized tree — bu har bir feature uchun tasodifiy split'lar tanlanadigan va barcha tanlangan nomzodlar orasidan eng yaxshi split qidiriladigan CART modeli. Amaliyotda ExtraTrees modeli RandomForest'ga qaraganda kamroq aniq, lekin overfitting'ga kamroq moyil.

Afsuski, ko'plab tree'larga asoslangan metodlar o'zining izohlanuvchanlik (interpretability) xususiyatlarini yo'qotadi: biz yuqori aniqlikni aniqroq bo'lmagan qaror jarayoniga almashtiramiz (trade-off). Black-box algoritmlarni tushuntirish uchun bir nechta yondashuvlar mavjud: SHAP, LIME va h.k.

### Gradient Boosting

O'qitilgan decision tree classifier'ni ko'rib chiqamiz. Birinchi tree tomonidan kiritilgan xatolarni tuzatishga harakat qiluvchi yana bir decision tree qo'shish orqali modelimiz sifatini yaxshilashimiz mumkin. Keyin bu jarayonni ensemble aniqligini oshiruvchi yana bir tree'ni o'qitish orqali davom ettirishimiz mumkin. Natija **additive model** deb ataladi.

*(Manba: [Murphy])* Rasm 3 ta o'qitilgan CART'dan iborat additive modelni ko'rsatadi. 2-tree 1-tree'dagi xatoni kamaytirishga harakat qiladi. 3-tree 1-tree va 2-tree ensembli tomonidan keltirib chiqarilgan xatolarni kamaytiradi. Bunday additive modelni qanday o'rganamiz?

Oldingi darslardan Stochastic Gradient Descent haqida bilamiz. Parametrlarni loss funksiyasining antigradienti tomon qadam bosish orqali yangilash ML algoritmining xatolarini kichraytiradi.

Xuddi shu tamoyilni additive modellashtirishga ham qo'llash mumkin:
* Differensiallanuvchi loss funksiyasi uchun joriy qadamdagi xatoning antigradientini hisoblang va antigradientni approksimatsiya qiluvchi yangi decision tree'ni o'qiting.
  Bu **x** input va **A(x)** — instansiya **x** uchun joriy ensemblening bashorati bo'lgan holda, A bo'yicha **Loss(y, A(x))** ning manfiy gradienti target bo'lgan supervised masalaga ekvivalentdir.
* Yangi tree'ning natijasini (kichik koeffitsient bilan) joriy ensembldagi modellar natijasiga qo'shing.
* Ensemblega yangi tree qo'shish orqali loss'imizni kamaytirdik. Ba'zi to'xtash mezonlariga (maksimal iteratsiyalar soni, juda kichik yaxshilanishlar va h.k.) yetgunimizcha oldingi qadamlarni takrorlashimiz mumkin.

GBDT o'qitish protsedurasi [4] manbada aniq tushuntirilgan.

Antigradientni approksimatsiya qilish regression tree'lardan foydalanishni talab qiladi. Natijaviy model **Gradient Boosting Decision Trees** deb ataladi. Umuman olganda, base learner sifatida istalgan turdagi tree'lar (yoki boshqa ML modellari) ishlatilishi mumkin, lekin tree'lar eng qulayi hisoblanadi. Amaliyotda GBDT'ning turli implementatsiyalari turli xil base tree turlaridan foydalanadi.

GBDT modelining yakuniy shakli RandomForest'ni eslatadi: bu ko'plab tree'larning lineer kombinatsiyasi. Ammo GBDT o'qitish sxemasida har bir tree oldingi tree'lar natijasi asosida quriladi, Random Forest'da esa alohida tree'lar to'g'ridan-to'g'ri bog'liq emas: biz turli CART'larni o'qitamiz, so'ngra o'rtachalaymiz. Odatda GBDT Random Forest'dan ba'zi hiylalarni ishlatadi: trening instansiyalari va feature'larni subsampling qilish.

### Zamonaviy Gradient Boosting implementatsiyalari

Zamonaviy GBDT paketlari foydalanuvchiga ensembldagi har bir tree'ning o'qitish jarayonini boshqaradigan o'nlab turli parametrlarni taqdim etadi. GBDT'ning bir nechta samarali implementatsiyalari mavjud: bugungi kunda eng mashhur 3 ta GBDT implementatsiyasi — LightGBM, Catboost va XGBoost. Ularning barchasida yuzlab sozlanadigan parametrlar, turli tree strukturalari, o'qitish rejimlari va h.k. mavjud.

Farqlarga qisqacha umumiy nazar:
* Uchalasi ham turli tree yoki o'qitish strategiyalaridan foydalanadi: depthwise (XGBoost), leafwise (LightGBM), oblivious trees (Catboost).
* LightGBM kategorik ma'lumotlar uchun ==/!= split'larni qo'llab-quvvatlaydi.
* Catboost kategorik feature'lar bilan ishlashga ixtisoslashgan: smart target encoding amalga oshirilgan.
* LightGBM va XGBoost'da DART rejimi bor: allaqachon o'rganilgan tree'larning subsample'ini tashlab yuborish (dropping) bilan o'qitish sxemasi.
* LightGBM ExtraTrees va Linear Tree'larni qo'llab-quvvatlaydi (biz leaf'dagi o'rtacha target bashoratini LinearRegression bashorati bilan almashtiramiz).

[1], [2], [3] maqolalari eng yaxshi GBDT kutubxonalari haqida yaxshi umumiy ko'rinish beradi.

GBDT algoritmlari aralash taqsimotli inputlardan juda murakkab funksiyalarni o'rganish qobiliyatiga ega. GBDT algoritmlari aralash taqsimotli tabular ma'lumotlar uchun (o'rtacha hisobda) eng yaxshi tanlov hisoblanadi. Additive tabiat va ko'plab tree'lardan foydalanish GBDT modellarini aniq izohlashni juda murakkablashtiradi.

### Manbalar

[1] https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

[2] https://medium.com/riskified-technology/xgboost-lightgbm-or-catboost-which-boosting-algorithm-should-i-use-e7fda7bb36bc

[3] https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm

[4] https://alexanderdyakonov.files.wordpress.com/2017/06/book_boosting_pdf.pdf

[5] https://link.springer.com/content/pdf/10.1007/s10994-006-6226-1.pdf?pdf=button

[6] https://www.simplilearn.com/tutorials/machine-learning-tutorial/decision-tree-in-python

## III bob. Maqsad

Bu topshiriqning maqsadi — classification uchun tree-asosidagi modellarni (CART, Random Forest, GBDT) va zamonaviy implementatsiyalarni (LightGBM, XGBoost, Catboost) chuqur tushunishga erishishdir.

## IV bob. Ko'rsatmalar

"School 21"da qanday o'qish kerak:

- Bu yerda siz katta erkinlikka ega noyob ta'lim tajribasini topasiz. Sizga vazifa beriladi va uni o'zingizga qulay bo'lgan har qanday resurslardan foydalanib (Internet yoki GigaChat kabi AI vositalari) o'zingiz yechish yo'lini topishingiz kerak bo'ladi. Faqat ma'lumot sifatiga e'tiborli bo'ling: tekshiring, tanqidiy fikrlang, tahlil qiling, solishtiring.
- Peer-to-peer (P2P) o'qitish — bilim va tajriba almashinuvi bo'lib, unda har kim ham mentor, ham talaba rolini o'ynaydi. Bu yondashuv bir-biringizdan o'rganish orqali materialni chuqurroq tushunish imkonini beradi.
- Yordam so'rashdan tortinmang: atrofingizda bu yo'lni birinchi marta bosib o'tayotgan hamkasblaringiz bor. O'z tajriba va g'oyalaringizni boshqalar bilan bo'lishing. Jamoat e'lonlaridan xabardor bo'lish uchun Rocket.Chat'ga qo'shiling.
- Agar shunchaki boshqa birovning yechimini ko'chirsangiz, o'qishingiz ma'nosiz bo'ladi. Boshqalardan yordam olayotganda, har doim yechim ortidagi "nega", "qanday" va "maqsad"ni to'liq tushunganingizga ishonch hosil qiling. Xato qilishdan qo'rqmang.
- Vazifa imkonsizdek tuyulyaptimi? Tanaffus qiling, toza havoda yuring va ongingizni tozalang — bu ko'pchilikka yordam bergan. Balki shundan keyin yechim o'z-o'zidan xayolingizga keladi.
- O'qish jarayoni natija kabi muhim. Bu faqat vazifani bajarish emas — bu qanday yechishni TUSHUNISH haqida.

Loyiha bilan qanday ishlash kerak:

* Bu loyiha faqat odamlar tomonidan baholanadi. Fayllaringizni xohlagancha tashkil qilishingiz va nomlashingiz mumkin.
* Biz Python 3'ni yagona to'g'ri Python versiyasi sifatida ishlatamiz.
* Deep learning algoritmlarini o'qitish uchun [Google Colab](https://colab.research.google.com)'dan foydalanib ko'rishingiz mumkin. U bepul GPU'li kernel'larni (Runtime) taklif etadi, bu esa bunday vazifalar uchun CPU'dan tezroq.
* Bu loyihaga standart (Norm/coding style) qo'llanilmaydi. Ammo source kodingizni aniq va tuzilgan (structured) qilib dizayn qilishingiz so'raladi.
* Datasetlarni `data` sub-papkasida saqlang.

## V bob. Vazifa

1\. [Don'tGetKicked musobaqasi](https://www.kaggle.com/c/DontGetKicked)'dan ma'lumotlarni yuklab oling. Train/validation/test split'ni loyihalang (design qiling).

Bo'lish uchun "PurchDate" maydonidan foydalaning, test vaqt bo'yicha validation'dan keyin bo'lishi kerak, xuddi shunday validation ham train'dan keyin: train.PurchDate < valid.PurchDate < test.PurchDate.

Ma'lumotning birinchi 33%'ini training uchun, oxirgi 33%'ini test uchun, o'rtadagi 33%'ini esa validation set uchun ishlating. *Test datasetini oxirigacha ishlatmang!*

Kategorik o'zgaruvchilarni preprocessing qilish uchun sklearn'dan LabelEncoder yoki OneHotEncoder'dan foydalaning. Data leakage'ga ehtiyot bo'ling (Encoder'ni training'ga fit qiling va validation & test'ga qo'llang). Agar validation & test'da yangi kategorik qiymatlarga duch kelsangiz (training'da ko'rilmagan), boshqa encoding yondashuvini ko'rib chiqing, masalan: https://contrib.scikit-learn.org/category_encoders/count.html

2\. Decision Tree Classifier va Decision Tree Regressor (MSE loss) uchun Python class yarating.

U *fit*, *predict_proba* va *predict* metodlarini qo'llab-quvvatlashi kerak. Shuningdek, maksimal chuqurlik (max_depth) sizning class'ingizning parametri bo'lishi kerak. Split tanlash mezoni sifatida Gini impurity criterion'dan foydalaning.

Blueprint quyidagicha:

```python
model = DecisionTreeClassifier(max_depth=7)
model.fit(Xtrain, ytrain)
model.predict_proba(Xvalid)
```
* Node uchun alohida class yarating. U ma'lumotni (sample feature'lar va target'lar) saqlashi, Gini impurity'ni hisoblashi va children'larga (left va right node) pointer'larga ega bo'lishi kerak. Regressor uchun Gini impurity o'rniga standart og'ishdan (standard deviation) foydalaning.
* Joriy tugunda eng yaxshi mumkin bo'lgan split'ni topadigan funksiya implement qiling.
* Oldingi qadamlarni ishlaydigan Decision Tree Classifier'ingizga birlashtiring.
* Boshqa best split topish funksiyasini loyihalash orqali Extra Randomized Tree'ni implement qiling.

3\. DecisionTree modulingiz bilan validation datasetda kamida 0.1 Gini score'ga erishishingiz kerak.

4\. sklearn'ning DecisionTreeClassifier'idan foydalaning va uning validation datasetdagi natijasini tekshiring. U sizning moduldan yaxshiroqmi? Agar shunday bo'lsa, nega?

5\. RandomForestClassifier'ni implement qiling va uning natijasini tekshiring. Bitta tree natijasini yaxshilashingiz va validation datasetda kamida 0.15 Gini score olishingiz kerak. Fixed random seed o'rnatish imkoniyatiga ega bo'ling.

6\. GBDT classifier uchun DecisionTree dizayn class'ingizdan foydalaning. Bu class *max_depth*, *number_of_trees* va *max_features* atributlariga ega bo'lishi kerak. Binary cross-entropy loss funksiyasining gradientini hisoblashingiz va incremental learning'ni implement qilishingiz kerak: keyingi tree'ni oldingi tree'lar natijasidan foydalanib o'qiting.

7\. Training set'ga fit qilish va validation set'da bashorat qilish uchun LightGBM, Catboost va XGBoost'dan foydalaning. Kutubxonalar dokumentatsiyasini ko'rib chiqing va vazifa uchun algoritmlarni fine-tune qiling.
Har bir implementatsiyaning asosiy farqlarini qayd eting. Har bir algoritmning maxsus xususiyatlarini tahlil qiling (Catboost'da "categorical feature" qanday ishlaydi, XGBoost'da DART rejimi nima?)
Qaysi GBDT modeli eng yaxshi natijani beradi? Nega ekanligini tushuntira olasizmi?

8\. Eng yaxshi modelni oling va uning natijasini test datasetda baholang: eng yaxshi modelingiz uchun barcha uchta datasetdagi Gini qiymatlarini tekshiring: training Gini, valid Gini, test Gini. Valid sifatini test sifati bilan solishtirganda natijada pasayishni ko'ryapsizmi? Modelingiz overfit qilyaptimi yoki yo'qmi? Tushuntiring.

9*. ExtraTreesClassifier'ni implement qiling va uning natijasini tekshiring. Bitta tree natijasini yaxshilashingiz va validation datasetda kamida 0.12 Gini score olishingiz kerak.

## VI bob. Bonus qism

[V bob. Vazifa](#v-bob-vazifa) bo'limida * belgisi bilan qayd etilgan qism bonus hisoblanadi.

> Iltimos, loyiha haqida fikr-mulohazangizni [feedback shakli](https://forms.yandex.ru/cloud/646b46eff47e732ee1311d6f/) orqali qoldiring.
