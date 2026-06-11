# Recunoașterea acțiunilor umane (HAR) pe KTH cu rețele convoluționale 3D

https://cnn-har-munteanu-mihai.streamlit.app/

> Acest document descrie complet pipeline-ul CNN clasic implementat în folder-ul `cnn_har_app/`. Scopul lui este să servească drept contrapartidă (baseline CNN) a rețelei spiking convoluționale (CSNN) descrise în capitolele anterioare ale lucrării de licență, permițând astfel o comparație directă, în condiții echivalente (același set de date, aceeași segmentare temporală, aceiași descriptori HOG, **același split standard pe subiecți, identic cu cel folosit la CSNN**).
>
> Pipeline-ul este organizat astfel încât majoritatea componentelor scumpe (detecție de persoană, extragere HOG, augmentări spațiale și fotometrice) se precalculează *o singură dată* într-un fișier `.npz`. Antrenarea efectivă a CNN-ului citește acest tensor și se concentrează pe învățarea reprezentărilor spațio-temporale.
>
> **Configurația curentă a lucrării** (aliniată cu datasetul CSNN): **T = 19 frame-uri** per eșantion, **frame_gap = 2**, **num_groups = 10** grupuri/video, split **train/val/test pe subiecți (8/8/9)**, descriptor HOG pe fereastră **64 × 128** (3780 features/frame). *(Numele fișierului `..._f10_g2` e doar o etichetă; valoarea reală e frame_gap=2.)*

---

## 1. Cuprins

1. [Privire de ansamblu asupra pipeline-ului](#2-privire-de-ansamblu)
2. [Setul de date și split-ul standard KTH](#3-setul-de-date-kth)
3. [Etapa 1 — Bounding box-uri (`hog_person_data*.json`)](#4-etapa-1--detecția-persoanei)
4. [Etapa 2 — Pre-calculul HOG augmentat (`extract_hog_augmented.py`)](#5-etapa-2--extract_hog_augmentedpy)
5. [Etapa 3 — Dataset PyTorch și fluxuri suplimentare (`dataset.py`)](#6-etapa-3--datasetpy)
6. [Etapa 4 — Arhitectura modelului (`model.py`)](#7-etapa-4--modelpy)
7. [Etapa 5 — Antrenarea (`train.py`)](#8-etapa-5--trainpy)
8. [Etapa 6 — Evaluare în ansamblu (`eval_ensemble.py`)](#9-etapa-6--eval_ensemblepy)
9. [Etapa 7 — Demonstrator Streamlit (`app.py`)](#10-etapa-7--apppy-streamlit)
10. [Rezultate experimentale](#11-rezultate-experimentale)
11. [Comparație CNN ↔ CSNN](#12-comparație-cnn--csnn)
12. [Direcții viitoare](#13-direcții-viitoare)
13. [Sugestii de figuri pentru lucrare](#14-sugestii-de-figuri)

---

## 2. Privire de ansamblu

Pipeline-ul CNN urmează strict aceeași logică de pre-procesare ca varianta CSNN, pentru ca singura sursă de diferență între cele două sisteme să fie *clasificatorul*. Diferența esențială apare doar la pasul final: CSNN folosește neuroni cu spike și plasticitate locală (STDP), iar CNN folosește convoluții 3D dense antrenate prin backpropagation.

```
       ┌──────────────────────┐
       │  Video KTH (.avi)    │
       └─────────┬────────────┘
                 │ (1) extract_bboxes_kth.py
                 ▼
       ┌──────────────────────┐
       │  bbox JSON           │   ← grupuri de T=19 frame-uri
       │  (frame_idx + bbox)  │     cu frame_gap = 2, 10 grupuri/video
       │  split în cheie:     │     train/ , val/ , test/
       └─────────┬────────────┘
                 │ (2) extract_hog_augmented.py
                 ▼
       ┌──────────────────────┐
       │  hog_aug_*.npz       │   ← features HOG augmentate
       │  (N, T·3780)=        │     + bboxes + labels + metadata
       │  (14715, 71820)      │
       └─────────┬────────────┘
                 │ (3) HOGDataset (dataset.py)
                 ▼
       ┌──────────────────────┐
       │  Tensor (T,C,H,W)    │   ← C = 36 HOG + 36 diff
       │  = (19, 80, 15, 7)   │       + bbox(4) + bbox_vel(4) = 80
       └─────────┬────────────┘
                 │ (4) HARConv3DNet (model.py)
                 ▼
       ┌──────────────────────┐
       │  Logits (6 clase)    │
       └─────────┬────────────┘
                 │ (5) AdamW + CosineLR + EMA + Mixup
                 │     selecție best pe VAL (early stopping)
                 ▼
       ┌──────────────────────┐
       │  Ansamblu de 5 seed-uri  │
       │  + TTA (reverse, shift)  │
       └──────────────────────┘
```

> **Figura sugerată #1**: schema de mai sus desenată ca pipeline pe orizontală cu icoane (cameră, JSON, tensor, rețea, evaluare).

---

## 3. Setul de date KTH

KTH conține 6 acțiuni × 25 subiecți × 4 scenarii = **600 clipuri** (rezoluție 160×120 px, ~25 fps, fundal static, grayscale).

| Index | Acțiune        | Caracter dominant            |
|------:|----------------|------------------------------|
| 0     | boxing         | mișcare locală în zona toracelui |
| 1     | handclapping   | mâini se apropie ritmic      |
| 2     | handwaving     | mâini sus-jos, amplitudine mare |
| 3     | jogging        | translație orizontală medie  |
| 4     | running        | translație orizontală rapidă |
| 5     | walking        | translație orizontală lentă  |

**Split pe subiecți train/val/test (8/8/9).** Spre deosebire de protocolul clasic train/test 1–16 / 17–25, aici se folosește un split în **trei** părți, identic cu cel din pipeline-ul CSNN, pentru ca:

- **train** să antreneze modelul,
- **val** să selecteze modelul cel mai bun (early stopping) — set *held-out*, niciodată folosit la antrenare,
- **test** să dea cifra finală de generalizare, atinsă o singură dată la final.

| Split | Subiecți                          | Nr. subiecți | Videoclipuri cu grupuri valide* |
|-------|-----------------------------------|:------------:|:-------------------------------:|
| Train | 11, 12, 13, 14, 15, 16, 17, 18    | 8            | 154                             |
| Val   | 19, 20, 21, 23, 24, 25, 1, 4      | 8            | 161                             |
| Test  | 2, 3, 5, 6, 7, 8, 9, 10, 22       | 9            | 190                             |

> *Numărul efectiv de videoclipuri care produc cel puțin un grup valid de T=19 frame-uri (detecția persoanei reușește), din extracția curentă (cu running re-extras, vezi 11.4). Total: 505 din 600 clipuri.

> În cod, aceste mulțimi sunt sursa de adevăr `TRAIN_SUBJECTS` / `VAL_SUBJECTS` / `TEST_SUBJECTS` din `dataset.py`. Split-ul este însă, în mod normal, citit **direct din prefixul cheii** din JSON (`train/`, `val/`, `test/`) — `split_from_video_key()` — clasificarea pe subiect fiind doar un fallback pentru JSON-uri vechi fără prefix.

---

## 4. Etapa 1 — Detecția persoanei

Scriptul `src/tool/extract_bboxes_kth.py` (folosit și de pipeline-ul CSNN, deci păstrat neschimbat aici) parcurge fiecare videoclip și produce un fișier `hog_person_data_*.json`. Detecția persoanei combină **HOG + MOG2** (fundal/mișcare) cu fuziune și netezire temporală a bbox-ului. Structura JSON:

```json
{
  "config": {
    "temporal_kernel": 19,
    "num_groups": 10,
    "frame_gap": 2,
    "frame_width": 160,
    "frame_height": 120
  },
  "videos": {
    "train/boxing/person11_boxing_d1_uncomp.avi": {
      "groups": [
        [
          { "frame_idx": 50,
            "selected_bbox": {"x": 68, "y": 54, "w": 30, "h": 61, ...}
          },
          ...  // T = 19 frame-uri
        ],
        ...  // până la num_groups grupuri per video
      ]
    }
  }
}
```

Split-ul (`train`/`val`/`test`) este codat ca **primul component al căii** din cheia videoclipului, scris de extractor pe baza structurii folderului `kth_organized_tvt/{train,val,test}/<acțiune>/<video>.avi`. Astfel pașii din aval rutează eșantioanele fără să mai facă lookup pe id-ul de subiect.

Cele două hiperparametre cheie reluate în întreg pipeline-ul CNN sunt:

- **`temporal_kernel` = T** — numărul de frame-uri din care e compus un eșantion (în configurația curentă **T = 19**);
- **`frame_gap` = g** — distanța dintre frame-urile consecutive din același grup. Frame-urile efective ale unui grup sunt $\{f_0, f_0 + g, f_0 + 2g, \dots, f_0 + (T-1)g\}$.

Frame gap-ul controlează cât *temporal context* captează un eșantion. Cu **T = 19, g = 2**, un eșantion acoperă $(T-1)\cdot g = 36$ frame-uri ≈ **1.4 s** la 25 fps — un sub-ciclu / un ciclu scurt de mișcare. Din fiecare video se extrag până la **num_groups = 10** astfel de ferestre (centrate pe detecțiile cu confidență mare).

| T  | `frame_gap` | Întindere (frame-uri) | Durata ≈ | Acoperă |
|---:|------------:|----------------------:|---------:|---------|
| **19** | **2**   | **36**                | **1.4 s**| **config curentă (aliniată CSNN), un sub-ciclu de mișcare** |
| 7  | 4           | 24                    | 0.96 s   | (config CNN veche) aproape un ciclu |

> **Figura sugerată #2**: o secvență de T = 19 frame-uri (frame_gap = 2) cu bbox-ul desenat în verde.

---

## 5. Etapa 2 — `extract_hog_augmented.py`

**Aceasta este componenta centrală a pre-procesării și merită explicată în detaliu**, pentru că de aici provine fișierul `.npz` care domină timpul total de antrenare și care decuplează partea „grea" (decodare video + augmentări spațiale + HOG) de partea „ușoară" (forward/backward prin CNN).

### 5.1. Motivația deciziei de pre-calcul

În implementările naive, HOG-ul se recalculează la fiecare epoch din videoclipurile sursă. Pe sute de epoch-uri, asta înseamnă sute de treceri prin decoderul codec + redimensionări + execuții ale `cv2.HOGDescriptor.compute()`, care domină timpul total. Soluția adoptată:

1. **O singură trecere** prin fiecare video pentru a obține toate feature-urile HOG.
2. **Augmentările spațiale și fotometrice** (flip, jitter de bbox, brightness, gamma, blur, zgomot) se aplică *înainte* de HOG, deci sunt deja încorporate în vectorii de feature.
3. **Augmentările feature-level** (Gaussian noise, dropout pe feature, shift/reverse temporal) se aplică *online* în `Dataset.__getitem__` — sunt suficient de ieftine.

Acest design transformă un experiment de ~ore de antrenare per seed într-unul de ~minute (după ce `.npz`-ul există).

### 5.2. Descriptorul HOG (Histogram of Oriented Gradients)

Pentru fiecare crop de persoană, descriptorul HOG operează pe o fereastră de **64 × 128 px** cu parametrii Dalal–Triggs:

| Parametru        | Valoare    | Comentariu                              |
|------------------|-----------:|-----------------------------------------|
| `winSize`        | 64 × 128   | Fereastră fixă, crop-ul e redimensionat |
| `blockSize`      | 16 × 16    | Bloc compus din 2 × 2 celule            |
| `blockStride`    | 8 × 8      | Suprapunere de 50%                      |
| `cellSize`       | 8 × 8      | Celulă elementară                       |
| `nbins`          | 9          | Histograme cu 9 orientări [0°, 180°)    |
| `nblocks_x`      | 7          | $(64-16)/8 + 1$                         |
| `nblocks_y`      | 15         | $(128-16)/8 + 1$                        |
| Features / bloc  | 36         | $(16/8)^2 \times 9$                     |
| **Total / frame**| **3780**   | $7 \times 15 \times 36$                 |

Astfel, un eșantion de **T = 19 frame-uri produce un vector de 19 × 3780 = 71 820 valori HOG**.

> **Notă de design (important).** Fereastra HOG trebuie să rămână **64 × 128**. O variantă experimentală pe 32 × 64 (756 features/frame, grilă de blocuri 3 × 7) a fost testată dar este **incompatibilă cu `HARConv3DNet`**: cele două straturi `MaxPool3d(2,2,2)` colapsează lățimea grilei $W = 3 \to 1 \to 0$ (tensor gol → eroare). La 64 × 128 grila e 15 × 7, iar pooling-ul rezistă: $W = 7 \to 3 \to 1$. Constanta `HOG_FEAT_PER_FRAME = 3780` trebuie să fie identică în `extract_hog_augmented.py`, `dataset.py` și implicit în geometria așteptată de `model.py`.

**Calcul matematic intuitiv pentru un pixel.** Gradientul în pixel $(x, y)$ are componentele:

$$
G_x = I(x+1, y) - I(x-1, y), \qquad G_y = I(x, y+1) - I(x, y-1)
$$

cu magnitudinea și orientarea:

$$
\|G\| = \sqrt{G_x^2 + G_y^2}, \qquad \theta = \operatorname{atan2}(G_y, G_x) \bmod \pi
$$

Fiecare pixel contribuie cu $\|G\|$ în bin-ul corespunzător $\theta$ din histograma celulei sale. Blocurile (2 × 2 celule) sunt apoi normalizate L2 (cu clamping la 0.2 — *L2-Hys*) pentru robustețe la iluminare.

> **Figura sugerată #3**: un crop de persoană (boxing) + harta gradienților + suprapunerea celulelor de 8 × 8 + vectorul HOG vizualizat ca „arici" peste imagine.

### 5.3. Politica de augmentare

Augmentările se aplică **doar pe split-ul `train`**. Pe `val` și `test`, fiecare eșantion produce *o singură* variantă originală (fără jitter, fără flip, fără brightness modificat), exact ca în pipeline-ul CSNN.

Pentru fiecare grup de T frame-uri (sample) se generează `num_aug` variante:

1. **Varianta `orig`** (mereu prezentă): fără perturbații, doar crop + redimensionare 64 × 128 + HOG.
2. **Varianta `flip`** (mereu prezentă pentru `num_aug ≥ 2`): orizontal flip.
3. **`num_aug − 2` variante `jit{i}`** (random): combinație aleatoare de jitter geometric + perturbații fotometrice.

Pentru un sample, augmentarea aleatoare aplicată unui bbox $(x, y, w, h)$ generează un bbox $(x', y', w', h')$ astfel:

$$
\begin{aligned}
c_x &= x + w/2, & c_y &= y + h/2 \\
w' &= \max(1,\ \lfloor w \cdot s \cdot p \rfloor), & h' &= \max(1,\ \lfloor h \cdot s \cdot p \rfloor) \\
x' &= \lfloor c_x - w'/2 + \delta_x \rfloor, & y' &= \lfloor c_y - h'/2 + \delta_y \rfloor
\end{aligned}
$$

cu $s$ — scalarea aleatoare, $p$ — `bbox_padding` (factor universal de margine, implicit 1.25), $\delta_x, \delta_y$ — translațiile pe orizontală/verticală. Apoi pe imaginea decupată și redimensionată se aplică:

- **flip orizontal** cu probabilitate 0.5;
- **brightness/contrast**: $I' = \operatorname{clip}(\alpha \cdot I + \beta,\ 0,\ 255)$;
- **gamma** (LUT pe [0, 255]): $I' = 255 \cdot (I/255)^{1/\gamma}$;
- **blur Gaussian** (kernel 3, cu probabilitatea `blur_p`);
- **zgomot Gaussian** după conversia la grayscale: $I' = \operatorname{clip}(I + \mathcal{N}(0, \sigma^2),\ 0,\ 255)$.

#### Profilele `mild` vs `strong`

| Parametru     | `mild`             | `strong`            |
|---------------|--------------------|---------------------|
| scale         | [0.92, 1.08]       | [0.88, 1.15]        |
| $\delta_x$    | [−5, +5] px        | [−8, +8] px         |
| $\delta_y$    | [−3, +3] px        | [−6, +6] px         |
| $\alpha$ (contrast) | [0.85, 1.15] | [0.75, 1.25]        |
| $\beta$ (brightness) | [−15, +15]  | [−25, +25]          |
| $\gamma$      | [0.95, 1.05]       | [0.85, 1.15]        |
| noise $\sigma$ | 2.0 (p = 0.5)     | 4.0 (p = 0.7)       |
| blur ksize    | 3 (p = 0.15)       | 3 (p = 0.25)        |

În experimentele finale s-a folosit `--aug_profile strong --num_aug 8`, ceea ce înseamnă **8 variante per video de antrenare** (1 originală + 1 flip + 6 perturbate aleator). Pe val/test, `num_aug` este forțat la 1.

#### Coerența între streamul HOG și streamul bbox

Un detaliu subtil dar important: când varianta include `flip`, bbox-ul stocat în output **este oglindit** și el (vezi `extract_hog_augmented.py`, `compute_hog_with_aug`):

```python
if aug["flip"]:
    post_box["x"] = frame_w - box["x"] - box["w"]
```

Astfel, atunci când modelul primește în paralel feature-urile HOG (din imaginea oglindită) și bbox-urile (descriere geometrică), ele rămân **consistente conceptual** — bbox-ul indică unde se află persoana în imaginea pe care a văzut-o efectiv HOG-ul.

### 5.4. Structura fișierului `.npz` rezultat

```
features  : (N, T*3780) float32       — vectori HOG concatenați pe T
bboxes    : (N, T, 4)   float32       — (cx, cy, w, h) normalizate ∈ [0, 1]
labels    : (N,)        int64         — index în [0, 5]
metadata  : (N,)        object        — listă de dict-uri Python
config    : (1,)        object        — dict cu config + parametri augmentare
```

Pentru datasetul curent (T = 19, g = 2, num_groups = 10, profil strong, num_aug = 8):

```
total samples : 14715
  train : 11376  (din 154 videoclipuri, augmentate ×8 → 1422 grupuri brute × 8)
  val   :  1529  (din 161 videoclipuri, fără augmentare)
  test  :  1810  (din 190 videoclipuri, fără augmentare)
features shape : (14715, 71820)   ≈ 4.23 GB float32
bboxes shape   : (14715, 19, 4)
```

Câmpul `metadata[i]` conține:

```python
{
  "video_key": "train/boxing/person11_boxing_d1_uncomp.avi",
  "subject": 11, "action": "boxing", "label_idx": 0,
  "group_idx": 3, "aug_idx": 5, "aug_name": "jit2",
  "frame_indices": [50, 52, 54, ..., 86],   # 19 indici, pas g=2
  "split": "train"
}
```

Câmpul `split` este crucial: el este precalculat (din prefixul căii) și permite `HOGDataset` să filtreze rapid eșantioanele fără să mai recalculeze nimic.

#### Layout binar al fișierului `.npz` și ordinea octeților

Pentru a înțelege de ce încărcarea e atât de rapidă comparativ cu re-decodarea video, e util de știut exact ce produce `np.savez` pe disk.

Un `.npz` este de fapt **un arhiv ZIP** care conține mai multe fișiere `.npy`, câte unul pentru fiecare array salvat:

```
hog_aug_tvt_19_f10_g2_runfix.npz  =  ZIP container
   ├── features.npy   (N × T × 3780 × 4 octeți = float32)
   ├── bboxes.npy     (N × T × 4 × 4 octeți = float32)
   ├── labels.npy     (N × 8 octeți = int64)
   ├── metadata.npy   (object array — pickle Python)
   └── config.npy     (object array — pickle Python)
```

Fișierele numerice (features, bboxes, labels) sunt salvate în formatul binar nativ NumPy `.npy`:

```
\x93NUMPY\x01\x00<hl><header_dict><raw_bytes>
   ^magic    ^ver  ^len  ^dict     ^valorile pe rând
```

Header-ul e un dict Python serializat ca string:

```python
{'descr': '<f4', 'fortran_order': False, 'shape': (14715, 71820)}
#         ^ float32 little-endian      ^ N=14715 sample-uri × 71820 features
```

Apoi urmează `N × (T · F) × sizeof(float32)` octeți consecutivi. Pentru `features`:

$$
N_{\text{octeți}} = 14\,715 \times 19 \times 3780 \times 4 = 4\,227\,310\,800 \approx 4.23\ \text{GB}
$$

NumPy folosește layout **C-order (row-major)**. Pentru un array conceptual de formă `(N, T, F)`, indexul liniar al elementului `[i, t, f]` este:

$$
\text{offset}(i, t, f) = i \cdot (T \cdot F) + t \cdot F + f
$$

iar adresa octetului concret (4 octeți per float32):

$$
\text{addr}_{\text{byte}} = \text{header\_size} + 4 \cdot \text{offset}(i, t, f)
$$

Asta înseamnă că **eșantionul `i` complet (toate cele T = 19 frame-uri × toate cele 3780 features) ocupă un bloc continuu** de octeți. Citirea unui sample = un `memcpy` din vectorul plat în tensor PyTorch.

`np.load(path, mmap_mode='r')` mapează fișierul în spațiul de adrese fără să-l copieze în RAM. La accesul `features[42]`, sistemul de operare aduce paginile necesare on-demand din disk. Pentru workload-ul nostru (date care încap în RAM), avantajul real este *evitarea preîncărcării totale* la pornirea procesului — paginile cele mai accesate ajung în page-cache-ul kernelului.

### 5.5. Exemple de utilizare

```bash
cd cnn_har_app

# Pre-calcul HOG augmentat pe JSON-ul FINAL (running reparat), T=19, g=2, 10 grupuri, profil strong, 8 augmentări:
python3 extract_hog_augmented.py \
    --bbox_json ../hog/hog_person_data_tvt_19_f10_g2_runfix.json \
    --output   ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --num_aug 8 --aug_profile strong \
    --video_root /home/mmuntean/kth_organized_tvt
```

La final scriptul afișează statisticile pe split-uri; verifică `features shape: (14715, 71820)` — dacă vezi 71820 (= 19 × 3780), geometria e corectă.

> **Figura sugerată #4**: pentru același sample (același video, același grup), 4–6 variante augmentate alăturate ca thumbnails (orig, flip, jit1 cu brightness scăzut, jit2 cu zoom in + dy negativ, etc.).

---

## 6. Etapa 3 — `dataset.py`

`HOGDataset` este o subclasă `torch.utils.data.Dataset` care încapsulează încărcarea, reorganizarea și augmentarea online. Decide automat formatul de intrare după extensia fișierului:

- **`.json`** — bbox-only; HOG-ul se recalculează la *runtime* (folosit doar pentru debug);
- **`.npz`** — fișierul produs de `extract_hog_augmented.py` (calea standard pentru producție).

Filtrarea pe split se face din câmpul `meta["split"]` al fiecărui eșantion (fallback pe subiect doar pentru NPZ-uri vechi fără tag).

### 6.1. Reorganizarea HOG → tensor 4D

Vectorul plat de 3780 features per frame este reorganizat ca tensor 3D înainte să intre în convoluțiile 3D. Convenția OpenCV pentru ordonarea blocurilor este row-major peste $(n_y, n_x, 36)$, astfel încât reshape-ul + permute-ul este:

$$
\underbrace{(T \cdot 3780)}_{\text{plat}}
\quad \xrightarrow{\text{view}} \quad
(T, n_y{=}15, n_x{=}7, 36)
\quad \xrightarrow{\text{permute}} \quad
\underbrace{(T, C{=}36, H{=}15, W{=}7)}_{\text{tensor 4D}}
$$

C corespunde celor 36 de features per bloc (deci 36 de canale „semantice" HOG), iar H × W = 15 × 7 reprezintă grila spațială de blocuri. Cu T = 19, tensorul rezultat (înainte de fluxurile auxiliare) este $(19, 36, 15, 7)$.

#### Acces concret la eșantion în `HOGDataset.__getitem__`

Iată ce face DataLoader-ul când cere eșantionul `i` (T = 19):

```python
def __getitem__(self, i):
    # 1. View în date (fără copie):
    feat_flat = self.features[i]          # shape (T*3780,)
    bb        = self.bboxes[i]            # shape (T, 4)
    label     = self.labels[i]            # int64

    # 2. Reshape + permute la (T, C=36, H=15, W=7):
    feat = feat_flat.reshape(T, 15, 7, 36).transpose(0, 3, 1, 2)

    # 3. Adaugă diff (motion):
    diff = np.diff(feat, axis=0, prepend=feat[:1])     # (T, 36, 15, 7)

    # 4. Broadcast bbox peste grila spațială:
    bbox_grid = np.broadcast_to(bb[:, :, None, None], (T, 4, 15, 7))
    bbox_vel  = np.diff(bb, axis=0, prepend=bb[:1])
    bbox_vel  = np.broadcast_to(bbox_vel[:, :, None, None], (T, 4, 15, 7))

    # 5. Concatenare canale:
    x = np.concatenate([feat, diff, bbox_grid, bbox_vel], axis=1)  # (T, 80, 15, 7)

    # 6. Augmentări online (doar train) ...
    # 7. Convert la tensor.
    return torch.from_numpy(x).float(), label
```

Costul fiecărui pas pentru un sample (T = 19):

| Pas | Memorie atinsă | Cost aprox. |
|---|---|---|
| view în date | 0 (lazy) | ~µs |
| reshape + transpose | 19 × 3780 × 4 ≈ 287 KB | ~25 µs |
| diff temporal | 287 KB | ~25 µs |
| broadcast bbox | 0 (broadcast, fără alocare) | ~µs |
| concatenare 80 canale | 19 × 80 × 15 × 7 × 4 ≈ 638 KB | ~50 µs |
| augmentări | 638 KB | ~60 µs |
| → torch | 638 KB | ~40 µs |

Per total: ~200 µs/sample. Pentru un batch de 64, ~13 ms — comparabil sau sub costul unui forward+backward pe GPU.

### 6.2. Fluxuri suplimentare (input streams)

Pe lângă streamul HOG nud, dataset-ul poate concatena pe axa canalelor 3 streamuri auxiliare. Modelul final folosește toate cele 3, ducând la **80 canale per frame**:

| Stream         | Canale | Definiție                                                  | Captură fizică                          |
|----------------|-------:|------------------------------------------------------------|-----------------------------------------|
| HOG            | 36     | $H_t$                                                      | aspect spațial al persoanei la $t$      |
| Diff (motion)  | 36     | $\Delta H_t = H_t - H_{t-1}$, cu $\Delta H_0 = 0$           | derivată temporală a aspectului          |
| BBox metadata  | 4      | $(c_x, c_y, w, h)$ broadcast peste grila $H \times W$       | poziția absolută în cadru (translație)  |
| BBox velocity  | 4      | $(\dot c_x, \dot c_y, \dot w, \dot h)$ broadcast peste grilă | viteza de translație                    |

**De ce sunt importante streamurile bbox.** După crop + resize la 64 × 128, *toți* subiecții arată în interiorul ferestrei HOG ca având aceeași dimensiune și aceeași poziție. Modelul pierde astfel exact informația care diferențiază walking de jogging și de running: viteza de translație globală a persoanei pe ecran. Inserând bbox-ul ca 4 canale broadcastate pe toată grila $H \times W$, modelul recâștigă acest semnal **fără să-i fie nevoie să recupereze indirect translația din mici artefacte de crop**.

> **Figura sugerată #5**: schema canalelor pe verticală — 36 HOG + 36 diff + 4 bbox + 4 bbox_vel = 80 (cu blocuri colorate diferit).

### 6.3. Augmentări online pe train (în `__getitem__`)

Aplicate doar dacă `split == "train"` și `augment=True`:

1. **Temporal reverse** cu probabilitate $p_{\text{rev}} = 0.3$: inversează ordinea celor T frame-uri (atât HOG, cât și bbox).
2. **Temporal shift** uniform în $[-\Delta, +\Delta]$ cu $\Delta = 2$ frame-uri, padded cu zerouri.
3. **Gaussian noise** pe vectorul HOG cu $\sigma = 0.003$.
4. **Feature dropout** Bernoulli cu $p = 0.015$ pe fiecare element al vectorului HOG.

Combinate cu augmentările din `.npz` (offline), modelul nu vede practic niciodată exact același eșantion de două ori.

---

## 7. Etapa 4 — `model.py`

Fișierul `model.py` definește patru arhitecturi expuse prin `build_model(model_type=...)`:

| `model_type` | Clasă             | Input                  | Comentariu                                  |
|--------------|-------------------|------------------------|---------------------------------------------|
| `mlp`        | `HARLinearNet`    | $(B,\ T \cdot 3780)$   | Baseline naiv pe vectorul plat              |
| `cnn`        | `HARConvNet`      | $(B,\ T \cdot C,\ H,\ W)$ | Frame-urile stivuite pe canale; 2D CNN   |
| `temporal`   | `HARTemporalNet`  | $(B,\ T,\ C,\ H,\ W)$  | Per-frame encoder + BiGRU + attention       |
| **`conv3d`** | **`HARConv3DNet`**| $(B,\ T,\ C,\ H,\ W)$  | **Modelul principal**; convoluții 3D dense  |

**Modelul de producție este `HARConv3DNet`** (toate rezultatele raportate în secțiunea 11 sunt obținute cu această arhitectură).

### 7.1. `HARConv3DNet` — convoluții spațio-temporale

Conceptual, această rețea este analogă cu **HOG3D** (Kläser et al., 2008), descriptorul HOG extins în spațiu-timp, care reprezintă cea mai puternică baseline bazată pe HOG pe KTH (~91%). Spre deosebire de HOG3D însă, kernelele 3D *se învață* din date împreună cu clasificatorul.

Tensorul de intrare $(B, T, C, H, W)$ este permutat intern la $(B, C, T, H, W)$ pentru a respecta convenția PyTorch `Conv3d` (canal-prim).

#### Arhitectura completă (exact ca în cod)

```
Block 1 — învățare locală spațio-temporală:
  Conv3d(80 → 96,  k=3×3×3, pad=1) + BN + ReLU
  Conv3d(96 → 128, k=3×3×3, pad=1) + BN + ReLU
  MaxPool3d(2×2×2)                              # (T,H,W): 19→9, 15→7, 7→3
  Dropout3d(0.25)

Block 2 — adâncire + pooling spațio-temporal:
  Conv3d(128 → 192, k=3×3×3, pad=1) + BN + ReLU
  Conv3d(192 → 256, k=3×3×3, pad=1) + BN + ReLU
  MaxPool3d(2×2×2)                              # (T,H,W): 9→4, 7→3, 3→1
  Dropout3d(0.25)

Block 3 — colapsare la descriptor global:
  Conv3d(256 → 256, k=3×3×3, pad=1) + BN + ReLU
  AdaptiveAvgPool3d(1)                          # (1,1,1)

Classifier (head):
  Flatten → Dropout(p=dropout)
  Linear(256 → 64) + ReLU
  Dropout(p=dropout/2)
  Linear(64 → 6)                                # 6 clase KTH
```

#### Decizii de design și de ce contează

- **Kernel 3 × 3 × 3 pe toate blocurile** — kernel cubic uniform, simetric pe toate cele 3 dimensiuni (timp, înălțime, lățime). Asta permite rețelei să învețe pattern-uri spațio-temporale locale (ex: un edge orizontal care se deplasează vertical pe 3 frame-uri = mâna care se ridică în handwaving).
- **Două `MaxPool3d(2,2,2)`** — fiecare înjumătățește toate cele 3 dimensiuni. Pe configurația T = 19, grilă 15 × 7, traseul este $19{\times}15{\times}7 \to 9{\times}7{\times}3 \to 4{\times}3{\times}1$. Important: lățimea grilei (W = 7) este exact suficientă ca să nu se anuleze ($7 \to 3 \to 1$); de aici constrângerea ferestrei HOG la 64 × 128 (vezi nota de la 5.2).
- **`BatchNorm3d` + ReLU** după fiecare convoluție — stabilizează antrenarea pe loturi (B = 64), important pentru reproductibilitate pe seed-uri multiple.
- **`Dropout3d` (canal-wise)** între blocuri în loc de Dropout standard — randomizează direct hărți de feature întregi, regularizator mai agresiv pentru convoluții.
- **`AdaptiveAvgPool3d(1)`** — elimină orice dependență de input size și produce un descriptor global de 256-d.
- **Head clasificator subțire** (256 → 64 → 6) — restul capacității stă în extractor; head-ul mic + dropout previne overfitting pe doar 8 subiecți de antrenare.

#### Capacitatea modelului

| Componentă             | Parametri |
|------------------------|----------:|
| Block 1 (Conv3d × 2)   | ~340 K    |
| Block 2 (Conv3d × 2)   | ~1.99 M   |
| Block 3 (Conv3d × 1)   | ~1.77 M   |
| Classifier             | ~17 K     |
| **Total** *(aprox., cu C = 80)* | **~4.1 M** |

> Numărul exact de parametri e tipărit la începutul antrenării (`Trainable params: X.XXM`).

> **Figura sugerată #6**: diagrama arhitecturii cu volume tensoriale notate pe muchii (input 80×19×15×7 → … → 6 logits).

### 7.2. Forward pass detaliat cu urmărirea formelor

Hai să urmărim un batch de 64 prin rețea, dimensiune cu dimensiune. Input: `(B=64, T=19, C=80, H=15, W=7)`.

**Pasul 0 — Permute pentru convenția PyTorch `Conv3d`**

`Conv3d` așteaptă `(B, C, D, H, W)` (canal-prim, apoi adâncime/timp). Convertim:

$$
(B, T, C, H, W) \xrightarrow{\text{permute}(0, 2, 1, 3, 4)} (B, C, T, H, W) = (64, 80, 19, 15, 7)
$$

**Block 1 — învățare locală spațio-temporală**

```
(64, 80, 19, 15, 7)
   │ Conv3d(in=80, out=96, k=3×3×3, padding=1)
   ▼
(64, 96, 19, 15, 7)           ← padding=1 păstrează T, H, W
   │ BatchNorm3d + ReLU
   │ Conv3d(96 → 128, k=3×3×3, padding=1) + BN + ReLU
   ▼
(64, 128, 19, 15, 7)
   │ MaxPool3d(kernel=(2, 2, 2))   ← T, H, W toate ÷ 2 (cu floor)
   ▼
(64, 128, 9, 7, 3)
   │ Dropout3d(0.25)
   ▼
(64, 128, 9, 7, 3)
```

**Block 2 — adâncire + pooling spațio-temporal**

```
(64, 128, 9, 7, 3)
   │ Conv3d(128 → 192) + BN + ReLU
   │ Conv3d(192 → 256) + BN + ReLU
   ▼
(64, 256, 9, 7, 3)
   │ MaxPool3d(kernel=(2, 2, 2))    ← T: 9→4, H: 7→3, W: 3→1
   ▼
(64, 256, 4, 3, 1)
   │ Dropout3d(0.25)
   ▼
(64, 256, 4, 3, 1)
```

**Block 3 — colapsare globală**

```
(64, 256, 4, 3, 1)
   │ Conv3d(256 → 256) + BN + ReLU
   │ AdaptiveAvgPool3d(output=(1,1,1))
   ▼
(64, 256, 1, 1, 1)
   │ Flatten
   ▼
(64, 256)
```

**Head clasificator**

```
(64, 256) → Dropout(0.2) → Linear(256→64) + ReLU → Dropout(0.1) → Linear(64→6) → (64, 6)
```

Regula de pooling folosită: $D_{\text{out}} = \lfloor (D_{\text{in}} - k)/s + 1 \rfloor$ cu $k = s = 2$. Verificare: $19 \to \lfloor 9.5 \rfloor = 9 \to \lfloor 4.5 \rfloor = 4$; $15 \to 7 \to 3$; $7 \to 3 \to 1$.

#### Matematica operației Conv3d într-un punct

Pentru un singur output `(b, c', t, h, w)` al primei convoluții (k=3, pad=1):

$$
y[b, c', t, h, w] = \beta_{c'} + \sum_{c=0}^{C_{\text{in}}-1} \sum_{dt=-1}^{+1} \sum_{dh=-1}^{+1} \sum_{dw=-1}^{+1} W[c', c, dt, dh, dw] \cdot x[b, c, t+dt, h+dh, w+dw]
$$

cu zero-padding pe ramurile la margine. Suma rulează peste $C_{\text{in}} \times 27$ produse pentru fiecare output → pentru Block 1 conv1 cu $C_{\text{in}}=80$: $80 \times 27 = 2160$ produse × $96 \times 19 \times 15 \times 7 = 191\,520$ output-uri ≈ **414 M MAC-uri** per batch element.

#### BatchNorm 3D

Pentru fiecare canal $c'$ separat, normalizează pe `(B, T, H, W)`:

$$
\mu_{c'} = \frac{1}{B T H W} \sum_{b,t,h,w} y[b, c', t, h, w], \qquad
\sigma_{c'}^2 = \frac{1}{B T H W} \sum_{b,t,h,w} (y - \mu_{c'})^2
$$

$$
\hat y[b, c', t, h, w] = \gamma_{c'} \cdot \frac{y[b, c', t, h, w] - \mu_{c'}}{\sqrt{\sigma_{c'}^2 + \epsilon}} + \beta_{c'}
$$

cu $\gamma, \beta$ ponderi învățabile per canal.

### 7.3. Loss, gradienți și actualizarea ponderilor

#### Cross-entropy cu label smoothing

Pentru un eșantion cu etichetă $y \in \{0, \dots, 5\}$ și logiți $z \in \mathbb{R}^6$:

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad q_k = \begin{cases} 1 - \varepsilon & k = y \\ \varepsilon / (K-1) & k \neq y \end{cases}
$$

$$
\mathcal{L} = -\sum_{k=0}^{K-1} q_k \log p_k
$$

Cu $\varepsilon = 0.02, K = 6$: $q_y = 0.98$, restul fiind $0.004$ fiecare.

#### Gradientul la stratul de logiți

Pentru cross-entropy + softmax, gradientul are formă închisă foarte simplă:

$$
\frac{\partial \mathcal{L}}{\partial z_k} = p_k - q_k
$$

Acest gradient se propagă înapoi cu chain rule. Pentru o convoluție 3D cu ponderi $W$:

$$
\frac{\partial \mathcal{L}}{\partial W[c', c, dt, dh, dw]} = \sum_{b, t, h, w} \frac{\partial \mathcal{L}}{\partial y[b, c', t, h, w]} \cdot x[b, c, t+dt, h+dh, w+dw]
$$

— o corelație între gradientul ieșirii și inputul.

#### Update AdamW

Pentru fiecare parametru $\theta$, la pasul $t$ (cu $g_t = \partial \mathcal{L} / \partial \theta$):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= m_t / (1 - \beta_1^t), \quad \hat v_t = v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_{t-1}\right)
\end{aligned}
$$

cu $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=10^{-3}$, $\lambda=3 \times 10^{-4}$. Diferența față de Adam clasic: weight decay $\lambda \theta_{t-1}$ se aplică direct pe parametri, nu prin gradient (Loshchilov & Hutter, 2019).

#### Cosine Annealing

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\tfrac{t}{T_{\max}} \pi\right)\right)
$$

cu $\eta_{\max}=10^{-3}$, $\eta_{\min}\approx 0$, $T_{\max}=200$ epoch-uri.

---

## 8. Etapa 5 — `train.py`

### 8.1. Buclă de antrenare

```
Pentru fiecare epoch:
  Pentru fiecare batch:
    1. Mixup pe input (α = 0.02 → aproape identitate, regularizator slab)
    2. Forward + loss (cross-entropy cu label_smoothing = 0.02)
    3. Backward + grad-clipping (max-norm = 1.0)
    4. Optimizer step (AdamW)
    5. Update EMA shadow weights (după epoch ≥ ema_start)
  Schedule LR (CosineAnnealingLR)
  Evaluare pe VAL (cu greutățile EMA, dacă active)
  Salvare checkpoint dacă acuratețea de VAL crește   ← selecția modelului
La final:
  Reîncarcă cel mai bun checkpoint și raportează acuratețea pe TEST (o singură dată)
```

> **Selecția modelului se face pe `val`, nu pe `test`.** Val este set held-out, deci early stopping-ul nu „vede" niciodată test-ul. Dacă datasetul nu are split val (NPZ vechi), `train.py` cade pe test ca surogat și avertizează zgomotos — dar datasetul curent are val (1529 eșantioane), deci protocolul e curat. Cifra de generalizare raportată este **acuratețea pe test** atinsă cu modelul ales pe val.

### 8.2. Tehnici de regularizare folosite

| Tehnică                  | Valoare          | Rol                                                          |
|--------------------------|------------------|--------------------------------------------------------------|
| **AdamW**                | lr = 1e-3, wd = 3e-4 | Optimizator standard cu weight decay decuplat              |
| **CosineAnnealingLR**    | T_max = epochs   | LR scade lin până la ~0 la finalul antrenării                |
| **Mixup**                | α = 0.02         | Foarte slab — combinație lin a două eșantioane               |
| **Label smoothing**      | ε = 0.02         | Țintele devin $1 - \varepsilon$ pe clasa corectă, $\varepsilon/5$ pe restul |
| **Gradient clipping**    | max_norm = 1.0   | Stabilitate pe gradienți rari de magnitudine mare            |
| **EMA (Exp. Moving Avg.)** | decay = 0.999  | Greutăți de evaluare = medie exponențială a celor de train   |
| **Early stopping**       | patience = 60    | Stop dacă acuratețea de **val** nu mai crește în 60 epoch-uri |
| **Dropout (head)**       | 0.2              | Regularizator clasic în clasificator                         |

#### Detaliu Mixup

Pentru două eșantioane $(x_i, y_i)$ și $(x_j, y_j)$ și $\lambda \sim \operatorname{Beta}(\alpha, \alpha)$:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \qquad
\mathcal{L} = \lambda \cdot \mathrm{CE}(\hat y, y_i) + (1 - \lambda) \cdot \mathrm{CE}(\hat y, y_j)
$$

Cu $\alpha = 0.02$, distribuția Beta este puternic concentrată la 0 și 1, deci în practică mixup-ul perturbează ușor doar o fracțiune mică din batch-uri.

#### EMA — de ce contează

Greutățile efectiv folosite la evaluare sunt:

$$
\theta_{\text{EMA}}^{(t)} = \rho \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \rho) \cdot \theta^{(t)}, \quad \rho = 0.999
$$

EMA produce un model „mai neted" în spațiul parametrilor, ce evită mini-oscilațiile de la finalul antrenării. În experimente, contribuie de obicei cu ~0.3–0.6% acuratețe absolută față de greutățile finale ne-mediatate.

### 8.3. Configurație folosită pentru rezultatele raportate

Hardware: nod **larochette** (Grid'5000), GPU **AMD Instinct MI210** (gfx90a), PyTorch pe stack **ROCm 6.3** (`torch 2.8.0+rocm6.3`). Nodul are 4 GPU-uri, deci cele 5 seed-uri pot rula 4-în-paralel (câte unul per GPU prin `HIP_VISIBLE_DEVICES`).

```bash
cd cnn_har_app
for s in 42 123 7 13 99; do
  python3 train.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --model_type conv3d \
    --balanced_sampler none \
    --seed $s \
    --save_suffix _tvt19fix_s$s \
    --temporal_reverse_p 0.3 \
    --temporal_shift_max 2 \
    --ema_decay 0.999 \
    --ema_start 5 \
    2>&1 | tee ../data/log_cnn_tvt19fix_s$s.txt
done
```

Varianta 4-GPU paralel (4 seed-uri simultan, al 5-lea după):

```bash
for i in 0 1 2 3; do
  seeds=(42 123 7 13); s=${seeds[$i]}
  HIP_VISIBLE_DEVICES=$i python3 train.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz --model_type conv3d \
    --balanced_sampler none --seed $s --save_suffix _tvt19fix_s$s \
    --temporal_reverse_p 0.3 --temporal_shift_max 2 \
    > ../data/log_cnn_tvt19fix_s$s.txt 2>&1 &
done
wait
```

> Prima linie a fiecărei rulări trebuie să fie `Using device: cuda` (pe ROCm, GPU-ul AMD e expus tot prin API-ul „cuda"). Dacă apare `cpu`, build-ul de torch e greșit (NVIDIA `+cu128` în loc de `+rocm6.3`).

> **Figura sugerată #7**: curbele de train/val loss și train/val accuracy pe parcursul antrenării pentru un seed reprezentativ, cu marcarea epoch-ului ales (best val).

---

## 9. Etapa 6 — `eval_ensemble.py`

### 9.1. Ensemble pe seed-uri

Antrenăm 5 modele identice cu seed-urile $\{42, 123, 7, 13, 99\}$ și combinăm predicțiile lor la inferență pe **test**. Două moduri de agregare:

| Mod        | Formulă                                              | Comentariu                         |
|------------|------------------------------------------------------|------------------------------------|
| `logits`   | $\bar z = \frac{1}{M} \sum_m z_m$, apoi $\operatorname{argmax} \bar z$ | **Recomandat**; păstrează scara raw |
| `softmax`  | $\bar p = \frac{1}{M} \sum_m \operatorname{softmax}(z_m)$, apoi $\operatorname{argmax} \bar p$ | Comprimă scorurile foarte încrezătoare |

Mean-logits e cel folosit pentru rezultate, deoarece în practică e mai stabil când unul dintre modele e foarte sigur și greșește — softmax-ul i-ar amplifica artificial vocea în vot.

> **Atenție la comparația cu CSNN.** Ensemble-ul de 5 modele dă **o singură** cifră (de obicei mai mare decât oricare seed individual) și NU se compară 1:1 cu un CSNN single-model. Pentru comparația corectă vs CSNN se folosește **media ± deviația standard a celor 5 acuratețe de test individuale** (care măsoară performanța tipică + variabilitatea). Ensemble-ul + TTA se raportează separat, ca limită superioară a metodei.

### 9.2. Test-Time Augmentation (TTA)

Pe lângă media pe seed-uri, agregăm și predicții pe versiuni transformate ale aceluiași eșantion:

- **`--tta_reverse`**: inferență pe clipul cu frame-urile inversate temporal. Util fiindcă acțiunile periodice (handwaving, handclapping, mers) sunt aproximativ invariante la inversare.
- **`--tta_shift 1`**: inferență pe clipul shiftat cu ±1 frame (padded cu zerouri). Crește robustețea la sincronizarea exactă a începutului acțiunii.

Numărul total de evaluări per eșantion devine:

$$
N_{\text{eval}} = M \cdot \big(1 + \mathbb{1}[\text{tta\_reverse}] + 2 \cdot \mathbb{1}[\text{tta\_shift}>0]\big)
$$

Pentru $M = 5$, `--tta_reverse --tta_shift 1`, asta înseamnă **5 × (1 + 1 + 2) = 20 evaluări per clip de test**, iar logit-ul final e media celor 20.

### 9.3. Comandă folosită

```bash
python3 eval_ensemble.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --model_type conv3d \
    --checkpoints "models/har_conv3d_tvt19fix_s*.pth" \
    --ensemble_mode logits \
    --tta_reverse \
    --tta_shift 1
```

---

## 10. Etapa 7 — `app.py` (Streamlit)

### 10.1. Scop

`app.py` este un demonstrator interactiv pe care îl deschidem la prezentarea licenței. El permite:

1. Selectarea unui subset de checkpoint-uri (ansamblu configurabil).
2. Alegerea unui clip din test split filtrat după subiect/acțiune.
3. Rularea inferenței reale (nu mock) cu agregarea pe modele selectate.
4. Afișarea distribuției probabilităților pe toate cele 6 clase.
5. Vizualizarea celor T = 19 frame-uri sursă cu bbox-ul desenat peste, pentru a arăta vizual că modelul „se uită" exact la persoană.

> **Demo configurat pe datasetul/modelele finale**: în `app.py`, `DATA_PATH = hog_aug_tvt_19_f10_g2_runfix.npz`, `ckpt_pattern = har_conv3d_tvt19fix_s*.pth`, iar `VIDEO_ROOT` pointează la rădăcina KTH reorganizată tvt (cu `test/<acțiune>/`). Pe Windows căile sunt absolute (vezi capul fișierului).

### 10.2. Fluxul utilizatorului

```
Sidebar:
  - selectarea checkpoint-urilor (multiselect) → încarcă modele
  - alegerea modului ensemble (logits / softmax)
  - panou informativ: arhitectură, nr. parametri, device

Main:
  Pas 1: alege subiect → acțiune → clip
  Pas 2: buton „Clasifică" → rulează inferență
  Pas 3: afișează:
    - predicție vs ground truth (✅ / ❌)
    - bar chart cu probabilități per clasă
    - frame-urile sursă cu bbox-ul în verde
```

### 10.3. Cache-uri pentru responsivitate

Streamlit re-rulează scriptul la fiecare interacțiune. Pentru a evita reîncărcarea modelului și a dataset-ului la fiecare click, folosim:

- `@st.cache_resource` pentru `HOGDataset` și pentru modelele PyTorch (resurse „grele");
- `@st.cache_data` pentru fetch-ul de frame-uri din videoclip (cheia de cache este tupla `(video_path, frame_indices)`).

### 10.4. Lansare

```bash
# Local:
streamlit run app.py

# Pe un nod remote, expus extern (cu portul 8501 deschis):
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

> **Figura sugerată #8**: screenshot al UI-ului Streamlit (sidebar + thumbnails cu bbox-uri verzi + bar chart-ul probabilităților).

---

## 11. Rezultate experimentale

> **Rezultate finale (2026-06-09)** — antrenare pe larochette (MI210, ROCm). Bottleneck-ul clasei `running` (vezi istoric în 11.4) a fost **rezolvat la sursă** prin re-extracția clasei running cu praguri de detecție permisive + toleranță de carry temporal mai mare (`--max_carry 4`). Datasetul folosit este `hog_aug_tvt_19_f10_g2_runfix.npz` (JSON `hog_person_data_tvt_19_f10_g2_runfix.json`).

Configurația tuturor rezultatelor de mai jos:

- **Arhitectură**: `HARConv3DNet` (4.32M parametri)
- **Dataset**: T = 19, frame_gap = 2, num_groups = 10, fereastră HOG 64 × 128, split tvt (8/8/9 subiecți), **running re-extras** (max_carry 4)
- **80 canale per frame**: HOG (36) + diff (36) + bbox (4) + bbox_vel (4)
- **5 seed-uri**: 42, 123, 7, 13, 99
- **Selecție model**: best pe **val**; cifră raportată pe **test**
- **Augmentare offline**: profil `strong`, `num_aug = 8` (doar train)
- **Augmentare online**: `temporal_reverse_p = 0.3`, `temporal_shift_max = 2`, Gaussian noise, feature dropout
- **Fără sampler** (`--balanced_sampler none`) — vezi 11.4 de ce oversampling-ul a fost contraproductiv
- **Ensemble**: mean logits + TTA reverse + TTA shift ±1

### 11.1. Acuratețe per seed (test)

| Seed | Best val | Acuratețe **test** | running recall |
|-----:|:--------:|:------------------:|:--------------:|
| 42   | 92.74 %  | **90.28 %**        | 77.6 %         |
| 123  | 92.15 %  | 89.78 %            | 78.1 %         |
| 7    | 93.00 %  | 88.73 %            | 73.8 %         |
| 13   | 92.41 %  | 88.90 %            | 79.5 %         |
| 99   | 94.11 %  | 91.33 %            | 69.5 %         |
| **Medie ± std (test)** | | **89.80 % ± 1.06** | **~75.7 %** |

### 11.2. Ensemble + TTA

| Configurație | Acuratețe test |
|---|:---:|
| Ensemble 5 seed-uri + TTA (reverse + shift ±1), mean logits | **90.55 %** |
| Δ ensemble față de cel mai bun seed individual (91.33 %) | **−0.78 %** |

### 11.3. Matrice de confuzie (test, ensemble)

|              | boxing | handclap | handwav | jogging | running | walking | Recall |
|--------------|-------:|---------:|--------:|--------:|--------:|--------:|-------:|
| **boxing**       | 300 | 0   | 0   | 0   | 0   | 8   | 97.4 % |
| **handclapping** | 4   | 328 | 18  | 0   | 0   | 0   | 93.7 % |
| **handwaving**   | 0   | 53  | 277 | 0   | 0   | 0   | 83.9 % |
| **jogging**      | 6   | 0   | 0   | 244 | 4   | 0   | 96.1 % |
| **running**      | 4   | 0   | 0   | 50  | 153 | 3   | **72.9 %** |
| **walking**      | 7   | 0   | 0   | 14  | 0   | 337 | 94.1 % |

### 11.4. Interpretarea erorilor

Două surse de eroare, foarte diferite ca natură:

1. **`running` → `jogging` (50 din 210) — recall 72.9 %, ridicat de la 12.9 % printr-un fix de date.** Inițial, din cele 32 de videoclipuri running din `train/running/`, doar **4** produceau grupuri valide la extracție (`extract_bboxes_kth.py` păstrează un grup doar dacă toate cele 19 frame-uri au bbox bun; la mișcare rapidă persoana iese din cadru și detecția pică) → 21 grupuri unice running vs 1648–2480 la celelalte clase, deci modelul aproape nu învăța clasa (recall 12.9 %).
   - **Fix aplicat (la sursă, nu prin sampler):** re-extracția *doar* a clasei running cu (a) praguri de detecție relaxate (`hit_threshold −1.2`, `min_bbox_area_ratio 0.004`, aspect 0.15–2.0, `mog2_min_area 300`) și (b) toleranță de carry temporal `--max_carry 4` (cariază bbox-ul anterior până la 4 frame-uri consecutive când persoana iese scurt din cadru). Apoi merge al cheilor `*/running/*` în JSON (restul claselor identic) și regenerarea npz-ului. Rezultat: **train running 4 → 16 videoclipuri**, recall **12.9 % → 72.9 %**, ensemble **89.22 % → 90.55 %**.
   - **De ce NU sampler:** o variantă anterioară cu `WeightedRandomSampler(inv)` a oversamplat cele 21 de grupuri running — n-a creat informație nouă, doar instabilitate (Val Acc colapsa în primele epoci), și a *scăzut* ensemble-ul. Oversampling-ul nu înlocuiește date reale; fix-ul corect e la extracție. Antrenarea finală e cu `--balanced_sampler none`.
   - Cele 50 de erori reziduale running→jogging sunt acum un **plafon intrinsec** (cele două clase diferă doar prin viteza de translație), nu o lipsă de date.
2. **`handwaving` ↔ `handclapping`** (53 handwaving → handclapping). Pereche clasic confundabilă pe KTH — ambele mișcări locale ale mâinilor, fără translație globală; diferă prin amplitudine/sincronizare.

> **Figura sugerată #9**: bar chart acuratețe per seed individual vs ensemble.
> **Figura sugerată #10**: heatmap al matricei de confuzie pe test (cu evidențierea celulei running→jogging).

---

## 12. Comparație CNN ↔ CSNN

Tot lanțul de pre-procesare (detecție de persoană → bbox → crop → HOG) și **același split de subiecți (tvt 8/8/9)** sunt identice între cele două pipeline-uri, ceea ce face comparația metric vs metric corectă. Diferențele se reduc la *cum* se face clasificarea pe vectorii HOG temporali.

| Aspect                  | CSNN (rețea spiking)                       | CNN (acest pipeline)                       |
|-------------------------|--------------------------------------------|--------------------------------------------|
| Codare                  | Latency / rate coding pe HOG               | Float dens, fără codare                    |
| Învățare                | STDP local + WTA, fără backprop            | Backprop end-to-end (AdamW + CosineLR)     |
| Unitate                 | Neuron LIF (Leaky Integrate-and-Fire)      | Conv3D + BatchNorm + ReLU                  |
| Reprezentare temporală  | Acumulare în timp continuu                 | Convoluții 3D pe T = 19 frame-uri          |
| Segmentare              | T = 19, frame_gap = 2                      | T = 19, frame_gap = 2 (identic)            |
| Augmentare              | Aceeași sursă HOG                          | Aceeași sursă HOG + augmentări CNN         |
| Selecție / raportare    | val + test pe seed-uri                     | best pe val, raportare medie ± std pe test |
| Energie & sparsitate    | Activitate sparsă, candidat hardware neuromorfic | Dens, energie proporțională cu MAC-uri  |
| Acuratețe test (medie ± std) | _____ % ± _____ *(din rezultatele CSNN)* | **89.80 % ± 1.06** |
| Acuratețe test (ensemble + TTA) | — | **90.55 %** |

> Cifrele CNN sunt **finale** (running reparat la sursă, vezi 11.4). Tabelul se completează când avem și cifrele CSNN. Interpretare: CNN-ul oferă **plafonul superior** pe această reprezentare HOG cu acest split; CSNN se evaluează relativ la acel plafon ca variantă energy-efficient pentru deployment neuromorfic.

---

## 13. Direcții viitoare

1. **Explorarea altor combinații (T, frame_gap).** Configurația curentă (T = 19, g = 2) acoperă ~1.4 s de context. Util de comparat cu g mai mare (context temporal mai lung) vs stabilitatea bbox-ului.
2. **Classificator secundar specializat pe perechi confundabile.** Dacă majoritatea erorilor reziduale provin din `handclapping ↔ handwaving` și `jogging ↔ running`, o strategie două-stadii ar putea ajuta: stadiul 1 = clasificator pe 6 clase; stadiul 2 = CNN dedicat doar perechii suspecte (de ex. doar pe bbox-velocity pentru `jogging ↔ running`).
3. **Robustețe pe mai multe seed-uri.** Curent: 5 seed-uri. Util de testat 10 seed-uri pentru test (ca în protocolul CSNN) pentru a întări claim-ul statistic.
4. **Calibrare a încrederii (temperature scaling)** pe ensemble — pentru ca „X% confidență" din Streamlit să fie semnificativă numeric.
5. **Transfer cross-scenariu**: train pe d1+d2+d3 și test pe d4 — robustețe la modificarea fundalului/îmbrăcămintei.

---

## 14. Sugestii de figuri

| # | Conținut | Loc în lucrare |
|--:|----------|----------------|
| 1 | Schema completă a pipeline-ului (video → JSON → NPZ → tensor → CNN → ensemble) | Începutul capitolului |
| 2 | O secvență de T = 19 frame-uri (pas g = 2) cu bbox-ul desenat | Secțiunea bbox-uri |
| 3 | Vizualizare HOG (crop persoană + harta gradienților + suprapunerea celulelor 8 × 8) | Subsecțiunea HOG |
| 4 | Mozaic de variante augmentate (orig + flip + 4 jit) pentru același sample | Subsecțiunea augmentare |
| 5 | Schema celor 80 canale (HOG + diff + bbox + bbox_vel) | Subsecțiunea fluxuri |
| 6 | Diagrama arhitecturii `HARConv3DNet` cu volume tensoriale (80×19×15×7 → … → 6) | Subsecțiunea arhitectură |
| 7 | Curbele train/val loss + accuracy pentru un seed reprezentativ | Subsecțiunea antrenare |
| 8 | Screenshot al UI-ului Streamlit cu un exemplu clasificat | Subsecțiunea demonstrator |
| 9 | Bar chart acuratețe per seed individual vs ensemble | Subsecțiunea rezultate |
| 10 | Matricea de confuzie pe test ca heatmap | Subsecțiunea rezultate |

---

## Anexă A — Reproducerea rezultatelor

```bash
# 1) Bbox-uri (CSNN-side; doar dacă JSON-ul nu există deja). T=19, frame_gap=2:
python3 src/tool/extract_bboxes_kth.py \
    --input_path /home/mmuntean/kth_organized_tvt/ \
    --temporal_kernel 19 --frame_gap 2 --num_groups 10 \
    --output hog/hog_person_data_tvt_19_f10_g2.json

# 1b) Fix running: re-extracție doar running cu praguri permisive + carry mai mare,
#     merge în JSON-ul de la pasul 1 → JSON-ul "runfix" canonic. Vezi 11.4.
python3 src/tool/extract_bboxes_kth.py \
    --input_path /home/mmuntean/kth_organized_tvt/ \
    --temporal_kernel 19 --frame_gap 2 --num_groups 10 \
    --frame_width 160 --frame_height 120 \
    --only_action running \
    --hit_threshold -1.2 --min_bbox_area_ratio 0.004 \
    --min_bbox_aspect 0.15 --max_bbox_aspect 2.0 \
    --mog2_min_area 300 --max_carry 4 \
    --merge_into hog/hog_person_data_tvt_19_f10_g2.json \
    --output    hog/hog_person_data_tvt_19_f10_g2_runfix.json
# (pașii 1b–4 sunt automatizați în cnn_har_app/reextract_running.sh)

# 2) HOG augmentat (rulat din cnn_har_app/):
cd cnn_har_app
python3 extract_hog_augmented.py \
    --bbox_json ../hog/hog_person_data_tvt_19_f10_g2_runfix.json \
    --output   ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --num_aug 8 --aug_profile strong \
    --video_root /home/mmuntean/kth_organized_tvt
# verifică: features shape: (14715, 71820)

# 3) Antrenare 5 seed-uri (best pe val, raportare pe test; FĂRĂ sampler):
for s in 42 123 7 13 99; do
  python3 train.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --model_type conv3d \
    --balanced_sampler none \
    --seed $s \
    --save_suffix _tvt19fix_s$s \
    --temporal_reverse_p 0.3 \
    --temporal_shift_max 2 \
    --ema_decay 0.999 \
    --ema_start 5 \
    2>&1 | tee ../data/log_cnn_tvt19fix_s$s.txt
done

# 4) Evaluare în ensemble cu TTA:
python3 eval_ensemble.py \
    --data_path ../hog/hog_aug_tvt_19_f10_g2_runfix.npz \
    --model_type conv3d \
    --checkpoints "models/har_conv3d_tvt19fix_s*.pth" \
    --ensemble_mode logits \
    --tta_reverse \
    --tta_shift 1

# 5) Media ± std pe test din loguri (comparația cu CSNN):
python3 - <<'PY'
import re, glob, statistics
accs=[]
for f in sorted(glob.glob("../data/log_cnn_tvt19fix_s*.txt")):
    m=re.findall(r"Final test accuracy:\s*([\d.]+)%", open(f).read())
    if m: accs.append(float(m[-1])); print(f.split('/')[-1], m[-1]+'%')
if len(accs)>=2:
    print(f"TEST: {statistics.mean(accs):.2f}% +- {statistics.stdev(accs):.2f} (n={len(accs)})")
PY

# 6) Demonstrator interactiv:
streamlit run app.py
```

---

## Anexă B — Glosar rapid

| Termen | Semnificație |
|--------|--------------|
| **HAR** | Human Action Recognition |
| **HOG** | Histogram of Oriented Gradients |
| **HOG3D** | Extensia HOG la spațiu-timp (Kläser et al., 2008) |
| **bbox** | Bounding box, dreptunghiul ce încadrează persoana |
| **frame_gap (g)** | Distanța în frame-uri între eșantioanele consecutive dintr-un clip (curent: 2) |
| **num_groups** | Câte ferestre de T frame-uri se extrag per video (curent: 10) |
| **temporal_kernel (T)** | Numărul de frame-uri folosite pentru un eșantion (curent: 19) |
| **tvt** | Split train / val / test pe subiecți (8 / 8 / 9) |
| **EMA** | Exponential Moving Average (peste greutățile modelului) |
| **TTA** | Test-Time Augmentation |
| **Mixup** | Tehnică ce combină liniar perechi de eșantioane și etichete |
| **Label smoothing** | Înlocuiește one-hot $(1, 0, \dots)$ cu $(1-\varepsilon,\ \varepsilon/(K-1), \dots)$ |
| **ROCm** | Stack-ul AMD pentru GPU compute (echivalent CUDA); MI210 = gfx90a |
| **CSNN** | Convolutional Spiking Neural Network |
| **STDP** | Spike-Timing-Dependent Plasticity |
| **LIF** | Leaky Integrate-and-Fire (model de neuron) |
