# 👗 TailorTalk --- AI-Powered Visual Saree Search

> **Upload a saree. Ask naturally. Discover visually similar products.**

TailorTalk is an end-to-end AI visual search agent for fine-grained
fashion retrieval. Given a saree image, it combines a **fashion-specific
vision encoder**, **vector similarity search**, and an **LLM
tool-calling agent** to return visually similar catalogue products with
similarity scores, prices, and product links.

## 🚀 Live Demo

**https://saree-search-engine.streamlit.app/**

## 💻 GitHub Repository

**https://github.com/younome71/tailortalk-visual-search**

> **The application is already deployed and can be tested directly from
> the live URL. No local setup is required for the reviewer.**

------------------------------------------------------------------------

## 🎯 What This Project Solves

This is not a generic image-search problem.

The supplied catalogue consists entirely of sarees, meaning the useful
differences are often subtle:

-   🎨 Colour and colour combinations
-   🧵 Fabric and weave
-   🪡 Border and pallu work
-   🖼️ Prints and motifs
-   👗 Overall design and composition

A basic embedding model can therefore return visually plausible but
insufficiently fine-grained results.

TailorTalk was designed around this challenge: **retrieve sarees that
are visually close to the query, rather than merely retrieving other
sarees.**

------------------------------------------------------------------------

# ✨ Key Features

-   📤 **Image upload** --- JPG, JPEG, PNG and WEBP
-   🔗 **Image URL input** --- search using a direct image URL
-   💬 **Natural-language chat** --- ask for similar or alternative
    sarees conversationally
-   🤖 **LLM agent with function calling** --- the agent decides when
    visual search is required
-   👗 **Fashion-specific visual embeddings** --- Marqo-FashionCLIP
-   ⚡ **FAISS vector search** --- fast nearest-neighbour retrieval
-   📊 **Similarity scores** --- transparent retrieval ranking
-   🛍️ **Product cards** --- image, product name, price and product-page
    link
-   ☁️ **Live deployment** --- works directly in the browser with no
    local setup

------------------------------------------------------------------------

# 🧠 Architecture

``` mermaid
flowchart TD
    A[User] --> B[Streamlit Frontend]
    B --> C[Gemini Agent]
    C -->|Function Call| D[search_similar_sarees]
    D --> E[SareeSearcher]
    E --> F[Marqo-FashionCLIP]
    F --> G[512-D Normalized Embedding]
    G --> H[FAISS Index]
    H --> I[Top-K Visual Matches]
    I --> J[Catalogue Metadata]
    J --> K[Results + Scores + Product Data]
    K --> C
    C --> B
```

### End-to-end retrieval flow

``` text
Query Image
     │
     ▼
Marqo-FashionCLIP
     │
     ▼
512-dimensional normalized embedding
     │
     ▼
FAISS nearest-neighbour search
     │
     ▼
Top-K visually similar catalogue images
     │
     ▼
Product metadata enrichment
     │
     ▼
Gemini tool result
     │
     ▼
Natural-language response
     │
     ▼
Streamlit product cards
```

------------------------------------------------------------------------

# 🏗️ Technology Choices

  -----------------------------------------------------------------------
  Component               Choice                  Why
  ----------------------- ----------------------- -----------------------
  Frontend                **Streamlit**           Interactive Python UI
                                                  and simple cloud
                                                  deployment

  Agent / LLM             **Gemini**              Natural-language
                                                  understanding and
                                                  function/tool calling

  Vision model            **Marqo-FashionCLIP**   Fashion-specific visual
                                                  representation

  Vector database         **FAISS**               Simple, fast and
                                                  reproducible vector
                                                  search for this
                                                  catalogue size

  Metadata                **Pandas + CSV**        Lightweight catalogue
                                                  metadata management

  Image processing        **Pillow**              Image loading and
                                                  preprocessing

  Language                **Python**              Single-language
                                                  implementation

  Deployment              **Streamlit Community   Public, zero-setup
                          Cloud**                 reviewer experience
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🔍 Search Quality: What Was Improved?

Search quality was the primary focus of the implementation.

## 1. Started with a generic CLIP baseline

The initial system used:

``` text
OpenAI CLIP
      ↓
512-D image embeddings
      ↓
FAISS
      ↓
Nearest neighbours
```

This worked as a baseline, but the results were often too generic for
this dataset. Since every image is a saree, generic visual
representations can over-emphasize broad similarities while missing
fashion-specific details.

## 2. Switched to FashionCLIP

The retrieval model was upgraded to:

``` text
Marqo-FashionCLIP
```

This is a fashion-specific CLIP model and is a better fit for
distinguishing attributes such as:

-   colour
-   material
-   style
-   patterns
-   garment-level visual details

The final embedding pipeline produces:

``` text
1069 images × 512 dimensions
```

with L2-normalized vectors.

## 3. Used cosine-style similarity through normalized FAISS vectors

Because embeddings are normalized, the FAISS inner-product search
corresponds to cosine similarity:

``` text
similarity(query, item) = query · item
```

This gives a simple and interpretable ranking signal.

## 4. Evaluated before finalizing

A fixed set of **20 query images** was used during development.

Each query was evaluated against the catalogue and the top 10 results
were recorded.

The final FashionCLIP evaluation produced:

``` text
Queries evaluated : 20
Results per query : 10
Total results     : 200

Average score     : 0.8463
Median score      : 0.8470
```

> **Important:** similarity score is a retrieval similarity signal, not
> an accuracy percentage. The values above should not be interpreted as
> 84.63% retrieval accuracy.

## Why the final system does not rely on a complicated reranker

A DINOv2-based reranking experiment was explored during development.

For the final submission, the system deliberately favours the simpler
and widely used approach:

``` text
Fashion-specific embedding
        +
FAISS similarity search
```

rather than introducing a manually tuned combination of unrelated
similarity signals.

This keeps the final retrieval pipeline:

-   deterministic
-   explainable
-   reproducible
-   easier to deploy
-   less sensitive to manually chosen ranking weights

The experiments are retained in the repository for transparency.

------------------------------------------------------------------------

# 🤖 AI Agent & Tool Calling

The LLM is not responsible for calculating visual similarity.

Instead, Gemini acts as the conversational orchestration layer.

It has access to a clearly defined tool:

``` text
search_similar_sarees
```

with a structured parameter:

``` json
{
  "top_k": 1
}
```

The actual tool schema constrains `top_k` to the supported range.

### Example interaction

**User**

> Find sarees similar to this one.

**Gemini**

``` text
search_similar_sarees(top_k=5)
```

**Python tool**

``` text
Uploaded image
      ↓
FashionCLIP
      ↓
FAISS
      ↓
Top 5 matches
```

**Gemini**

Turns the structured tool result into a natural response.

This separation is intentional:

``` text
Gemini
  → intent + tool selection + conversation

FashionCLIP
  → visual representation

FAISS
  → deterministic retrieval
```

The result is a genuine tool-using agent rather than a chatbot that
merely describes the search system.

------------------------------------------------------------------------

# 📦 Dataset & Index

The supplied catalogue contains:

``` text
Catalogue rows : 1074
Unique SKUs    : 655
Valid images   : 1069
Embedding size : 512
FAISS vectors  : 1069
```

There are duplicate SKUs in the supplied catalogue, so catalogue rows
and unique SKU counts are intentionally treated as different concepts.

The five unavailable images are retained in catalogue metadata but
excluded from the searchable image index.

The deployed application ships with the precomputed FAISS index so that
the reviewer does **not** need to rebuild embeddings at runtime.

------------------------------------------------------------------------

# 🖥️ Frontend Experience

The application is designed to make the retrieval system immediately
understandable to a reviewer.

### Input

Users can either:

1.  upload an image, or
2.  provide an image URL.

### Search

They can then ask naturally:

``` text
Find sarees similar to this.

Show me alternatives to this saree.

Can you find sarees like this one?

Find visually similar options.
```

### Results

Each result can display:

-   product image
-   ranking position
-   product name
-   visual similarity score
-   discounted price
-   original price
-   product-page link

The structured result cards are rendered by Streamlit rather than
relying solely on LLM-generated text, ensuring that the product
information shown to the user comes from the catalogue retrieval
pipeline.

------------------------------------------------------------------------

# 📊 Example

For a query image corresponding to:

``` text
Kathan Banaras Saree Deep Purple AA205129
```

the system retrieved visually close catalogue variants such as:

``` text
1. Kathan Banaras Saree Deep Purple AA205129
   Similarity: 1.0000

2. Kathan Banaras Saree Royal Purple AA205129
   Similarity: 0.9221

3. Kathan Banaras Saree Coffee Brown AA205129
   Similarity: 0.9188

4. Kathan Banaras Saree Light Maroon AA205129
   Similarity: 0.8978

5. Kathan Banaras Saree Olive Green AA205129
   Similarity: 0.8900
```

This demonstrates the intended fine-grained behaviour: the results
remain within the same visual/product family rather than simply
returning arbitrary sarees.

------------------------------------------------------------------------

# 📁 Project Structure

``` text
tailortalk-visual-search/
│
├── app.py
│
├── src/
│   ├── agent.py
│   ├── embeddings.py
│   ├── search.py
│   ├── ranking.py
│   └── reranker.py
│
├── scripts/
│   ├── download_images.py
│   ├── validate_images.py
│   ├── create_metadata.py
│   ├── build_index.py
│   ├── test_similarity.py
│   ├── evaluate_fashionclip.py
│   └── test_agent.py
│
├── data/
│   ├── products.csv
│   ├── image_metadata.csv
│   └── images/
│
├── index/
│   ├── embeddings.npy
│   └── sarees.faiss
│
├── evaluation/
│   └── fashionclip/
│
├── requirements.txt
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

# ⚙️ Local Setup

The live application requires **no local setup**.

If you want to run the project locally:

## 1. Clone the repository

``` bash
git clone https://github.com/younome71/tailortalk-visual-search.git
cd tailortalk-visual-search
```

## 2. Create a virtual environment

### Windows

``` powershell
python -m venv .venv
.venv\Scriptsctivate
```

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

``` bash
pip install -r requirements.txt
```

## 4. Configure Gemini

Create a local `.env` file:

``` env
GEMINI_API_KEY=your_api_key_here
```

**Never commit `.env` to GitHub.**

## 5. Run

``` bash
streamlit run app.py
```

Open:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# 🧪 Development & Evaluation Commands

### Validate catalogue images

``` bash
python scripts/validate_images.py
```

### Generate metadata

``` bash
python scripts/create_metadata.py
```

### Build the embedding/index pipeline

``` bash
python scripts/build_index.py
```

### Test similarity search

``` bash
python scripts/test_similarity.py
```

### Evaluate FashionCLIP retrieval

``` bash
python scripts/evaluate_fashionclip.py
```

### Test the Gemini agent

``` bash
python -m scripts.test_agent
```

------------------------------------------------------------------------

# ☁️ Deployment

TailorTalk is deployed on **Streamlit Community Cloud**.

### Production entrypoint

``` text
app.py
```

### Live application

**https://saree-search-engine.streamlit.app/**

The deployed application is designed to work **out of the box**:

``` text
Open URL
   ↓
Upload image
   ↓
Ask for similar sarees
   ↓
Receive results
```

The reviewer does not need:

-   Python
-   a virtual environment
-   the dataset locally
-   FAISS installation
-   a Gemini API key
-   any model setup

The Gemini API credential is configured privately through deployment
secrets.

------------------------------------------------------------------------

# 🔐 Security

Secrets are not stored in the repository.

Local credentials are kept in:

``` text
.env
```

and deployment credentials are supplied through Streamlit's
secret-management system.

The `.gitignore` excludes local environments and secrets:

``` text
.venv/
.env
__pycache__/
*.pyc
```

------------------------------------------------------------------------

# ⚖️ Assumptions & Trade-offs

## FAISS instead of a hosted vector database

**Chosen:** FAISS

For a catalogue of 1,069 searchable images, FAISS is sufficient and
avoids unnecessary external infrastructure.

### Advantages

-   simple
-   fast
-   deterministic
-   inexpensive
-   easy to reproduce
-   easy to deploy with the precomputed index

### Trade-off

A hosted vector database such as Qdrant or Pinecone would be more
appropriate for a much larger, frequently changing catalogue.

------------------------------------------------------------------------

## Precomputed embeddings

The catalogue embeddings are generated offline and stored with the
application.

This avoids recomputing 1,069 embeddings every time the application
starts.

### Trade-off

If the catalogue changes, the embedding matrix and FAISS index need to
be rebuilt.

------------------------------------------------------------------------

## FashionCLIP rather than a generic vision model

The model was chosen specifically because the problem is fine-grained
fashion retrieval.

### Trade-off

A larger or more specialized retrieval model could potentially improve
quality further, but would increase model size, inference cost and
deployment complexity.

------------------------------------------------------------------------

## Deterministic retrieval rather than aggressive reranking

The final ranking relies primarily on FashionCLIP + FAISS rather than a
heavily hand-tuned ensemble.

This favours reproducibility and robustness over potentially overfitting
a small evaluation set.

------------------------------------------------------------------------

# 🚀 Future Improvements

If this system were taken beyond the assignment, the next improvements
would include:

### 1. Larger-scale vector infrastructure

Move from local FAISS to a managed or distributed vector database for
continuously changing catalogues.

### 2. Multimodal filtering

Combine visual similarity with structured attributes such as:

``` text
colour
fabric
price range
occasion
weave
```

### 3. Attribute-aware reranking

Use a dedicated fashion retrieval/reranking model trained on
product-level similarity rather than manually combining generic visual
features.

### 4. Conversational refinement

Allow users to refine searches conversationally:

> "Show me the same style but in pink."

### 5. Personalization

Rank visually similar products according to user preferences, browsing
history and price range.

------------------------------------------------------------------------

# 🏆 Assignment Requirements --- Covered

  Requirement                              TailorTalk
  ---------------------------------------- ------------------------------
  Process the supplied image dataset       ✅
  Create image embeddings                  ✅ FashionCLIP
  Vector database/search                   ✅ FAISS
  Agent framework / function calling       ✅ Gemini tool calling
  Natural conversational interface         ✅
  Image upload                             ✅
  Image URL input                          ✅
  Similarity scores                        ✅
  Improve search quality                   ✅ FashionCLIP + evaluation
  Streamlit / Gradio frontend              ✅ Streamlit
  Deploy publicly                          ✅ Streamlit Community Cloud
  No reviewer-side local setup             ✅
  Working app URL                          ✅
  GitHub source code                       ✅
  README with setup                        ✅
  Model choices documented                 ✅
  Vector DB choice documented              ✅
  Framework choices documented             ✅
  Search-quality improvements documented   ✅
  Assumptions and trade-offs documented    ✅

------------------------------------------------------------------------

# 🔗 Submission Links

### 🌐 Live Application

**https://saree-search-engine.streamlit.app/**

### 💻 GitHub Repository

**https://github.com/younome71/tailortalk-visual-search**

------------------------------------------------------------------------

## Built with

**Python · Streamlit · Gemini · FashionCLIP · FAISS · PyTorch**

> **TailorTalk --- turning a fashion image into a conversation about
> what to discover next.**
