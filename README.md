# Google Photos Review Analyser

An LLM-powered review analysis project that analyses Google Photos Play Store reviews to identify customer sentiment, common complaints, feature requests and product improvement opportunities.

The project combines Large Language Model (LLM) classification, exploratory data analysis, an interactive Streamlit dashboard and a Gemini-powered AI chatbot.

## Live Application

**Streamlit Dashboard:** https://es4h8n9p9pj7uxwngfjucl.streamlit.app/

## Project Overview

Build a review analyser. It reads each review, uses an LLM to work out its sentiment, category and topic, and shows the result in a dashboard that summarises all the reviews: how many are good, how many are bad, and what they are about.

- Every review goes to the LLM, including one-word reviews. The work is in the prompt: you engineer it until the LLM gives the - right labels.
- You may use a self-hosted LLM (for example Ollama with llama3.1:8b) or an LLM API (any free tier is enough).
- The code that calls the LLM must be in your submission.
- The dashboard can be a static HTML page or a dynamic app (Streamlit, Dash, Gradio). Your choice.

## Key Features
- **LLM-based Review Classification:** Classifies reviews based on sentiment, category and topic.
- **Sentiment Analysis:** Identifies positive, neutral and negative user experiences.
- **Category Analysis:** Distinguishes bug reports, feature requests, opinions and rating-related feedback.
- **Topic Analysis:** Identifies the product areas discussed in reviews.
- **Manual Evaluation:** Compares LLM predictions against 50 manually labelled reviews.
- **Interactive Dashboard:** Provides charts, filters, review exploration and summary statistics.
- **AI Chatbot:** Allows users to ask natural-language questions about the analysed reviews and receive answers using Gemini.
- **Product Insights:** Highlights important complaints, helpful reviews and developer response patterns.

## Classification Definitions

### 1. Sentiment

Identifies the emotional tone expressed in a review.

| Sentiment | Definition |
|---|---|
| Happy | Strongly positive experience or excitement. |
| Satisfied | Generally positive experience or satisfaction. |
| Neutral | Neither clearly positive nor negative. |
| Negative | Expresses dissatisfaction or criticism. |
| Frustrated | Shows annoyance due to repeated problems or inconvenience. |
| Angry | Expresses strong anger or serious dissatisfaction. |

For broader analysis, the six sentiments are grouped into three categories:

- **Good:** Happy, Satisfied
- **Neutral:** Neutral
- **Bad:** Negative, Frustrated, Angry

### 2. Category

Identifies the nature of the feedback.

| Category | Definition |
|---|---|
| bug | Reports a technical issue, malfunction or unexpected behaviour. |
| feature_request | Suggests a new feature or improvement. |
| opinion | Expresses a general opinion, preference or experience. |
| reflect_rating | Review text mainly reflects or explains the given star rating. |

### 3. Topic

Identifies the Google Photos feature or area discussed in the review.

| Topic | Definition |
|---|---|
| Backup_Sync | Backup, syncing and photo storage synchronization. |
| Editing | Photo and video editing features. |
| Sharing_Albums | Sharing, albums and collaborative features. |
| UI_Design | Interface, layout and navigation. |
| Search | Searching and finding photos or videos. |
| Memories | Memories and automatic photo collections. |
| Storage_Plans | Storage limits, Google One and subscription plans. |
| App_Crash | Application crashes and unexpected closures. |
| Performance | Speed, loading and responsiveness. |
| Privacy_Security | Privacy, permissions and security concerns. |
| Device_Support | Device compatibility and platform-specific issues. |
| AI_Features | AI-powered features and functionality. |
| Positive_Comment | General positive feedback without a more specific topic. |
| Not_Applicable | Reviews that do not fit the available topics. |

## Technologies Used

- **Python** – Data processing and analysis.
- **Pandas** – Data manipulation and analysis.
- **Ollama (Llama 3.2 3B)** – LLM-based review classification.
- **Jupyter Notebook** – Experimentation, prompt engineering and evaluation.
- **Streamlit** – Interactive dashboard development.
- **Plotly** – Interactive charts and visualizations.
- **Google Gemini API** – AI-powered chatbot.
- **Git and GitHub** – Version control and project hosting.

## Project Structure

```text
Google-Photo-Review-Analyser/
│
├── Review_Analyser_Starter.ipynb
├── google_photos_reviews.csv
├── labelled_reviews.csv
├── manual_labels_50.csv
├── expected_output_examples.csv
├── requirements.txt
│
├── streamlit_app/
│   ├── analytics.py
│   ├── chatbot.py
│   └── dashboard.py
│
└── README.md
```

The notebook contains the review analysis, classification prompts, experiments, evaluation and observations. The Streamlit application contains the dashboard and chatbot implementation.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/vishal0205/Google-Photo-Review-Analyser-.git
cd Google-Photo-Review-Analyser-
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```


### 5. Run the Streamlit application

```bash
streamlit run streamlit_app/dashboard.py
```

The application will open in your browser.

**Note:** The Gemini API key is required for chatbot functionality. The LLM classification experiments in the notebook use Ollama and Llama 3.2 (3B), which must be installed separately to reproduce those experiments.

## AI Tool Usage and Acknowledgement

AI tools were used as development assistance during different stages of the project.

- **Prompt Engineering (Step 4):** ChatGPT assisted in drafting, refining and improving the LLM classification prompt.
- **Classification Definitions:** ChatGPT assisted in formulating and refining the definitions and guidelines for sentiment, category and topic classification.
- **Code Development:** ChatGPT assisted with Python implementation, debugging and improving parts of the analysis and Streamlit application.
- **Dashboard and Chatbot:** ChatGPT assisted in developing the Streamlit dashboard and integrating the Gemini-powered chatbot.
- **Review Classification:** Ollama running Llama 3.2 (3B) performed the review classification.
- **Chatbot Responses:** Google Gemini API generates responses to user questions using relevant review information retrieved by the application.

AI-generated suggestions and code were reviewed and integrated during development.

## Evaluation

A manually labelled set of 50 reviews was used to evaluate the LLM classification results.

| Classification | Accuracy |
|---|---:|
| Sentiment | 64% |
| Category | 60% |
| Topic | 74% |

These results provide an indication of classification performance on the manually evaluated sample.

## Future Improvements

- Improve classification accuracy through further prompt refinement.
- Expand manual evaluation with a larger sample.
- Improve chatbot retrieval and response relevance.
- Investigate recurring complaints and their technical root causes.
- Extend analysis to a larger portion of the original dataset.

---