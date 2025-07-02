This is an AI-assisted project. The AI agent (Jules) is responsible for:
1.  Creating the directory structure.
2.  Implementing all Python code for preprocessing, model training, API, and dashboard.
3.  Writing unit tests.
4.  Creating a Dockerfile.
5.  Writing documentation (`README.md`).

Instructions for the Agent:
-   Follow the plan provided.
-   Organize code into modules as specified in the plan (`src/preprocessing.py`, `src/models.py`, etc.).
-   Use FastAPI for the API.
-   Use Streamlit for the dashboard.
-   Ensure all code is well-commented.
-   Write clear instructions in the `README.md` for setup and execution.
-   For features like "Customizable Sentiment Sensitivity Engine" or "Domain Adaptation Layer", if a full implementation is too complex within the scope, provide a clear conceptual explanation and a basic functional implementation.
-   Prioritize clarity and correctness.
-   Use Python 3.8+ features where appropriate.
-   Add all dependencies to `requirements.txt`.
-   If you need to download datasets or pre-trained models, try to use libraries like Hugging Face `datasets` or `transformers`, or `nltk.download()` to make it easier for the user. Clearly document these steps.
