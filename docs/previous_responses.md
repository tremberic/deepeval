# DeepEval on Azure with Databricks

## 1. What you’re setting up

DeepEval will sit alongside your Databricks-hosted RAG chatbot to (a) collect retrieved context plus answers, and (b) score them with research-backed metrics. Installation is just a Python package, but you’ll want Azure resources (Azure OpenAI for evaluator models, Key Vault/Secret Scopes for secrets, etc.) so everything stays in your subscription.

---

## 2. Azure prerequisites

1. **Azure OpenAI resource** with at least one chat deployment (e.g., `gpt-4o`, `gpt-4.1`) for DeepEval’s judge model. You’ll need the endpoint URL, deployment name, API version, and key.
2. **Optional embeddings deployment** (text-embedding-3-* or similar) if you plan to generate synthetic test data with DeepEval’s synthesizer.
3. **Secret storage** (Azure Key Vault or Databricks secret scope) to hold the Azure OpenAI key and, if you use Confident AI, its API key.
4. **Databricks workspace/cluster** with Python ≥3.9; DeepEval itself is a pure Python package.

---

## 3. Install DeepEval in Databricks

1. On your cluster (or as part of a job), run:

   ```bash
   pip install -U deepeval
   ```

   DeepEval supports Python 3.9+ and can be included in cluster init scripts or notebook `%pip install` cells.

2. Configure environment loading:
   - DeepEval auto-loads `.env.local` then `.env` from the current working directory at import time (process env vars take precedence). Set `DEEPEVAL_DISABLE_DOTENV=1` if you prefer manual control.

3. In Databricks, expose secrets as environment variables for the job or notebook (e.g., `AZURE_OPENAI_KEY`, `CONFIDENT_API_KEY`).

---

## 4. Authenticate with DeepEval and Confident AI (optional)

1. Run `deepeval login` locally or inside Databricks with your Confident AI key to sync run history to the SaaS dashboard (highly recommended for sharing reports).
2. For headless Databricks jobs, use `deepeval login --confident-api-key <key> --save=dotenv` so credentials are written to `.env.local` (ensure it stays secret-scoped). The CLI can persist or clear credentials via `deepeval logout`.

---

## 5. Configure Azure OpenAI as the judge model

1. Set Azure OpenAI globally via CLI (the `--save` flag writes values to your dotenv file, useful on Databricks images that restart):

   ```bash
   deepeval set-azure-openai \
       --openai-endpoint https://<your-resource>.openai.azure.com/ \
       --openai-api-key $AZURE_OPENAI_KEY \
       --openai-model-name gpt-4o \
       --deployment-name <deployment> \
       --openai-api-version 2024-06-01 \
       --save=dotenv
   ```

   To undo: `deepeval unset-azure-openai --save=dotenv`.

2. If you also need Azure embeddings (for synthetic data generation), run:

   ```bash
   deepeval set-azure-openai-embedding --embedding-deployment-name <embedding_deployment> --save=dotenv
   ```

   This ensures both LLM and embedding calls go through Azure.

3. Prefer to hardcode in Python? Instantiate `AzureOpenAIModel` directly inside notebooks or jobs—handy when you need different models per metric:

   ```python
   from deepeval.models import AzureOpenAIModel
   from deepeval.metrics import AnswerRelevancyMetric

   model = AzureOpenAIModel(
       model_name="gpt-4o",
       deployment_name="<deployment>",
       azure_openai_api_key=os.environ["AZURE_OPENAI_KEY"],
       openai_api_version="2024-06-01",
       azure_endpoint="https://<resource>.openai.azure.com/",
       temperature=0,
   )
   answer_relevancy = AnswerRelevancyMetric(model=model)
   ```

   Required parameters are endpoint, deployment, model name, API key, and API version.

---

## 6. Instrument your Databricks RAG pipeline

1. Make sure the pipeline returns both the generated answer and the retrieved context chunks—DeepEval metrics like Answer Relevancy and Faithfulness need them.
2. If refactoring is hard, you can wrap retrieval/generation calls with DeepEval’s `@observe` decorator to capture spans without major code changes:

   ```python
   from deepeval.tracing import observe, update_current_span
   from deepeval.metrics import GEval
   from deepeval.test_case import LLMTestCase, LLMTestCaseParams
   from deepeval import evaluate

   correctness = GEval(
       name="Correctness",
       criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
       evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
   )

   @observe(metrics=[correctness])
   def generate_answer(prompt, retrieved_docs):
       response = call_your_chain(prompt, retrieved_docs)
       update_current_span(
           test_case=LLMTestCase(
               input=prompt,
               actual_output=response.answer,
               retrieval_context=retrieved_docs,
           )
       )
       return response.answer
   ```

   The `@observe` pattern lets you evaluate individual components (retriever, generator, tools) separately.

---

## 7. Author DeepEval tests for your chatbot

1. **Simple end-to-end pytest** (great for Databricks jobs):

   ```python
   import pytest
   from deepeval import assert_test
   from deepeval.metrics import AnswerRelevancyMetric
   from deepeval.test_case import LLMTestCase

   @pytest.mark.parametrize(
       "test_case",
       [
           LLMTestCase(
               input="Where is my order?",
               actual_output=chatbot("Where is my order?")[0],
               retrieval_context=chatbot("Where is my order?")[1],
           ),
       ],
   )
   def test_customer_support(test_case):
       metric = AnswerRelevancyMetric(threshold=0.7)
       assert_test(test_case, [metric])
   ```

   Run it via a Databricks job or notebook cell: `deepeval test run test_chatbot.py`.

2. **Notebook-friendly evaluation** (no pytest):

   ```python
   from deepeval import evaluate
   from deepeval.metrics import AnswerRelevancyMetric
   from deepeval.test_case import LLMTestCase

   metric = AnswerRelevancyMetric(threshold=0.7)
   test_case = LLMTestCase(
       input="Where is my order?",
       actual_output=answer,
       retrieval_context=retrieved_chunks,
   )
   evaluate([test_case], [metric])
   ```

   This is handy when iterating inside Databricks notebooks.

3. **Bulk datasets** – load evaluation datasets and run parametrized pytest or `dataset.evaluate([...])` to score many prompts at once.

---

## 8. Running evaluations in Azure/Databricks

1. Schedule Databricks jobs that call `deepeval test run ...` so evaluations become part of CI/CD (e.g., nightly). DeepEval CLI supports parallel test execution (`-n 4`) if you have multiple test cases.
2. The CLI loads dotenv files plus process env vars, so ensure your job passes `AZURE_OPENAI_KEY`, `OPENAI_ENDPOINT`, etc. via environment configuration or secret scopes.
3. Use `deepeval view` after runs to open the latest report in Confident AI (uploads artifacts automatically).

---

## 9. Optional: synthetic data & embeddings

If you want to bootstrap evaluation datasets from documentation or logs:

1. Configure Azure embeddings globally with `set-azure-openai-embedding` (see §5) so the Synthesizer uses your Azure deployment.
2. Or implement a custom embedding class inheriting `DeepEvalBaseEmbeddingModel` if you need bespoke credential handling or reuse of LangChain helpers inside Databricks.

---

## 10. Observability and monitoring

1. DeepEval integrates tightly with Confident AI for hosting reports, comparing runs, and production monitoring; you can push evaluation results there automatically once logged in.
2. To capture live traffic for monitoring, wrap key RAG components with `@observe` and feed production interactions as `Golden` test cases later (bulk evaluation example in §7).
3. The CLI provides debug controls (`deepeval set-debug`) to turn on verbose logging and gRPC traces—useful when diagnosing Azure networking issues from Databricks clusters.

---

# Running DeepEval on Azure resources outside Databricks

- DeepEval is distributed as a standard Python package (`pip install -U deepeval`) and only requires Python 3.9 or later, so you can install it on any Azure-hosted compute such as Azure VMs, Azure Container Apps, or Azure Machine Learning compute clusters in exactly the same way you would on a local machine. 【F:README.md†L140-L200】
- The framework runs locally and can drive evaluations with whichever LLM endpoints you configure, including Azure OpenAI; you can point DeepEval’s metrics at your Azure OpenAI deployments via the CLI or Python SDK helpers without needing Databricks. 【F:README.md†L73-L101】【F:docs/static/llms-full.txt†L7906-L7963】
- Choosing between Databricks and standalone Azure services usually comes down to operational convenience versus cost control: Databricks adds managed Spark, notebooks, and job orchestration, whereas running on general Azure compute means you manage the runtime yourself but only pay for the underlying VM/container usage. (This trade-off is based on how Azure services are priced and orchestrated; select the option that matches your team’s existing tooling and budget oversight.)

---

### Putting it all together on Azure

1. **Provision** Azure OpenAI + Databricks.
2. **Secure** secrets in Key Vault or Databricks secret scopes.
3. **Install** DeepEval via pip on the cluster/job.
4. **Login/configure** DeepEval CLI with Confident AI (optional) and Azure OpenAI.
5. **Instrument** your RAG pipeline to expose retrieved context.
6. **Write tests** (pytest or notebook) that construct `LLMTestCase` objects.
7. **Schedule jobs** in Databricks to run `deepeval test run ...` and stream results to Confident AI.
8. **Iterate** by adding more metrics (faithfulness, hallucination, DAG) as you mature the evaluation suite.

Following the above will give you a repeatable evaluation and monitoring loop entirely within your Azure environment while keeping secrets and compute on your subscription.

---

# Confident AI Benefits and Alternatives

## Benefits of Confident AI with DeepEval
* **Centralized evaluation lifecycle:** Confident AI layers dashboards, dataset curation, metric tuning, and debugging traces on top of DeepEval so you can monitor regressions, compare runs, and analyze failures beyond raw metric scores.
* **Keeps results organized:** It hosts your evaluation reports in the cloud, making it easier to revisit historical runs, collaborate with teammates, and iterate on prompts or retrievers without copying artifacts around manually.
* **Fits into existing DeepEval workflows:** You log in from the CLI/notebook, run the same `deepeval test run ...` commands, and a shareable link appears—no extra orchestration needed once configured.

## Pricing & Whether It’s Required
* **Free to start:** Multiple sections of the documentation emphasize that creating a Confident AI account costs nothing initially (no credit card required).
* **Optional add-on:** DeepEval itself remains fully open-source; you can run all metrics, tests, and CLI commands locally without ever connecting to Confident AI if you prefer to keep everything in Databricks or Azure storage. Confident AI simply provides extra telemetry, collaboration, and monitoring features on top of the same evaluations.

## What Happens Without Confident AI?
* You can still achieve “good results” in terms of metric scoring, regression testing, and CI/CD automation because all evaluation logic runs locally. The trade-off is that you’ll have to manage storage, reporting, and comparison of historical runs yourself (e.g., persist artifacts in your own databases or dashboards) instead of using the hosted UI/observability features Confident AI supplies.

## 100% Free Alternatives
If you want a purely open-source stack with no hosted services, you can combine DeepEval with other OSS observability tools (e.g., self-hosted MLflow, Superset, or plain notebooks) to visualize metrics. Other evaluation frameworks like Ragas or TruLens are also open-source, but they focus on different slices of the problem (RAG QA metrics, tracing + feedback loops) rather than the broader workflow Confident AI covers. Using them together with DeepEval is feasible if you’re willing to stitch the pieces yourself; just be aware you’ll be responsible for building dashboards, alerts, and collaboration features manually.
