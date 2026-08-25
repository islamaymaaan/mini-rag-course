# mini-RAG Project Documentation

## Scope

This is a read-only technical analysis of the inspected repository. The only project change made for this request is this file. The application is a small FastAPI Retrieval-Augmented Generation (RAG) backend: it saves uploaded TXT/PDF files per project, extracts and chunks text, stores chunk records in MongoDB, stores embeddings in local Qdrant, then supports semantic search and RAG answers.

The Python source, settings, dependency manifest, Compose file, prompt templates, Postman collection, ignore files, and existing ignored runtime data directories were inspected. The implementation was not run. There are no tests, migrations, Dockerfile, auth middleware, global exception handler, or background-worker source in the inspected code.

## Repository structure

~~~text
.
├── README.md                         setup notes (UTF-16 encoded)
├── docker/docker-compose.yml         MongoDB only
├── src/main.py                       FastAPI application entry point
├── src/helpers/config.py             Pydantic Settings loader
├── src/routes/                       HTTP endpoints and request schemas
├── src/controllers/                  upload, processing, RAG orchestration
├── src/models/                       MongoDB access models, schemas, enums
├── src/stores/llm/                   LLM abstraction, adapters, templates
├── src/stores/vectordb/              vector abstraction and Qdrant adapter
├── src/assets/files/                 ignored uploaded-file storage
└── src/assets/database/qdrant_db/    ignored local Qdrant storage
~~~

Tracked source is concentrated in 55 Python files, of which 35 contain implementation. Existing asset files and Qdrant files are runtime state, not application source. The tracked Postman collection contains only a minimal placeholder welcome request.

## Architecture

~~~mermaid
flowchart LR
  Client[HTTP client] --> API[FastAPI main.py]
  API --> Routes[Routes /api/v1]
  Routes --> Controllers
  Controllers --> MongoModels[Motor MongoDB models]
  MongoModels --> Mongo[(MongoDB: projects, asset, chunks)]
  Controllers --> Files[(assets/files/project_id)]
  Controllers --> LangChain[TextLoader, PyMuPDFLoader, splitter]
  Controllers --> LLM[LLMInterface / provider factory]
  LLM --> OpenAI
  LLM --> Cohere
  LLM --> Gemini
  Controllers --> Vector[VectorDBInterface / factory]
  Vector --> Qdrant[(local Qdrant)]
  Controllers --> Templates[English/Arabic RAG templates]
~~~

The effective layering is route to controller to MongoDB model/provider. Routes still perform significant orchestration. Controllers coordinate local file I/O, LangChain processing, LLMs, and Qdrant. Mongo model classes own direct Motor calls. Provider clients are initialized once in FastAPI lifespan and kept on request.app; controllers are instantiated per request.

## Startup lifecycle

src/main.py creates FastAPI with an async lifespan, then registers base, data, and nlp routers.

~~~mermaid
sequenceDiagram
  participant F as FastAPI lifespan
  participant S as Settings
  participant LF as LLMProviderFactory
  participant VF as VectorDBProviderFactory
  participant Q as QdrantDBProvider
  F->>S: get_settings()
  F->>F: AsyncIOMotorClient(MONGODB_URL)
  F->>LF: create(GENERATION_BACKEND)
  F->>LF: create(EMBEDDING_BACKEND)
  F->>F: set generation and embedding model settings
  F->>VF: create(VECTOR_DB_BACKEND)
  VF->>Q: construct local provider
  F->>Q: connect()
  F->>F: TemplateParser(PRIMARY_LANG, DEFAULT_LANG)
  Note over F: App attributes: mongo_conn, db_client,\n generation_client, embedding_client,\n vectordb_client, template_parser
  F->>F: shutdown: close Mongo, disconnect Qdrant
~~~

Mongo collections and their indexes are not initialized in lifespan. Each model class initializes its collection only when its async create_instance factory is called from an endpoint. Startup does not verify connectivity, credentials, provider validity, models, or collection availability.

## Configuration

helpers/config.py defines Settings using Pydantic Settings v2. It loads .env, ignores extra keys, creates a fresh settings object on every get_settings call, and requires all non-optional values. Secret API key values are deliberately omitted as [SECRET - VALUE NOT DOCUMENTED].

| Group | Settings | Inspected configured/default behavior | Consumers |
|---|---|---|---|
| App | APP_NAME, APP_VERSION | mini-RAG, 0.1 | welcome route |
| File validation | FILE_ALLOWED_TYPES | text/plain and application/pdf | DataController |
| File validation | FILE_MAX_SIZE | 10, used as MiB | DataController |
| File write | FILE_DEFAULT_CHUNK_SIZE | 512000 bytes per write | upload route |
| Mongo | MONGODB_URL, MONGODB_DATABASE | configured; URL omitted | lifespan |
| LLM selection | GENERATION_BACKEND, EMBEDDING_BACKEND | COHERE for both | lifespan/factory |
| Generation | GENERATION_MODEL_ID | configured Cohere model ID | generation client |
| Embeddings | EMBEDDING_MODEL_ID, EMBEDDING_MODEL_SIZE | embed-multilingual-light-v3.0, 384 | embedding client/Qdrant |
| Provider defaults | INPUT_DAFAULT_MAX_CHARACTERS, GENERATION_DAFAULT_MAX_TOKENS, GENERATION_DAFAULT_TEMPERATURE | 1024 characters, 200 tokens, 0.1 | every provider |
| Vector | VECTOR_DB_BACKEND, VECTOR_DB_PATH, VECTOR_DB_DISTANCE_METHOD | QDRANT, qdrant_db, cosine | vector factory |
| Localization | PRIMARY_LANG, DEFAULT_LANG | en, en | TemplateParser |
| Declared only | GROQ_API_KEY | optional; no consumer | none |

OPENAI_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, and GROQ_API_KEY are [SECRET - VALUE NOT DOCUMENTED]. OPENAI_API_URL is optional and forwarded as OpenAI base_url. Class defaults are None for vector distance, en for primary/default language, and None for optional credentials.

## Data model and persistence

~~~mermaid
erDiagram
  PROJECTS ||--o{ ASSET : asset_project_id
  PROJECTS ||--o{ CHUNKS : chunk_project_id
  ASSET ||--o{ CHUNKS : chunk_asset_id
  PROJECTS { ObjectId _id string project_id }
  ASSET { ObjectId _id ObjectId asset_project_id string asset_type string asset_name integer asset_size object asset_config datetime asset_pushed_at }
  CHUNKS { ObjectId _id string chunk_text object chunk_metadata integer chunk_order ObjectId chunk_project_id ObjectId chunk_asset_id }
~~~

| Schema/model | Fields, indexes, and role |
|---|---|
| Project / ProjectModel | id aliases Mongo _id; project_id is non-empty and Pydantic-validated alphanumeric. A unique project_id index is created. get_project_or_create_one reads or inserts it. get_all_projects is unused by routes. |
| Asset / AssetModel | asset_project_id ObjectId, non-empty type/name, non-negative size, optional config, UTC pushed timestamp. Indexes: project and unique project/name. Upload creates assets; process reads them. |
| DataChunk / ChunckModel | Non-empty chunk text; source metadata; positive per-file chunk order; project and asset IDs. Index on project. Processing inserts batches of 100; indexing pages chunks in 50. The class name and several method names use the spelling Chunck/creat/get_poject in actual code. |
| RetrievedDocument | Transient text and float score returned by Qdrant search. It deliberately omits payload metadata and point ID. |

All schema classes permit arbitrary BSON ObjectId types. Chunk deletion supports ObjectId/string representations with an OR filter. No foreign keys, transactions, ownership fields, or cascade deletion exist.

## API

No endpoint declares a response model, authentication requirement, authorization check, or global error schema. Explicit business errors usually use HTTP 400 and a ResponseSignal string.

| Endpoint | Request | Success response and flow |
|---|---|---|
| GET /api/v1/ | none | app_name and app_version from Settings. |
| POST /api/v1/data/upload/{project_id} | multipart UploadFile | Gets/creates project; validates MIME/size; writes disk file; inserts asset; returns signal plus Mongo asset ObjectId as file_id. |
| POST /api/v1/data/process/{project_id} | ProcessRequest: file_id optional, chunk_size=100, overlap_size=20, do_reset=0 | Gets/creates project; selects named/all assets; loads, splits, bulk inserts chunks; returns inserted_chunks and processed_files. |
| POST /api/v1/nlp/index/push/{project_id} | PushRequest: do_reset=0 | Gets/creates project; pages Mongo chunks; embeds and inserts Qdrant points; returns inserted count. |
| GET /api/v1/nlp/index/info/{project_id} | none | Gets/creates project, obtains Qdrant collection info, JSON-normalizes object attributes. |
| POST /api/v1/nlp/index/search/{project_id} | SearchRequest: text required, limit=5 | Embeds query, runs Qdrant search, returns text/score results. |
| POST /api/v1/nlp/index/answer/{project_id} | SearchRequest | Retrieves chunks, builds localized RAG prompt, calls generation provider, returns answer, full_prompt, and chat_history. |

Schema numeric values have no positive or range constraints. Pydantic/FastAPI default validation handles malformed body input. Most unexpected MongoDB, Qdrant, loader, splitting, and SDK exceptions escape as default server errors.

## Execution flows

### Upload flow

~~~mermaid
flowchart TD
  A[POST upload] --> B[ProjectModel get/create]
  B --> C[DataController validate_uploaded_file]
  C -->|bad type/size| D[400 ResponseSignal]
  C -->|valid| E[ProjectController creates assets/files/project_id]
  E --> F[random 12-char prefix + cleaned filename]
  F --> G[aiofiles writes chunks]
  G -->|exception| H[log and 400 upload failed]
  G --> I[AssetModel.create_asset]
  I --> J[(Mongo asset)]
  J --> K[success, asset ObjectId]
~~~

DataController accepts only configured client MIME types and compares file.size to 10 MiB under current config. It removes non-word/non-dot filename characters and generates a random lowercase-alphanumeric 12-character prefix. The saved asset name is the generated filename; the returned file_id is instead the database ObjectId.

### Processing flow

~~~mermaid
flowchart LR
  A[POST process] --> B[choose one asset name or all file assets]
  B --> C{saved extension}
  C -->|.txt| D[LangChain TextLoader UTF-8]
  C -->|.pdf| E[LangChain PyMuPDFLoader]
  D --> F[LangChain documents]
  E --> F
  F --> G[RecursiveCharacterTextSplitter]
  G --> H[DataChunk records with source metadata]
  H --> I[Motor bulk_write batches of 100]
  I --> J[(Mongo chunks)]
~~~

ProcessController gets a file by saved extension, not by validated MIME. It passes loader page text and metadata to RecursiveCharacterTextSplitter using len as the length function. The requested chunk size/overlap are character values. chunk_order restarts at one for each processed file. If do_reset equals exactly integer 1, it deletes this project's Mongo chunks before processing; it does not delete assets, local files, or Qdrant vectors.

### Index, search, and RAG flow

~~~mermaid
flowchart TD
  I[POST index push] --> MC[Read Mongo chunks, 50/page]
  MC --> DE[embed_text text as document]
  DE --> CC[create collection_projectId]
  CC --> UP[Qdrant upload_records]
  Q[Search question] --> QE[embed_text as query]
  QE --> VS[Qdrant nearest-vector search]
  VS --> RD[RetrievedDocument text + score]
  RD --> DP[render document prompts]
  Q --> FP[render footer prompt]
  DP --> GP[generation provider]
  FP --> GP
  SP[system prompt in history] --> GP
  GP --> AN[answer]
~~~

NLPController names a project collection collection_<project.project_id>. Qdrant runs in embedded/local mode through QdrantClient(path=src/assets/database/qdrant_db). It receives configured vector dimension 384 and cosine similarity under current settings. Index points contain payload text and metadata. Qdrant searches use supplied top-k limit and return only text/score.

For answers, NLPController retrieves semantic results then gets rag.system_prompt, rag.document_prompt, and rag.footer_prompt from TemplateParser. It creates one system-message history item, joins rendered documents and the question footer as full_prompt, and calls generation_client.generate_text. There is no user-supplied chat history, conversation persistence, citations, source metadata in answers, or token-aware aggregate context budget.

## LLM and prompt architecture

~~~mermaid
classDiagram
  class LLMInterface {
    <<abstract>>
    set_generation_model
    set_embedding_model
    generate_text
    embed_text
    construct_prompt
  }
  LLMInterface <|.. OpenAIProvider
  LLMInterface <|.. CoHereProvider
  LLMInterface <|.. GeminiProvider
  LLMProviderFactory --> OpenAIProvider
  LLMProviderFactory --> CoHereProvider
  LLMProviderFactory --> GeminiProvider
~~~

LLMProviderFactory selects exact uppercase OPENAI, COHERE, or GEMINI; an unknown backend returns None. It supplies API configuration plus common input/output/temperature defaults to adapters. Each adapter stores its configured model ID and embedding_size. Every input text is truncated to INPUT_DAFAULT_MAX_CHARACTERS (1,024 currently), including individual retrieved chunks and prompt text.

| Provider | Actual SDK translation | Embedding behavior | Prompt shape |
|---|---|---|---|
| OpenAIProvider | OpenAI Responses create with model, input history, max_output_tokens, temperature; returns output_text. | embeddings.create; ignores document/query distinction. | dict role/content; mutates supplied history by appending user prompt. |
| CoHereProvider | ClientV2 chat using messages, max_tokens, temperature; joins text-bearing response content parts. | embed with float vectors and search_document/search_query input type. | dict role/content. This is configured provider for both roles. |
| GeminiProvider | google.genai models.generate_content with GenerateContentConfig. | embed_content with RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY and output dimensionality. | types.Content; assistant/model maps to model, every other role maps to user. |

OpenAI and Cohere providers mainly log invalid responses but do not catch SDK exceptions. Gemini catches generation and embedding exceptions, logs, and returns None. Gemini strips models/ from configured model IDs. The provider abstraction is real but leaky: adapters need different history structures and behavior.

TemplateParser dynamically imports stores.llm.templates.locales.<language>.<group>. It selects PRIMARY_LANG when that directory exists and falls back to DEFAULT_LANG for missing language/group. Available locales are English and Arabic, each with group rag. Both templates have a system prompt requiring concise, polite, document-based same-language responses; numbered document content; and a footer that demands an answer based only on the documents.

## Important file reference

| Path | Responsibility / calls |
|---|---|
| src/main.py | Application entry point/lifespan; calls Settings, Motor, both factories, Qdrant connect, TemplateParser; includes routers. |
| src/helpers/config.py | Typed environment configuration and get_settings. Called by lifespan, controller base, model base, route dependency. |
| src/routes/base.py | Welcome API route. |
| src/routes/data.py | Upload/process HTTP workflows; calls models and Data/Project/Process controllers; owns aiofiles write loop. |
| src/routes/nlp.py | Index/info/search/answer APIs; creates NLPController from app dependencies. |
| src/routes/schemes/data.py | ProcessRequest Pydantic schema. |
| src/routes/schemes/nlp.py | PushRequest and SearchRequest schemas. |
| src/controllers/BaseController.py | Computes src asset directories, reads settings, random identifier generation, creates database directory. |
| src/controllers/ProjectController.py | Creates/returns per-project file directory. |
| src/controllers/DataController.py | MIME/size validation and sanitized/randomized file path generation. |
| src/controllers/ProcessController.py | Selects LangChain loader and chunks loaded documents. |
| src/controllers/NLPController.py | Collection naming, vector index/search, RAG prompt orchestration. |
| src/models/BaseDataModel.py | Shared Motor client/settings holder. |
| src/models/ProjectModel.py | Projects collection initialization and get/create/list operations. |
| src/models/AssetModel.py | Asset collection initialization/create/read operations. |
| src/models/ChunkModel.py | Chunk initialization, bulk insert, get, delete, project paging. |
| src/models/db_schemes/project.py | Project Pydantic document and unique index definition. |
| src/models/db_schemes/asset.py | Asset Pydantic document and indexes. |
| src/models/db_schemes/data_chunk.py | DataChunk and RetrievedDocument schemas. |
| src/models/enums/*.py | Collection, asset-type, processing-extension, and response signal constants. |
| src/stores/llm/LLMInterface.py | Abstract contract for all LLM adapters. |
| src/stores/llm/LLMProviderFactory.py | Creates OpenAI, Cohere, Gemini adapters. |
| src/stores/llm/providers/*.py | Concrete provider-specific SDK translations. |
| src/stores/llm/templates/template_parser.py | Localized template selection/dynamic import. |
| src/stores/llm/templates/locales/{en,ar}/rag.py | RAG system/document/footer template definitions. |
| src/stores/vectordb/VectorDBInterface.py | Abstract vector-store operations. |
| src/stores/vectordb/VectorDBProviderFactory.py | Selects Qdrant and builds local storage path. |
| src/stores/vectordb/providers/QdrantDBProvider.py | Qdrant collection, upload, and vector-search implementation. |
| src/requirements.txt | Pinned/limited runtime dependencies. |
| docker/docker-compose.yml | MongoDB 7.0-jammy, port 27007:27017, named volume; no app or Qdrant service. |
| README.md | Setup information. Several instructions mention absent components. |
| src/assets/min-rag-app.postman_collection.json | Bare Postman collection with welcome entry. |

Empty/re-export package files are architecturally minor. routes/__Init__.py has an uppercase I, unlike conventional __init__.py.

## Dependencies

| Dependency | Used for |
|---|---|
| fastapi 0.110.2, uvicorn standard 0.29.0 | ASGI HTTP application/server. |
| python-multipart, aiofiles | multipart parsing and async upload persistence. |
| python-dotenv, pydantic-settings | .env configuration binding. |
| motor 3.4.0, pymongo 4.x | MongoDB async access and InsertOne batches. |
| langchain, langchain-text-splitters, PyMuPDF | TXT/PDF loading and text chunking. |
| openai, cohere, google-genai | selectable LLM/generation/embedding SDKs. |
| qdrant-client 1.10.1 | embedded vector storage and nearest-neighbor search. |

## Error handling and logging

ResponseSignal provides named string signals such as file_upload_sucess, processing_sucess, vectordb_search_error, and rag_answer_error (spellings are source behavior). Routes explicitly return 400 on known validation/falsy-result paths. They do not consistently catch external errors. Route loggers use uvicorn.error; provider/Qdrant classes use module loggers. The app has no configured logging policy, structured logs, metrics, tracing, health check, retry mechanism, or provider/vector/database error translation.

## Known issues / observations

No issue below was fixed.

| Location | Observation and potential impact |
|---|---|
| data upload/process | Upload response file_id is a Mongo asset ObjectId, while single-file process lookup expects asset_name, the randomized stored filename. A client cannot process that one file by directly reusing the returned ID. |
| nlp index push/NLPController | do_reset reaches create_collection on every 50-chunk page. A reset with multiple pages repeatedly deletes the collection, leaving only the final page. |
| NLPController.index_into_vector_db | It ignores false results from collection creation/insert_many and returns True even if embeddings or vector writes fail. Successful API signals can be inaccurate. |
| nlp routes | get_project_or_create_one makes the project-not-found branch unreachable and causes info/search/answer to create empty projects. |
| Process reset | Processing reset deletes Mongo chunks only. Existing project vectors remain until a reset indexing call. |
| duplicate operation | Reprocessing without reset can duplicate chunks; indexing has no durable Mongo chunk point IDs and no pre-index deduplication. |
| upload validation | It trusts request MIME type but later chooses a loader by filename extension; file contents are not sniffed. file.size may be absent in some upload contexts, making comparison fail. |
| path handling | ProjectController creates a filesystem path from raw project_id before Project Pydantic validation. Invalid values can create paths; filesystem path handling is insufficiently separated from logical-ID validation. |
| Qdrant failures | Collection info/search assume a client and collection exist; exceptions are uncaught and become server errors. Unknown distance yields None rather than a validated distance. |
| providers | Unknown factory backend returns None; startup does not validate it. OpenAI/Cohere SDK exceptions are uncaught. Provider-native history values leak through controller/API. |
| Gemini prompts | construct_prompt maps system to user; the controller gives a types.Content system item, so Gemini may not receive it as system_instruction. |
| RAG context | Individual strings are silently truncated at 1,024 characters under current settings. There is no total context token budget, metadata filtering, source citation, prompt-injection control, or provenance enforcement. |
| API security | No authentication, authorization, project ownership, rate limiting, file scanning, quotas, or CORS policy was found. full_prompt returned in answer responses may disclose retrieved material. |
| schema validation | chunk size, overlap, and retrieval limit lack positive/range checks. asset_size is non-optional annotated but defaults to None. |
| naming/package consistency | ChunckModel, creat_chunk, get_poject_chunks, DAFAULT/SUCESS enums, and routes/__Init__.py are inconsistent spelling/casing. |
| deployment/docs | README is UTF-16 and references Alembic, Celery, Flower, monitoring, Ollama, and Compose services absent from source/Compose. Qdrant is process-local, limiting multi-instance scalability. |
| testability | No tests or test configuration were found. Fresh Settings creation and request.app dependencies hinder isolation without mocks. |

## Architectural assessment

The project has a useful basic separation: endpoint routing, controller orchestration, Mongo model access, and pluggable LLM/vector adapters. LLMInterface and VectorDBInterface offer clear extension points, but provider-specific prompt/history semantics leak into application flow and only Qdrant is implemented. Direct model use from routes and multi-responsibility controllers make the layers less strict than their names suggest.

It is appropriate for a small single-process demonstration: uploads stream to local disk, Mongo is externalized through configuration, and Qdrant is local. It is not architected as a horizontally shared service because uploaded data and Qdrant collections live on API-local disk; document extraction, embedding, and local Qdrant SDK calls occur synchronously inside async request handlers. Indexing is paged/batched but not transactional or backgrounded. Security, operational resilience, observability, testing, and lifecycle/delete workflows are not implemented in the available source.

# AI CONTEXT — PROJECT KNOWLEDGE BASE

mini-rag-proj is a minimal FastAPI RAG backend in src. main.py startup loads Pydantic .env settings, opens Motor MongoDB, creates a generation client and embedding client through LLMProviderFactory, opens an embedded Qdrant client, and creates a localized TemplateParser. Dependencies are stored as request.app attributes. Configured backends are Cohere for generation and embeddings; vector storage is Qdrant at assets/database/qdrant_db with cosine distance. Current configured embedding size is 384, provider input truncation 1024 characters, generation maximum 200 tokens, and temperature 0.1. Never expose credential values: use [SECRET - VALUE NOT DOCUMENTED].

Core flow: POST /api/v1/data/upload/{project_id} gets/creates a project, validates declared text/plain or application/pdf MIME/size, writes a randomized filename under assets/files/<project_id>, and inserts an Asset Mongo document. POST /data/process gets/creates the project, loads TXT with TextLoader UTF-8 or PDF with PyMuPDFLoader, splits with RecursiveCharacterTextSplitter, and inserts DataChunk records into Mongo. POST /nlp/index/push pages chunks, embeds each as a document, and upserts them into collection_<project_id>. Search embeds as a query and Qdrant returns text plus score. Answer retrieves chunks, renders localized RAG system/document/footer templates, and sends the documents plus question to the generation provider. There is no conversation history persistence.

Mongo collections: projects (unique alphanumeric project_id), asset (project ID, generated file name/type/size), chunks (text, loader metadata, order, project/asset IDs). Qdrant payloads retain text/metadata, but public results expose text/score only. Routes call controllers directly; controllers call Mongo models, LangChain, providers, and local filesystem. LLM adapters exist for OpenAI, Cohere V2, and Gemini; Cohere uses document/query-specific embedding input types, Gemini uses retrieval task types, and OpenAI ignores document type. API responses use ResponseSignal strings; no auth, global error translator, tests, queue, migration, or remote vector service exists.

Before modifying behavior, account for current defects: upload returns an ObjectId called file_id but process-one-file expects generated asset_name; reset indexing on more than one page loses earlier pages; vector insertion errors are ignored; all NLP operations implicitly create projects; process reset does not remove vectors; inputs use raw project IDs in filesystem paths before validation; and README deployment instructions are stale. Current implementation is a compact demo/single-process architecture, not a hardened multi-user production system.

