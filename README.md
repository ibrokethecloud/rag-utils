# rag-utils
Utils to build and query a rag from a markdown repo



rag is a utility which allows users to create a new vectorstore from markdown files and manage operations such as deleting
add adding more content to an existing vector store. it uses milvus as the backend and ollama for embedding

```
Usage:
rag [flags]
rag [command]

Available Commands:
add         embed a document/directory to a collection
delete      delete a vectorstore collection
help        Help about any command
init        initialise a vectorstore
query       run a query against a collection

Flags:
--embeddingModelName string   llm model name for embedding text (default "mxbai-embed-large")
-h, --help                        help for rag
--llmEndpoint string          llm model endpoint, defaults to http://localhost:11434/v1 (default "http://localhost:11434/v1")
--milvusEndpoint string       milvus vector store endpoint, defaults to http://localhost:19530 (default "http://localhost:19530")
--modelName string            llm model name, defaults to llama3.1 (default "llama3.1")

Use "rag [command] --help" for more information about a command.
```


## init command

initialise a vectorstore connection and will create a new collection in milvus vector store. if one already exists then nothing else will be done.
to re-initialise the collection, please refer to the drop command

```
Usage:
rag init [flags]

Flags:
--collection string   milvus collection name (default "rag")
-h, --help                help for init

Global Flags:
--embeddingModelName string   llm model name for embedding text (default "mxbai-embed-large")
--llmEndpoint string          llm model endpoint, defaults to http://localhost:11434/v1 (default "http://localhost:11434/v1")
--milvusEndpoint string       milvus vector store endpoint, defaults to http://localhost:19530 (default "http://localhost:19530")
--modelName string            llm model name, defaults to llama3.1 (default "llama3.1")
```

## add command
add a specific document or directory of markdown files to a collection. the command will walk the directory searching for files matching the filesuffix argument and embed it in the collection

```
Usage:
rag add [flags]

Flags:
--collection string   milvus collection name (default "rag")
--dir string          directory to parse and add markdown files from (default ".")
--fileSuffix string   suffix of files to index in specified directory (default ".md")
-h, --help                help for add
```

## delete command
initialise a vectorstore and delete the specified collection in milvus vector store if one exists

```
Usage:
rag delete [flags]

Flags:
--collection string   milvus collection name (default "rag")
-h, --help                help for delete

Global Flags:
--embeddingModelName string   llm model name for embedding text (default "mxbai-embed-large")
--llmEndpoint string          llm model endpoint, defaults to http://localhost:11434/v1 (default "http://localhost:11434/v1")
--milvusEndpoint string       milvus vector store endpoint, defaults to http://localhost:19530 (default "http://localhost:19530")
--modelName string            llm model name, defaults to llama3.1 (default "llama3.1")
```

## query command
run a query against a collection and will return the results of the query

```
Usage:
rag query [flags] [query]

Flags:
-h, --help   help for query

Global Flags:
--embeddingModelName string   llm model name for embedding text (default "mxbai-embed-large")
--llmEndpoint string          llm model endpoint, defaults to http://localhost:11434/v1 (default "http://localhost:11434/v1")
--milvusEndpoint string       milvus vector store endpoint, defaults to http://localhost:19530 (default "http://localhost:19530")
--modelName string            llm model name, defaults to llama3.1 (default "llama3.1")
```

## To build
`go build -o /tmp/rag ./cmd`