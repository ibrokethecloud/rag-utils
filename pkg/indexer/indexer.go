package indexer

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/sirupsen/logrus"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/milvus"
)

type Indexer struct {
	ctx               context.Context
	ModelName         string
	EmbeddedModelName string
	DBEndpoint        string
	embedder          embeddings.Embedder
	URL               string
}

const (
	fileNameKey = "FileName"
)

func NewIndexer(ctx context.Context, ModelName, EmbeddedModelName, URL, DBEndpoint string) (*Indexer, error) {
	if err := os.Setenv("OPENAI_API_KEY", "fake"); err != nil {
		return nil, fmt.Errorf("error setting fake openai_api_key: %v", err)
	}

	llm, err := openai.New(openai.WithBaseURL(URL), openai.WithEmbeddingModel(EmbeddedModelName), openai.WithModel(ModelName))
	if err != nil {
		return nil, fmt.Errorf("error generating llm object: %v", err)
	}

	embedder, err := embeddings.NewEmbedder(llm)

	if err != nil {
		return nil, fmt.Errorf("error generating embedder: %v", err)
	}
	return &Indexer{
		ctx:               ctx,
		ModelName:         ModelName,
		EmbeddedModelName: EmbeddedModelName,
		DBEndpoint:        DBEndpoint,
		embedder:          embedder,
		URL:               URL,
	}, nil
}

// GenerateRAG will walk the directory and add add markdown files to vector store
// using the embedder and this can be used later for querying information
func (i *Indexer) GenerateRAG(ctx context.Context, dbname string, dir string, filesuffix string) error {

	store, err := i.InitStore(dbname)
	if err != nil {
		return fmt.Errorf("error generating vector store: %v", err)
	}
	return i.addDocuments(ctx, store, dir, filesuffix)
}

// newStore initialises the vector store connector
func newStore(name string, embedder embeddings.Embedder, dbendpoint string) (vectorstores.VectorStore, error) {

	// can try out different indexing options to check
	// if it affects result quality
	idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	milvusConfig := client.Config{
		Address: dbendpoint,
	}

	logrus.Infof("setting up milvus store %s", name)

	opts := []milvus.Option{
		milvus.WithCollectionName(name),
		milvus.WithIndex(idx),
		milvus.WithEmbedder(embedder),
		milvus.WithSkipFlushOnWrite(),
	}

	store, err := milvus.New(
		ctx,
		milvusConfig,
		opts...)

	return store, err
}

func (i *Indexer) addDocuments(ctx context.Context, store vectorstores.VectorStore, dir string, filesuffix string) error {
	// Add documents to the vector store.
	docs, err := generateDocument(dir, filesuffix)
	if err != nil {
		return err
	}

	for _, doc := range docs {
		logrus.Infof("processing file: %s", doc.Metadata[fileNameKey])
		splitter := textsplitter.NewMarkdownTextSplitter(textsplitter.WithChunkOverlap(250),
			textsplitter.WithChunkSize(2500), textsplitter.WithSecondSplitter(textsplitter.NewRecursiveCharacter()))
		splitdocs, err := textsplitter.SplitDocuments(splitter, docs)
		if err != nil {
			return err
		}

		//chunked := chunkDocs(splitdocs, 500)

		if _, err = store.AddDocuments(ctx, splitdocs); err != nil {
			logrus.Errorf("error processing document %v: %v", doc, err)
			return err
		}
		time.Sleep(10 * time.Second)

	}
	return nil
}

// generate documents will read the documents and generate a schema document for markdown files
func generateDocument(dir string, fileSuffix string) ([]schema.Document, error) {
	var resp []schema.Document
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Printf("prevent panic by handling failure accessing a path %q: %v\n", path, err)
			return err
		}

		if !info.IsDir() {
			absPath, err := filepath.Abs(path)
			if err != nil {
				return err
			}
			if filepath.Ext(absPath) == fileSuffix {
				contents, err := os.ReadFile(absPath)
				if err != nil {
					return nil
				}
				resp = append(resp, schema.Document{
					PageContent: string(contents),
					Metadata: map[string]any{
						fileNameKey: filepath.Base(absPath),
					},
				})
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// QueryRAG uses similarity search to lookup info
func (i *Indexer) QueryRAG(ctx context.Context, dbname, query string, options []vectorstores.Option) ([]schema.Document, error) {
	store, err := newStore(dbname, i.embedder, i.DBEndpoint)
	if err != nil {
		return nil, fmt.Errorf("error generating vector store: %v", err)
	}
	return store.SimilaritySearch(ctx, query, 1, options...)
}

func chunkDocs(docs []schema.Document, chunkSize int) [][]schema.Document {
	var chunkedDocs [][]schema.Document
	if len(docs) < chunkSize {
		return [][]schema.Document{docs}
	}

	for i := 0; i < len(docs); i = i + chunkSize {
		limit := i + chunkSize
		if len(docs[i:]) < chunkSize {
			chunkedDocs = append(chunkedDocs, docs[i:])
		} else {
			chunkedDocs = append(chunkedDocs, docs[i:limit])
		}
	}
	return chunkedDocs
}

// saveDocumentsWithRetry retries on 500 status code when adding document
func saveDocumentsWithRetry(ctx context.Context, store vectorstores.VectorStore, docs []schema.Document) error {
	var err error
	for i := 0; i < 10; i++ {
		if _, err = store.AddDocuments(ctx, docs); err != nil {
			time.Sleep(2 * time.Second)
		}
	}
	return err
}

func (i *Indexer) DropDB(collectionName string) error {
	milvusConfig := client.Config{
		Address: i.DBEndpoint,
	}
	milvusClient, err := client.NewClient(i.ctx, milvusConfig)
	if err != nil {
		return fmt.Errorf("error setting up milvus client: %v", err)
	}

	list, err := milvusClient.ListCollections(i.ctx, client.WithShowInMemory(true))
	if err != nil {
		return fmt.Errorf("error listing collections: %v", err)
	}

	var found bool
	for _, v := range list {
		if v.Name == collectionName {
			found = true
		}
	}

	if found {
		return milvusClient.DropCollection(i.ctx, collectionName)
	}
	return nil

}

func (i *Indexer) InitStore(name string) (vectorstores.VectorStore, error) {
	// newStore initialises the vector store connector
	// can try out different indexing options to check
	// if it affects result quality
	idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	if err != nil {
		return nil, fmt.Errorf("error initialising index: %v", err)
	}

	milvusConfig := client.Config{
		Address: i.DBEndpoint,
	}

	opts := []milvus.Option{
		milvus.WithCollectionName(name),
		milvus.WithIndex(idx),
		milvus.WithEmbedder(i.embedder),
	}

	store, err := milvus.New(
		i.ctx,
		milvusConfig,
		opts...)

	return store, err
}
