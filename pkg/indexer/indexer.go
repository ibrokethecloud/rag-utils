package indexer

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

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

// GenerateRAG will walk the directory and add markdown files to vector store
// using the embedder and this can be used later for querying information
func (i *Indexer) GenerateRAG(ctx context.Context, dbname string, dir string, filesuffix string) error {

	store, err := i.InitStore(dbname)
	if err != nil {
		return fmt.Errorf("error generating vector store: %v", err)
	}
	return i.addDocuments(ctx, store, dir, filesuffix)
}

func (i *Indexer) addDocuments(ctx context.Context, store vectorstores.VectorStore, dir string, filesuffix string) error {
	// Add documents to the vector store.
	fileList, err := generateDocumentList(dir, filesuffix)
	if err != nil {
		return fmt.Errorf("error generating document list: %v", err)
	}

	for _, file := range fileList {
		logrus.Infof("Adding document: %s", filepath.Base(file))
		contents, err := os.ReadFile(file)
		if err != nil {
			return fmt.Errorf("error reading file: %v", err)
		}

		docs := schema.Document{
			PageContent: string(contents),
			Metadata: map[string]any{
				fileNameKey: filepath.Base(file),
			},
		}
		splitter := textsplitter.NewMarkdownTextSplitter(textsplitter.WithChunkOverlap(250),
			textsplitter.WithChunkSize(2500), textsplitter.WithSecondSplitter(textsplitter.NewRecursiveCharacter()))
		splitdocs, err := textsplitter.SplitDocuments(splitter, []schema.Document{docs})

		if err != nil {
			return fmt.Errorf("error splitting documents: %v", err)
		}
		if _, err = store.AddDocuments(ctx, splitdocs); err != nil {
			logrus.Errorf("error processing document %v: %v", docs, err)
			return err
		}
	}
	return nil
}

func generateDocumentList(dir string, fileSuffix string) ([]string, error) {
	var fileList []string
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
				fileList = append(fileList, absPath)
			}
		}
		return nil
	})
	return fileList, err
}

// QueryRAG uses similarity search to lookup info
func (i *Indexer) QueryRAG(ctx context.Context, dbname, query string, options []vectorstores.Option) ([]schema.Document, error) {
	store, err := i.InitStore(dbname)
	if err != nil {
		return nil, fmt.Errorf("error generating vector store: %v", err)
	}
	return store.SimilaritySearch(ctx, query, 1, options...)
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
