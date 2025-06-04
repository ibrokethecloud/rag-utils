package command

import (
	"context"

	"github.com/spf13/cobra"
)

var (
	rootCmd = &cobra.Command{
		Use:   "rag",
		Short: "rag is a utility to manage conversion of markdown files to a vector store",
		Long: `rag is a utility which allows users to create a new vectorstore from markdown files and manage operations such as deleting
	add adding more content to an existing vector store. it uses milvus as the backend and ollama for embedding`,
		Run: func(cmd *cobra.Command, args []string) {
			ctx = context.Background()
		},
	}
	ctx                context.Context
	url                string
	dbendpoint         string
	modelName          string
	embeddingModelName string
	dir                string
	fileSuffix         string
	collectionName     string
)

func init() {
	rootCmd.PersistentFlags().StringVar(&url, "llmEndpoint", "http://localhost:11434/v1", "llm model endpoint, defaults to http://localhost:11434/v1")
	rootCmd.PersistentFlags().StringVar(&dbendpoint, "milvusEndpoint", "http://localhost:19530", "milvus vector store endpoint, defaults to http://localhost:19530")
	rootCmd.PersistentFlags().StringVar(&modelName, "modelName", "llama3.1", "llm model name, defaults to llama3.1")
	rootCmd.PersistentFlags().StringVar(&embeddingModelName, "embeddingModelName", "mxbai-embed-large", "llm model name for embedding text")
}
func Execute() error {
	return rootCmd.Execute()
}
