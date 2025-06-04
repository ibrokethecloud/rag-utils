package command

import (
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/vectorstores"

	"github.com/ibrokethecloud/rag-utils/pkg/indexer"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var ()

var initCommand = &cobra.Command{
	Use:   "init",
	Short: "initialise a vectorstore",
	Long: `initialise a vectorstore connection and will create a new collection in milvus vector store. if one already exists then nothing else will be done.
	to re-initialise the collection, please refer to the drop command`,
	RunE: func(cmd *cobra.Command, args []string) error {
		idx, err := indexer.NewIndexer(ctx, modelName, embeddingModelName, url, dbendpoint)
		if err != nil {
			return err
		}
		_, err = idx.InitStore(collectionName)
		if err != nil {
			return err
		}
		logrus.Infof("collection %s initialised", collectionName)
		return nil
	},
}

var deleteCommand = &cobra.Command{
	Use:   "delete",
	Short: "delete a vectorstore collection",
	Long:  `initialise a vectorstore and delete the specified collection in milvus vector store if one exists`,
	RunE: func(cmd *cobra.Command, args []string) error {
		idx, err := indexer.NewIndexer(ctx, modelName, embeddingModelName, url, dbendpoint)
		if err != nil {
			return err
		}
		err = idx.DropDB(collectionName)
		if err != nil {
			return err
		}
		logrus.Infof("collection %s initialised", collectionName)
		return nil
	},
}

var addCommand = &cobra.Command{
	Use:   "add",
	Short: "embed a document/directory to a collection",
	Long:  `add a specific document or directory of markdown files to a collection. the command will walk the directory searching for files matching the filesuffix argument and embed it in the collection`,
	RunE: func(cmd *cobra.Command, args []string) error {
		idx, err := indexer.NewIndexer(ctx, modelName, embeddingModelName, url, dbendpoint)
		if err != nil {
			return err
		}
		return idx.GenerateRAG(ctx, collectionName, dir, fileSuffix)
	},
}

var queryCommand = &cobra.Command{
	Use:   "query",
	Short: "run a query against a collection",
	Long:  `run a query against a collection and will return the results of the query`,
	RunE: func(cmd *cobra.Command, args []string) error {
		idx, err := indexer.NewIndexer(ctx, modelName, embeddingModelName, url, dbendpoint)
		if err != nil {
			return err
		}
		options := []vectorstores.Option{
			vectorstores.WithScoreThreshold(0.8),
		}

		for _, v := range args {
			results, err := idx.QueryRAG(ctx, collectionName, v, options)
			if err != nil {
				return err
			}
			var contents []string
			for _, v := range results {
				contents = append(contents, v.PageContent)
			}
			fmt.Println(strings.Join(contents, "\n"))
		}

		return nil
	},
}

func init() {
	initCommand.PersistentFlags().StringVar(&collectionName, "collection", "rag", "milvus collection name")
	addCommand.PersistentFlags().StringVar(&collectionName, "collection", "rag", "milvus collection name")
	addCommand.PersistentFlags().StringVar(&dir, "dir", ".", "directory to parse and add markdown files from")
	addCommand.PersistentFlags().StringVar(&fileSuffix, "fileSuffix", ".md", "suffix of files to index in specified directory")
	deleteCommand.PersistentFlags().StringVar(&collectionName, "collection", "rag", "milvus collection name")
	rootCmd.AddCommand(initCommand)
	rootCmd.AddCommand(addCommand)
	rootCmd.AddCommand(queryCommand)
	rootCmd.AddCommand(deleteCommand)
}
