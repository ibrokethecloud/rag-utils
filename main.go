package main

import (
	command "github.com/ibrokethecloud/rag-utils/cmd"
	"github.com/sirupsen/logrus"
)

func main() {
	err := command.Execute()
	if err != nil {
		logrus.Fatal(err)
	}
}
