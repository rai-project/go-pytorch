package main

import "C"

import (
	// "bufio"
	"context"
	//"fmt"
	// "image"
	// "os"
	"path/filepath"
	// "sort"

	// "github.com/anthonynsimon/bild/imgio"
	// "github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	// "github.com/rai-project/dlframework"
	// "github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	// "github.com/rai-project/downloadmanager"
	"github.com/rai-project/go-pytorch"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/all"
)

var (
	batchSize  = 1
	model      = "ncf"
	// graph_url  = "https://s3.amazonaws.com/store.carml.org/models/pytorch/alexnet.pt"
	// synset_url = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
)


func main() {
	defer tracer.Close()

	dir, _ := filepath.Abs(".")
	dir = filepath.Join(dir, model)
	graph := filepath.Join(dir, "drqa_pytorch_model.pt")
	// synset := filepath.Join(dir, "synset.txt")

	opts := options.New()

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		pytorch.SetUseGPU()
		device = options.CUDA_DEVICE
	} else {
		pytorch.SetUseCPU()
	}

	ctx := context.Background()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "pytorch_batch")
	defer span.Finish()

	predictor, err := pytorch.New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.BatchSize(batchSize))
	if err != nil {
		panic(err)
	}
	defer predictor.Close()
	/*
	// input defined
	var input []float32
	input = append(input, 0.001)
	input = append(input, 2906.001)
	input = append(input, 1.001)

	err = predictor.Predict(ctx, input)
	if err != nil {
		panic(err)
	}

	output, err := predictor.ReadPredictionOutput(ctx)
	if err != nil {
		panic(err)
	}
	fmt.Printf("output = %f\n",output[0])
	*/
	// var labels []string
	// f, err := os.Open(synset)
	// if err != nil {
	// 	panic(err)
	// }
	// defer f.Close()
	// scanner := bufio.NewScanner(f)
	// for scanner.Scan() {
	// 	line := scanner.Text()
	// 	labels = append(labels, line)
	// }

	// features := make([]dlframework.Features, batchSize)
	// featuresLen := len(output) / batchSize

	// for ii := 0; ii < batchSize; ii++ {
	// 	rprobs := make([]*dlframework.Feature, featuresLen)
	// 	for jj := 0; jj < featuresLen; jj++ {
	// 		rprobs[jj] = feature.New(
	// 			feature.ClassificationIndex(int32(jj)),
	// 			feature.ClassificationLabel(labels[jj]),
	// 			feature.Probability(output[ii*featuresLen+jj]),
	// 		)
	// 	}
	// 	sort.Sort(dlframework.Features(rprobs))
	// 	features[ii] = rprobs
	// }

	// if true {
	// 	for i := 0; i < 1; i++ {
	// 		results := features[i]
	// 		top1 := results[0]
	// 		pp.Println(top1.Probability)
	// 		pp.Println(top1.GetClassification().GetLabel())

	// 		top2 := results[1]
	// 		pp.Println(top2.Probability)
	// 		pp.Println(top2.GetClassification().GetLabel())
	// 	}
	// } else {
	// 	_ = features
	// }

	// INFO
	pp.Println("End of prediction...")
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
