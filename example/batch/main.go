package main

import "C"

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/downloadmanager"
	"github.com/rai-project/go-pytorch"
  //cupti "github.com/rai-project/go-cupti"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/all"
  "github.com/rai-project/tracer/ctimer"
)

var (
	batchSize  = 64
	model      = "alexnet"
	graph_url  = "https://s3.amazonaws.com/store.carml.org/models/pytorch/alexnet.pt"
	synset_url = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
)

// convert go Image to 1-dim array
func cvtImageTo1DArray(src image.Image, mean []float32, stddev []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	b := src.Bounds()
	h := b.Max.Y - b.Min.Y // image height
	w := b.Max.X - b.Min.X // image width

	res := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(x+b.Min.X, y+b.Min.Y).RGBA()
			res[y*w+x] = ((float32(r>>8) / 255.0) - mean[0]) / stddev[0]
			res[w*h+y*w+x] = ((float32(g>>8) / 255.0) - mean[1]) / stddev[1]
			res[2*w*h+y*w+x] = ((float32(b>>8) / 255.0) - mean[2]) / stddev[2]
		}
	}

	return res, nil
}

func main() {
	defer tracer.Close()

	dir, _ := filepath.Abs(".")
	dir = filepath.Join(dir, model)
	graph := filepath.Join(dir, "alexnet.pt")
	synset := filepath.Join(dir, "synset.txt")

	if _, err := os.Stat(graph); os.IsNotExist(err) {
		if _, err := downloadmanager.DownloadInto(graph_url, dir); err != nil {
			panic(err)
		}
	}
	if _, err := os.Stat(synset); os.IsNotExist(err) {
		if _, err := downloadmanager.DownloadInto(synset_url, dir); err != nil {
			panic(err)
		}
	}

	// INFO
	pp.Println("Model + Weights url - ", graph_url)
	pp.Println("Labels url - ", synset_url)

	imgDir, _ := filepath.Abs("./_fixtures")
	imagePath := filepath.Join(imgDir, "platypus.jpg")
	// INFO
	pp.Println("Input path - ", imagePath)

	img, err := imgio.Open(imagePath)
	if err != nil {
		panic(err)
	}

	var input []float32
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, 224, 224, transform.Linear)
		res, err := cvtImageTo1DArray(resized, []float32{0.486, 0.456, 0.406}, []float32{0.229, 0.224, 0.225})
		if err != nil {
			panic(err)
		}
		input = append(input, res...)
	}

  dims := append([]int{len(input)}, 3, 224, 224)

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

  //enableCupti := true

  //var cu *cupti.CUPTI
  //if enableCupti && nvidiasmi.HasGPU {
  //  cu, err = cupti.New(cupti.Context(ctx))
  //  if err != nil {
  //    panic(err)
  //  }
  //}

  predictor.EnableProfiling()

  predictor.StartProfiling("predict", "")

	err = predictor.Predict(ctx, input, dims)
	if err != nil {
		panic(err)
	}

  predictor.EndProfiling()

  //if enableCupti && nvidiasmi.HasGPU {
  //  cu.Wait()
  //  cu.Close()
  //}

  profBuffer, err := predictor.ReadProfile()
  if err != nil {
    panic(err)
  }
  predictor.DisableProfiling()

  // INFO
  pp.Println("Profiler output - ", profBuffer)

  t, err := ctimer.New(profBuffer)
  if err != nil {
    panic(err)
  }
  t.Publish(ctx, tracer.APPLICATION_TRACE)

	output, err := predictor.ReadPredictionOutput(ctx)
	if err != nil {
		panic(err)
	}

	var labels []string
	f, err := os.Open(synset)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	features := make([]dlframework.Features, batchSize)
	featuresLen := len(output) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(output[ii*featuresLen+jj]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		features[ii] = rprobs
	}

	if true {
		for i := 0; i < 1; i++ {
			results := features[i]
			top1 := results[0]
			pp.Println(top1.Probability)
			pp.Println(top1.GetClassification().GetLabel())

			top2 := results[1]
			pp.Println(top2.Probability)
			pp.Println(top2.GetClassification().GetLabel())
		}
	} else {
		_ = features
	}

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
