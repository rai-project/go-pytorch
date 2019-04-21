# Batch Image Classification

## Without profiling

```
cd batch
go build
./batch
```

## With profiling (pytorch autograd + nvprof)

```
cd batch_nvprof
go build
nvprof --profile-from-start off ./batch_nvprof
```

## With jaeger

Start jaeger docker container by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
812bba651a4a1c5a5d3c5eac5de610759bf54f716d1e531017b4e206b964e1e8
```

Then run the example by

```
cd batch
go run main.go
```

Go to `xxx:16686` to see the trace

# Autograd profiler

Autograd is the automatic differentiation backend of Pytorch which enables on-the-fly differentation. It has the ability to track each function being called. This data can be used to profile ML model inference/training. The C++ frontend of Pytorch has access to the following function which enables profiling.
```
autograd::profiler::RecordProfile()
```
We envelop model inference call within this function call to generate a Trace Viewer compatible `profile.trace` file. As explained in [Pytorch Profiler Documentation](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview), the trace is a list of unordered events, with the format of each event as follows

```
{
  "name": "_convolution",
  "ph": "X",
  "ts": 20555.611000,
  "dur": 1865.102000,
  "tid": 0,
  "pid": "CPU Functions",
  "args": {}
}
```

where `name` is the name of the event (generally a kernel call correponding to a layer in the model), `ph` is the event type (for instance, `X` means complete), `ts` is the clock timestamp of the event, `dur` is the duration of the event, `tid` and `pid` are the IDs of the thread and process who output the event.

To visualize the trace generated, one can go to `chrome://tracing` in Google Chrome and load the file.  
