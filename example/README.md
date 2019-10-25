# Profiling

Autograd is the automatic differentiation backend of Pytorch which enables on-the-fly differentation. It has the ability to track each function being called. This ability can be used to profile ML model inference/training. The C++ frontend of Pytorch has access to the following function which enables profiling.
```
autograd::profiler::RecordProfile()
```
We envelop model inference call within the scope of this function call to generate a Trace Viewer compatible `profile.trace` file. As explained in [Pytorch Profiler Documentation](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview), the trace is a list of unordered events, with the format of each event as follows

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

where `name` is the name of the event (CPU function calls correponding to the model layers), `ph` is the event type (for instance, `X` means complete), `ts` is the clock timestamp of the event, `dur` is the duration of the event, `tid` and `pid` are the IDs of the thread and process who output the event.

To visualize the trace generated, one can go to `chrome://tracing` in Google Chrome and load the file.  

To turn autograd profiling on/off, use `EnableProfiling()` and `DisableProfiling()` methods respectively in your go code. Refer to `./batch_mlmodelscope/main.go` for more details.
