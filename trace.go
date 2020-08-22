package pytorch

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	opentracing "github.com/opentracing/opentracing-go"
	"github.com/rai-project/tracer"
)

type TraceEvent struct {
	Name      string    `json:"name,omitempty"`
	Phase     string    `json:"ph,omitempty"`
	Timestamp float32   `json:"ts,omitempty"`
	Duration  float32   `json:"dur,omitempty"`
	ProcessID string    `json:"pid,omitempty"`
	ThreadID  int64     `json:"tid,omitempty"`
	Start     int64     `json:"-"`
	End       int64     `json:"-"`
	StartTime time.Time `json:"-"`
	EndTime   time.Time `json:"-"`
}

func (t TraceEvent) ID() string {
	return fmt.Sprintf("%s/%v", t.Name, t.ThreadID)
}

type TraceEvents []TraceEvent

func (t TraceEvents) Len() int           { return len(t) }
func (t TraceEvents) Swap(i, j int)      { t[i], t[j] = t[j], t[i] }
func (t TraceEvents) Less(i, j int) bool { return t[i].Start < t[j].Start }

type Trace struct {
	StartTime   time.Time
	TraceEvents TraceEvents
}

func (t Trace) Len() int           { return t.TraceEvents.Len() }
func (t Trace) Swap(i, j int)      { t.TraceEvents.Swap(i, j) }
func (t Trace) Less(i, j int) bool { return t.TraceEvents.Less(i, j) }

func NewTrace(data string, start_time int64) (*Trace, error) {
	trace := new(Trace)
	err := json.Unmarshal([]byte(data), &trace.TraceEvents)
	if err != nil {
		return nil, err
	}
	trace.StartTime = time.Unix(0, start_time)
	for ii, event := range trace.TraceEvents {
		trace.TraceEvents[ii].Start = start_time + int64(event.Timestamp*1000)
		trace.TraceEvents[ii].StartTime = time.Unix(0, trace.TraceEvents[ii].Start)
		trace.TraceEvents[ii].End = start_time + int64(event.Timestamp*1000+event.Duration*1000)
		trace.TraceEvents[ii].EndTime = time.Unix(0, trace.TraceEvents[ii].End)
	}
	return trace, nil
}

func (event *TraceEvent) Publish(ctx context.Context, lvl tracer.Level, opts ...opentracing.StartSpanOption) error {
	tags := opentracing.Tags{
		"phase":      event.Phase,
		"process_id": event.ProcessID,
		"thread_id":  event.ThreadID,
	}
	s, _ := tracer.StartSpanFromContext(
		ctx,
		lvl,
		event.Name,
		opentracing.StartTime(event.StartTime),
		tags,
	)
	if s == nil {
		log.WithField("event_name", event.Name).
			WithField("tags", tags).
			Error("failed to create span from context")
		return nil
	}
	s.FinishWithOptions(opentracing.FinishOptions{
		FinishTime: event.EndTime,
	})
	return nil
}

func (t *Trace) Publish(ctx context.Context, lvl tracer.Level, opts ...opentracing.StartSpanOption) error {
	for _, event := range t.TraceEvents {
		if err := event.Publish(ctx, lvl, opts...); err != nil {
			return err
		}
	}
	return nil
}
