# A Beginner's Guide to Streaming Joins

So you have two streams of events and you want to combine them. This
guide walks through the three patterns you will reach for most often
and explains the trade-offs of each in plain language.

## What is a streaming join?

Imagine two faucets dripping events into a sink. One faucet emits
order-placed events; the other emits payment-confirmed events. A
streaming join is the logic that decides which drips from one faucet
belong with which drips from the other, in near real time.

Unlike a batch SQL join, you do not have the luxury of seeing all the
data at once. Events arrive over time, sometimes out of order, and
your join has to make a decision about each one as it shows up.

## Pattern 1: windowed equi-join

The simplest pattern is the **windowed equi-join**. For every event
in stream A, look at every event in stream B that arrived within the
last *N* minutes and share the same key (in our example, the order
ID). If a match is found, emit the joined record.

This pattern is easy to reason about and easy to implement on top of
any event-processing engine. The main downside is that the window
size is a hard knob: too small and you miss late arrivals, too large
and you keep state for events that will never match.

## Pattern 2: stream-table join

The **stream-table join** treats one of the inputs as a slowly
changing reference table rather than a stream. For each event on the
streaming side, you look up the latest value in the reference table
and enrich the event.

This pattern is the right choice when one input is reference data
(customer profiles, product catalogs) that changes far slower than
the event stream. The state cost is bounded by the table size rather
than by the event rate.

## Pattern 3: interval join

The **interval join** generalizes the windowed equi-join. For each
event in stream A, you match against events in stream B whose
timestamps fall within an open or closed interval relative to A's
timestamp. The interval need not be symmetric: "any B that arrived
between 5 minutes before A and 10 minutes after A" is a valid
specification.

Interval joins are useful when the two streams have known clock
skew or when the business event you are detecting has a known time
shape (a delivery event arrives within 48 hours of a shipment
event).

## How to pick

Start with the simplest pattern that fits your data shape. Reach for
the windowed equi-join unless one input is reference data (use
stream-table) or unless you need an asymmetric matching window (use
interval). Only consider richer patterns — such as
session-windowed joins or temporal joins — once you have profiled the
simpler implementation and confirmed it is the bottleneck.

## A note on watermarks

Every streaming join relies on a watermark to know when it is safe
to garbage-collect old state. Without a watermark you are guessing,
and your join's state will grow unbounded. Most production engines
let you configure the watermark on the input streams; pick a value
that is comfortably larger than your expected event lateness and be
prepared to tune it.
