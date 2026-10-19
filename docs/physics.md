# The physics behind SurgeCast

This note explains what the model is predicting and why the inputs, graph and
storm nodes look the way they do.

## What a tide gauge sees

A NOAA gauge records the total water level. NOAA also publishes a harmonic tide
prediction for every gauge: a sum of sinusoids (M2, S2, K1, O1, ...) fitted to
decades of record. Subtracting the two leaves the **non-tidal residual**:

```
residual = observed water level - predicted astronomical tide
```

Storm surge lives in the residual, along with slower signals: seasonal sea
level, El Niño, sea-level rise since the tidal datum epoch (1983–2001), and
small tide-prediction errors. SurgeCast forecasts the *change* in residual over
the next 1–24 h, so the slow parts cancel and the network spends its capacity on
weather-driven water.

## What raises the sea surface in a storm

**Inverse barometer.** Lower air pressure lets the sea surface rise. In
hydrostatic balance a pressure drop Δp lifts the surface by Δp / (ρ g), about
1 cm per hPa. A hurricane with a 950 hPa centre adds roughly 60 cm on its own.
The dataset converts every gauge's barometer reading into this static response
(`physics.inverse_barometer`), in the same normalised units as the target.

**Wind setup.** Wind drags on the surface with stress τ = ρ_air C_d |U| U. On a
shelf of depth h and width L the steady balance is ∂η/∂x ≈ τ / (ρ g h), so the
setup scales like τ L / (ρ g h). Shallow, wide shelves amplify surge, which is
why Florida's Big Bend and the Louisiana coast flood far worse than steep
coasts in comparable storms. The model gets wind stress, not raw wind. The drag
coefficient follows Large & Pond (1981) and is capped above 33 m/s, where
measurements show C_d stops growing (Powell et al., 2003).

**Negative surge.** The same physics run in reverse. Strong offshore wind on a
storm's weak side pushes water away from the coast. When Ian (2022) came ashore
south of Tampa Bay, the Fort Myers gauge, to the right of the eye, rose 2.2 m
above the predicted tide, while St. Petersburg, about 145 km north on the left side,
fell 1.75 m below it. Same storm, same hour, opposite signs. A model that only
knows "storms raise water" gets one of those two gauges badly wrong.

**Tide–surge interaction.** Surge and tide don't simply add. Bottom friction is
quadratic in current speed, and wave speed depends on total depth, so a surge
arriving at high tide behaves differently from one arriving at low tide. The
decoder is given the known future tide at every lead time for this reason.

## Why a graph

Gauges are irregularly spaced points along a coastline, which is a graph, not a
grid. Two physical reasons make neighbours informative:

1. **Propagation along the coast.** Surge travels along the shelf as coastally
   trapped (Kelvin and shelf) waves, which in the northern hemisphere move with
   the coast on their right. Phase speed is about √(g h): around 50 km/h on a
   20 m shelf. A gauge 200 km "upstream" therefore sees the signal hours
   earlier.
2. **Shared forcing.** Neighbouring gauges feel the same weather system with a
   delay set by the storm's motion.

Because coastal waves have a preferred direction, edges are directional: each
edge carries the sender's east/north offset from the receiver, and the
attention weights can treat upstream and downstream neighbours differently.

## Why hurricanes are nodes

A gauge's own barometer and anemometer only see a hurricane once it is close,
and they often fail at landfall. The hurricane's state is known independently
from NHC tracks: position, maximum wind, central pressure, size (34-kt wind
radius) and motion. SurgeCast adds every active storm as an extra node that
messages each gauge within 1000 km, with edge features giving the storm's
position relative to that gauge. That is enough to learn the classic asymmetry:
the strongest surge sits in the right-front quadrant, where storm motion adds
to the onshore wind.

Only the storm's state *at* forecast time and its motion over the previous
6 hours are used. The model never sees where the storm goes next. Best tracks
are post-season reanalyses, so they're slightly cleaner than the advisories
available in real time; that small advantage is a known caveat.

## What this model is not

Operational surge forecasting (NOAA's P-Surge with SLOSH, ESTOFS with ADCIRC)
solves the shallow-water equations on fine coastal meshes, forced by
*forecast* winds. SurgeCast uses observations only: past water levels,
pressure, wind and the current storm state. It's a data-driven short-range
forecast that could run every hour on a laptop, complementing those systems. It
doesn't replace them. Feeding it forecast wind fields (for example from GFS) is
the obvious next step.
