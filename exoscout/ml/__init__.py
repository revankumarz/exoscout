"""ExoScout ML - an AstroNet-style CNN for transit vetting.

Implements the dual-view (global + local phase-folded light curve) 1D CNN from
Shallue & Vanderburg (2018), the reference architecture for distinguishing real
transiting planets from false positives. The trained model plugs into the
vetting tool via the `cnn_score` hook.
"""
