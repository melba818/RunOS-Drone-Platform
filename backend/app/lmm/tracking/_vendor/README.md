# Third-party tracking library (vendored)

This directory is a vendored third-party component. It is NOT covered by
the RunHive MIT License in the repository root.

## Contents and provenance

- lib/ — the OSTrack visual object tracking library
  (upstream: https://github.com/botaoye/OSTrack), distributed by its
  authors under the MIT License (see the LICENSE file in the upstream
  repository). RunHive uses it for single-object tracking in RunOS.
  Local changes are limited to packaging (this _vendor layout) and
  integration glue that lives outside this directory.

- Parts of lib/train and lib/test derive from the PyTracking library
  (upstream: https://github.com/visionml/pytracking), which is
  distributed under the GNU General Public License v3.0. Those parts
  remain under GPL-3.0; consult the upstream repositories for the exact
  per-file provenance before redistributing this directory on its own.

- timm/ — a small, independently written substitute for the subset of
  the timm (PyTorch Image Models) API that the tracking code needs
  (upstream: https://github.com/huggingface/pytorch-image-models,
  Apache License 2.0). It reimplements compatible interfaces and does
  not copy timm code.

## Redistribution

If you redistribute this directory, keep this notice and the upstream
attributions, and obtain the upstream license texts from the projects
linked above.
