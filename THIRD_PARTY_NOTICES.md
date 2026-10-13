# Third-party notices

Copyright 2026 FFXIV Prism contributors. Original project code is licensed
under Apache License 2.0; see LICENSE. This grant excludes third-party code,
model weights and their derivatives, game assets, website images and player
submissions. Each retains its applicable license and rights.

## Models

- DINOv2 — Meta / facebookresearch: https://github.com/facebookresearch/dinov2
  Code and the DINOv2 base model used here: Apache-2.0.
- SigLIP 2 — Google: https://huggingface.co/google/siglip2-base-patch16-224
  Model card: Apache-2.0.
- ViTPose — https://huggingface.co/usyd-community/vitpose-base-simple
  Model card: Apache-2.0.
- DINOv3 — Meta: https://github.com/facebookresearch/dinov3
  DINOv3 and its derivatives remain subject to the DINOv3 License, not the
  project's Apache-2.0 grant. The full license is included at
  var/models/dinov3-vitb16-pretrain-lvd1689m/LICENSE.md in source checkouts
  and licenses/DINOv3-LICENSE.md in the portable app payload.

The Apache-2.0 license text is included in LICENSE. Preserve upstream notices
and any additional terms accompanying model or library distributions.

## Libraries and runtime

Direct dependencies are listed in pyproject.toml; portable runtime artifacts
and their exact sources are recorded in manifests/runtime*.json. They include
Python, PyTorch, TorchVision, Transformers, NumPy, Pillow, Beautiful Soup,
HTTPX, cloudscraper, Selenium and websocket-client, with curl_cffi available
for optional collection workflows. Their own licenses and notices remain
applicable, including those packaged in wheels and the Python runtime.

## Data and images

Equipment examples: https://ffxiv.eorzeacollection.com/

Item information: https://ff14.huijiwiki.com/, https://v2.xivapi.com/ and
https://github.com/xivapi/ffxiv-datamining

FINAL FANTASY XIV game assets and trademarks belong to their respective
rightsholders, including SQUARE ENIX. Website content and player submissions
retain their respective rights. Public availability does not itself grant
redistribution rights. FFXIV Prism is an unofficial project; these sources
do not imply sponsorship or endorsement.
