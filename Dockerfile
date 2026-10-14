# Grove image = the cockpit image (StoneSage + Hearth) + the Grove plugin. It adds the board tab, the model proxy the worker talks to
# and the scheduler; the worker and the web reader run from their own images (grove-worker.Dockerfile, grove-fetcher.Dockerfile),
# which the Compose stack from `python -m lodestone.installer --docker --with-garden` wires together.
#
# Shipped as `Dockerfile` in the grove repository:
#     docker build -t grove:local .
# BASE_IMAGE is the cockpit image to add the plugin to. Until the images are published, build the layers first and say so:
#     docker build -t hearth:local ../hearth && docker build --build-arg BASE_IMAGE=hearth:local -t grove:local .
# The default is a release tag, never `latest`. digest: pin at release.
ARG BASE_IMAGE=ghcr.io/lottage/hearth:0.1.0-rc.5
FROM ${BASE_IMAGE}

# The platform and Lodestone are already in the base image's environment, and Grove's own code needs nothing else, so only Grove
# itself is installed. pip builds from a copy, not from /src.
USER root
COPY pyproject.toml README.md LICENSE /src/
COPY grove /src/grove
RUN pip install --no-cache-dir --no-deps /src && rm -rf /src
USER stonesage

# Apps the cockpit loads on its first start (docker/entrypoint.py writes them into config.json); the Compose file sets it too.
ENV STONESAGE_APPS=hearth,grove
