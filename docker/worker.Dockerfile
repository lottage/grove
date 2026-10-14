# Grove worker image: the HTTP service that holds the residents (souls, works, journals, session transcripts). Standard library
# only: nothing from StoneSage is installed here, and the container needs to reach exactly one thing on the network, the cockpit's
# model proxy (and the web reader). Runs unprivileged; everything it keeps is in the /data volume.
#
# Shipped as docker/worker.Dockerfile in the grove repository, built from the repository root:
#     docker build -f docker/worker.Dockerfile -t grove-worker:local .
# Needs GROVE_CONTROL_KEY, GROVE_MODEL_KEY and GROVE_MODEL_URL at run time (see the README); no key is baked in.
# Base image: an exact version tag, never a floating one. digest: pin at release.
FROM python:3.12.7-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 GROVE_DATA_DIR=/data GROVE_PORT=8780 GROVE_BIND=0.0.0.0
RUN useradd --system --uid 10002 --no-create-home --shell /usr/sbin/nologin grove \
 && mkdir -p /data && chown grove:grove /data
COPY pyproject.toml README.md LICENSE /src/
COPY grove /src/grove
RUN pip install --no-cache-dir --no-deps /src && rm -rf /src
USER grove
VOLUME ["/data"]
EXPOSE 8780
# every route of the worker, /health included, needs the control key: the check sends the one this container was given
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import os, urllib.request as u; k = os.environ.get('GROVE_CONTROL_KEY') or os.environ['GARDEN_CONTROL_KEY']; u.urlopen(u.Request('http://127.0.0.1:8780/health', headers={'Authorization': 'Bearer ' + k}), timeout=3)"]
CMD ["python", "-m", "grove.worker"]
