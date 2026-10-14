# Grove fetcher image: the read-only web reader that sits between the residents and the internet. It serves GET /read and GET /search
# for the worker, refuses anything that is not an ordinary public web page (private, loopback, link-local and CGNAT addresses, odd
# ports, odd content types), and asks a SearXNG container (pinned by tag in the Compose file) for search results. Standard library
# only; runs unprivileged. The reader is a second wall: the first is the firewall the installer writes for the network it sits on.
#
# Shipped as docker/fetcher.Dockerfile in the grove repository, built from the repository root:
#     docker build -f docker/fetcher.Dockerfile -t grove-fetcher:local .
# Needs FETCHER_KEY at run time; READER_SEARX_URL says where SearXNG is. No key is baked in.
# Base image: an exact version tag, never a floating one. digest: pin at release.
FROM python:3.12.7-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 READER_PORT=8090 READER_BIND=0.0.0.0
RUN useradd --system --uid 10003 --no-create-home --shell /usr/sbin/nologin grovereader
COPY pyproject.toml README.md LICENSE /src/
COPY grove /src/grove
RUN pip install --no-cache-dir --no-deps /src && rm -rf /src
USER grovereader
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import os, urllib.request as u; u.urlopen(u.Request('http://127.0.0.1:8090/health', headers={'Authorization': 'Bearer ' + os.environ['FETCHER_KEY']}), timeout=3)"]
CMD ["python", "-m", "grove.reader"]
