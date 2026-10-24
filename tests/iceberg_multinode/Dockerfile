# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY runtime-requirements.txt /opt/qualification/runtime-requirements.txt
COPY runtime-wheels/ /opt/qualification/runtime-wheels/
RUN python -m pip install --no-index --find-links=/opt/qualification/runtime-wheels -r /opt/qualification/runtime-requirements.txt
RUN python -c "import os,ray; from pathlib import Path; p=Path(ray.__file__).parent/'core/src/ray/gcs/gcs_server'; assert os.access(p,os.X_OK),p"
RUN mkdir -p /opt/qualification/helpers /run/qualification
COPY wheels/ /opt/qualification/wheels/
RUN python -m pip install --no-index --no-deps /opt/qualification/wheels/*.whl && python -m pip check
ENV RAY_USAGE_STATS_ENABLED=0 AWS_EC2_METADATA_DISABLED=true PYTHONUNBUFFERED=1 VANE_RUNNER=ray PYTHONPATH=/opt/qualification/helpers
WORKDIR /run/qualification
CMD ["python", "-c", "import time; time.sleep(86400)"]
