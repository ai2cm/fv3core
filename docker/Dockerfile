ARG build_image
FROM $build_image as fv3ser
COPY external/fv3util/requirements.txt /usr/src/fv3util/requirements.txt
COPY requirements_core.txt requirements_core.txt
RUN pip install --no-cache-dir -r requirements_core.txt && \
    pip install -r /usr/src/fv3util/requirements.txt
COPY external/fv3util /usr/src/fv3util
RUN pip install --no-cache-dir -e /usr/src/fv3util

COPY fv3 /fv3

