# TODO
FROM python:3.6-alpine3.6

COPY . /src/
WORKDIR /src
RUN apk update
RUN apk add ffmpeg make automake gcc g++ subversion python3-dev
RUN pip install -r requirements.txt
RUN sed -i 's/\r//' start_prod.sh
CMD ["sh", "-c", "sed -i 's/\\r//' start_prod.sh && sh start_prod.sh"]
