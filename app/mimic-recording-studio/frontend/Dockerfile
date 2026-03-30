FROM node:18-alpine

RUN apk add --no-cache bash
WORKDIR /src/
COPY . /src/
RUN sed -i 's/\r//' start.sh
CMD ["bash", "-c", "sed -i 's/\\r//' start.sh && bash start.sh"]
