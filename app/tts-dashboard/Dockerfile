FROM node:18-alpine

WORKDIR /app

# Install dependencies specifically
COPY package.json ./
RUN npm install

# Copy source code
COPY . .

# Exposure port
EXPOSE 3001

# Start command
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3001"]
