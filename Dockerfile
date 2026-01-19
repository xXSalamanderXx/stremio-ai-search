FROM node:20-alpine
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 7000
ENV NODE_ENV=production
ENV ENABLE_LOGGING=false
RUN mkdir -p logs
CMD ["node", "server.js"] 
