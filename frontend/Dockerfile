# Use the official Node.js image.
FROM node:18 AS build

# Set the working directory in the container.
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock) and install dependencies.
COPY package*.json ./
RUN npm install

# Copy the rest of the application code.
COPY . .

# Build the React application.
RUN npm run build

# Use an official Nginx image to serve the React application.
FROM nginx:alpine

# Copy the build output to the Nginx html directory.
COPY --from=build /app/build /usr/share/nginx/html

# Expose port 80 to access the application.
EXPOSE 80

# Start Nginx.
CMD ["nginx", "-g", "daemon off;"]
