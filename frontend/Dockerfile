FROM node:20-alpine AS build
WORKDIR /srv/runos/web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN mkdir -p public && npm run build

FROM node:20-alpine AS runtime
ENV NODE_ENV=production
WORKDIR /srv/runos/web
COPY --from=build /srv/runos/web/.next/standalone ./
COPY --from=build /srv/runos/web/.next/static ./.next/static
COPY --from=build /srv/runos/web/public ./public
EXPOSE 4000
CMD ["node", "server.js"]
