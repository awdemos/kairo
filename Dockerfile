FROM rust:slim-bookworm AS builder
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/kairo
COPY . .
RUN cargo build --release -p kairo-cli

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/kairo/target/release/kairo-cli /usr/local/bin/kairo
EXPOSE 3000
ENV RUST_LOG=info
ENTRYPOINT ["kairo"]
CMD ["server", "--port", "3000"]
