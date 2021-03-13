test:
	RUST_BACKTRACE=1 cargo test --all-features -- --nocapture --test-threads 1

build:
	cargo test  --all-features

lint:
	touch src/lib.rs
	cargo check
	cargo clippy

clean:
	cargo clean

run:
	cargo run --example simple

doc:
	cargo doc --open

example:
	cargo run --example simple

fmt:
	cargo-fmt
