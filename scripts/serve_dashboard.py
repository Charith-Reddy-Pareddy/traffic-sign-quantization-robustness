"""Serve the saved dashboard for a local browser preview."""

from argparse import ArgumentParser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "docs" / "dashboard"


def create_server(port: int = 8000, dashboard_dir: Path = DASHBOARD_DIR):
    root = Path(dashboard_dir).resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError(f"Dashboard page not found: {root / 'index.html'}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main() -> None:
    parser = ArgumentParser(description="Preview the saved traffic-sign dashboard locally")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = create_server(args.port)
    print(f"Dashboard preview: http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
