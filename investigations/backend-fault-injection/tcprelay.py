"""A byte-level TCP relay, so "what happens when a server goes away" is testable at a chosen moment.

Point an app at this instead of at the real server, and killing this process is the server vanishing —
timed to the second, with nothing to restart afterwards but the relay. The real server keeps running, which
matters when it holds several gigabytes of models that take minutes to load.

Usage::

    python tcprelay.py --port 8999 --upstream localhost:5100
    raven-librarian --server-url http://127.0.0.1:8999

Then, to make the server disappear::

    pkill -f 'tcprelay.py --port 8999'    # careful: match on the port, not on the bare script name

...and to bring it back, run the same command again.

**Why a socket relay rather than `faultproxy.py`.** That one speaks HTTP, and speaking HTTP means rebuilding
each request and response from parts — which loses every header it does not know to copy. Raven-server puts
real data in headers (`X-Full-Size` and `X-Crop` on avatar frames, word timestamps and phonemes on TTS
audio), so an HTTP proxy in front of it is transparent only for the endpoints somebody remembered. This one
copies bytes and cannot know what they mean, which is exactly the property wanted here.

The trade is the other way round for *content* faults: this cannot answer an SSE error or stall one endpoint,
because it does not know where one request ends. Use `faultproxy.py` for those and this for "the server is
not there".
"""

import argparse
import socket
import threading

def pump(source: socket.socket, sink: socket.socket) -> None:
    """Copy bytes from `source` to `sink` until either end is done, then close the write side."""
    try:
        while True:
            data = source.recv(65536)
            if not data:
                break
            sink.sendall(data)
    except OSError:  # the other end went away mid-copy, which is the normal way a connection ends here
        pass
    finally:
        # Half-close rather than close: the other direction may still have data to deliver, and a stream
        # the app is reading should end tidily rather than as a reset.
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass

def serve_one(client: socket.socket, upstream_host: str, upstream_port: int) -> None:
    """Connect to the upstream and pump both directions until the connection ends."""
    try:
        server = socket.create_connection((upstream_host, upstream_port))
    except OSError:
        client.close()
        return
    with client, server:
        threads = [threading.Thread(target=pump, args=(client, server), daemon=True),
                   threading.Thread(target=pump, args=(server, client), daemon=True)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8999, help="port to listen on")
    parser.add_argument("--upstream", default="localhost:5100", help="host:port of the real server")
    args = parser.parse_args()

    upstream_host, _, upstream_port = args.upstream.partition(":")
    upstream_port = int(upstream_port or 80)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # So that restarting the relay right after killing it does not wait out TIME_WAIT — bringing the server
    # "back" is half the test, and a minute of "address already in use" is a poor way to run it.
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(64)
    print(f"[tcprelay] {args.upstream} -> 127.0.0.1:{args.port}", flush=True)

    try:
        while True:
            client, _ = listener.accept()
            threading.Thread(target=serve_one, args=(client, upstream_host, upstream_port), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()

if __name__ == "__main__":
    main()
