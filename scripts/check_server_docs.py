#!/usr/bin/env python
"""Check that the server README documents every web API endpoint and every client function.

`raven/server/README.md` has two reference sections that repeat the code in prose: one section per web API
endpoint, and a list of the Python client's functions. Nothing else in the toolchain reads them, so they drift
silently: when this was written, twelve endpoints and twenty client functions were missing, one endpoint was
documented under a path that did not exist, and both sections carried a disclaimer saying they were maintained
"on a best-effort basis".

Two checks:

- **Endpoints, both ways.** Every `@app.route` in `raven/server/app.py` must have a `### METHOD "/path"` section,
  and every such section must name a route that exists.
- **Client functions, one way.** Every name exported by `raven.client.api` and `raven.client.tts` must be
  mentioned, in backticks, in the Python bindings section. Not the reverse: that section also documents the
  DPG renderer's and the avatar controller's methods, which are not exports of either module.

It checks completeness, not the prose: a section can be present and wrong. The routes' docstrings stay the
authority on what each endpoint does.

Reads the source; imports nothing from Raven.
"""

import ast
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
APP = REPO / "raven" / "server" / "app.py"
README = REPO / "raven" / "server" / "README.md"
CLIENT_MODULES = (REPO / "raven" / "client" / "api.py", REPO / "raven" / "client" / "tts.py")

ENDPOINTS_HEADING = "# Web API endpoints"
BINDINGS_HEADING = "# Python bindings"


def routes(source: str) -> set[tuple[str, str]]:
    """`(method, path)` for every `@app.route` in `source`; a route naming no method is a GET, as in Flask."""
    out = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", None) == "route"):
                continue
            path = decorator.args[0].value
            methods = next((keyword.value for keyword in decorator.keywords if keyword.arg == "methods"), None)
            for method in ([element.value for element in methods.elts] if methods is not None else ["GET"]):
                out.add((method, path))
    return out


def documented_routes(readme: str) -> set[tuple[str, str]]:
    """`(method, path)` for every `### METHOD "/path"` heading in the endpoints section."""
    section = readme[readme.index(ENDPOINTS_HEADING):]
    return set(re.findall(r'^### ([A-Z]+) "([^"]+)"', section, re.MULTILINE))


def exports(sources: list[str]) -> set[str]:
    """Every name in the `__all__` of each of `sources`."""
    out = set()
    for source in sources:
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "__all__" for target in node.targets):
                out |= {element.value for element in node.value.elts}
    return out


def mentioned(readme: str) -> set[str]:
    """Every backquoted identifier in the Python bindings section."""
    section = readme[readme.index(BINDINGS_HEADING):readme.index(ENDPOINTS_HEADING)]
    return set(re.findall(r"`(\w+)`", section))


def problems(app_source: str, readme: str, client_sources: list[str]) -> list[str]:
    """What is wrong, one line each; empty when the README is complete."""
    found = []
    in_code, in_readme = routes(app_source), documented_routes(readme)
    found += [f'endpoint not documented: {method} "{path}"' for method, path in sorted(in_code - in_readme,
                                                                                         key=lambda route: route[1])]
    found += [f'documented endpoint does not exist: {method} "{path}"' for method, path in
              sorted(in_readme - in_code, key=lambda route: route[1])]
    found += [f"client function not in the Python bindings list: {name}" for name in
              sorted(exports(client_sources) - mentioned(readme))]
    return found


def main() -> int:
    readme = README.read_text(encoding="utf-8")
    found = problems(APP.read_text(encoding="utf-8"), readme,
                     [path.read_text(encoding="utf-8") for path in CLIENT_MODULES])
    if not found:
        n_routes = len(routes(APP.read_text(encoding="utf-8")))
        n_exports = len(exports([path.read_text(encoding="utf-8") for path in CLIENT_MODULES]))
        print(f"OK: the server README documents all {n_routes} endpoints and all {n_exports} client functions.")
        return 0
    print(f"{README.relative_to(REPO)}:")
    for line in found:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
