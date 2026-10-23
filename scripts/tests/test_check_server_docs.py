"""Whether `scripts/check_server_docs.py` would notice a server README that has fallen behind the code.

Every case perturbs a baseline the checker accepts, so that a passing case shows the checker can see the
difference rather than that it has nothing to report.
"""

from scripts.check_server_docs import problems, routes

APP = '''
@app.route("/health")
def health(): ...

@app.route("/api/things/info", methods=["GET"])
def things_info(): ...

@app.route("/api/things/make", methods=["POST"])
def things_make(): ...
'''

CLIENT = '__all__ = ["things_info", "things_make"]\n'

README = '''\
# Python bindings

- `things_info`: what the module loaded.
- `things_make`: make a thing.
- `render`: a method of a GUI class, which is no export and is fine here.

# Web API endpoints

### GET "/health"

### GET "/api/things/info"

### POST "/api/things/make"
'''


def test_the_baseline_is_accepted():
    assert problems(APP, README, [CLIENT]) == []


def test_a_route_with_no_method_named_is_a_get():
    assert ("GET", "/health") in routes(APP)


def test_an_undocumented_endpoint_is_reported():
    app = APP + '\n@app.route("/api/things/undo", methods=["POST"])\ndef undo(): ...\n'
    assert problems(app, README, [CLIENT]) == ['endpoint not documented: POST "/api/things/undo"']


def test_an_endpoint_documented_under_the_wrong_path_is_reported_both_ways():
    readme = README.replace('### POST "/api/things/make"', '### POST "/api/thing/make"')
    assert problems(APP, readme, [CLIENT]) == ['endpoint not documented: POST "/api/things/make"',
                                               'documented endpoint does not exist: POST "/api/thing/make"']


def test_an_endpoint_documented_under_the_wrong_method_is_reported():
    readme = README.replace('### POST "/api/things/make"', '### GET "/api/things/make"')
    assert len(problems(APP, readme, [CLIENT])) == 2


def test_a_client_function_missing_from_the_list_is_reported():
    client = '__all__ = ["things_info", "things_make", "things_forget"]\n'
    assert problems(APP, README, [client]) == ["client function not in the Python bindings list: things_forget"]


def test_a_heading_outside_the_endpoints_section_does_not_count():
    readme = README.replace('### GET "/health"\n', "").replace("# Python bindings\n",
                                                                '# Python bindings\n\n### GET "/health"\n')
    assert problems(APP, readme, [CLIENT]) == ['endpoint not documented: GET "/health"']
