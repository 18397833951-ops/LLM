import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from test_alignment import configuration
from src.web_app import DemoService, handler_for


class WebTests(unittest.TestCase):
    def test_report_marks_missing_results(self):
        from src.report_tasks import export_report
        with tempfile.TemporaryDirectory(dir="work") as directory:
            cfg = configuration()
            for key in ("training_dir", "attention_dir", "science_dir"):
                cfg["web"][key] = str(Path(directory).resolve() / key)
            cfg["alignment"]["output_dir"] = str(Path(directory).resolve() / "alignment")
            result = export_report(cfg)
            content = Path(result["report"]).read_text(encoding="utf-8")
            self.assertFalse(result["full_results_available"])
            self.assertIn("Post-training has not produced", content)
            self.assertNotIn("0.8385", content)

    def test_empty_results_http_static_and_request_boundaries(self):
        with tempfile.TemporaryDirectory(dir="work") as directory:
            cfg = configuration()
            for key in ("training_dir", "attention_dir", "science_dir"):
                cfg["web"][key] = str(Path(directory) / key)
            cfg["alignment"].update(output_dir=str(Path(directory) / "alignment"), checkpoint=str(Path(directory) / "absent.pt"))
            service = DemoService(cfg)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                for path in ("/", "/style.css", "/app.js"):
                    with urlopen(url + path) as response:
                        self.assertEqual(response.status, 200)
                        self.assertGreater(len(response.read()), 100)
                summary = json.load(urlopen(url + "/api/summary"))
                self.assertFalse(summary["training"]["available"])
                self.assertFalse(summary["models"]["base"])
                for path in ("/asset?key=../../secret", "/../requirements.txt"):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(url + path)
                    self.assertEqual(error.exception.code, 404)
                bad = Request(url + "/api/generate", data=b'{}', headers={"Content-Type":"application/json", "Origin":"http://elsewhere.example"})
                with self.assertRaises(HTTPError) as error:
                    urlopen(bad)
                self.assertEqual(error.exception.code, 403)
                with patch.object(service, "generate", return_value=dict(text="Actual test response", new_tokens=3)):
                    request = Request(url + "/api/generate", data=b'{"model":"base"}', headers={"Content-Type":"application/json"})
                    self.assertEqual(json.load(urlopen(request))["text"], "Actual test response")
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
