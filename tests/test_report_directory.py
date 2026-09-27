from __future__ import annotations

import http.client
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import web_app


class ReportDirectoryTests(unittest.TestCase):
    def test_native_opener_uses_only_the_completed_jobs_report_folder(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "reports"
            directory = output / "aaaaaaaaaaaa"
            directory.mkdir(parents=True)
            (directory / "report.html").write_text("report", encoding="utf-8")
            job = web_app.Job(id="aaaaaaaaaaaa", kind="recognition", title="报告", status="succeeded", output_dir=str(directory))
            with patch.object(web_app, "OUTPUT_ROOT", output), patch.object(web_app.sys, "platform", "win32"), patch.object(web_app.os, "startfile", create=True) as opener:
                self.assertEqual(web_app.open_report_directory(job), directory.resolve())
                opener.assert_called_once_with(str(directory.resolve()))
                opener.reset_mock()
                job.status = "running"
                with self.assertRaises(RuntimeError):
                    web_app.open_report_directory(job)
                job.status = "succeeded"
                job.output_dir = str(root)
                with self.assertRaises(PermissionError):
                    web_app.open_report_directory(job)
                job.output_dir = str(directory)
                (directory / "report.html").unlink()
                with self.assertRaises(FileNotFoundError):
                    web_app.open_report_directory(job)
                opener.assert_not_called()

    def test_http_action_requires_local_same_origin_post_and_server_job_id(self):
        registry = web_app.JobRegistry()
        job = web_app.Job(id="aaaaaaaaaaaa", kind="recognition", title="报告", status="succeeded", output_dir="data/outputs/web/aaaaaaaaaaaa")
        registry._jobs[job.id] = job
        with patch.object(web_app, "execution_runtime_status", return_value={}):
            server = web_app.PrismServer(("127.0.0.1", 0), registry)
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        host = f"127.0.0.1:{server.server_port}"
        url = job.public()["open_directory_url"]

        def request(method="POST", headers=None, path=url):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                connection.request(method, path, body="{}", headers=headers or {})
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        try:
            good = {"Content-Type": "application/json", "X-Prism-Local-Action": "open-directory", "Origin": f"http://{host}"}
            with patch.object(web_app, "open_report_directory", return_value=Path(job.output_dir)) as opener:
                self.assertEqual(request(headers=good)[0], 200)
                opener.assert_called_once_with(job)
                opener.reset_mock()
                self.assertEqual(request(method="GET", headers=good)[0], 404)
                self.assertEqual(request()[0], 403)
                self.assertEqual(request(headers={**good, "Origin": "https://example.com"})[0], 403)
                self.assertEqual(request(headers={**good, "Host": "example.com", "Origin": "http://example.com"})[0], 403)
                self.assertEqual(request(headers=good, path="/api/jobs/bbbbbbbbbbbb/open-directory")[0], 404)
                opener.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
