from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from scripts import web_app


class WebAppTests(unittest.TestCase):
    def test_build_recognition_command_contains_selected_options(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.png"
            output = root / "result"
            command = web_app.build_recognition_command(
                python_executable="python",
                image_path=image,
                output_dir=output,
                slots=["HEAD", "BODY"],
                top_k=20,
                gender="female",
                profile="current-best",
                device="cuda",
            )

        self.assertEqual(command[0], "python")
        self.assertEqual(command[2], str(image))
        self.assertEqual(
            command[command.index("--slots") + 1 : command.index("--top-k")],
            ["HEAD", "BODY"],
        )
        self.assertEqual(command[command.index("--top-k") + 1], "20")
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(command[-2:], ["--gender", "female"])

    def test_build_recognition_command_rejects_unknown_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "CPU 或 GPU"):
            web_app.build_recognition_command(
                python_executable="python",
                image_path=Path("source.png"),
                output_dir=Path("result"),
                slots=["BODY"],
                top_k=10,
                gender=None,
                profile="current-best",
                device="auto",
            )

    def test_cpu_execution_hides_cuda_from_child_process(self) -> None:
        self.assertEqual(
            web_app.execution_environment_overrides("cpu"),
            {"CUDA_VISIBLE_DEVICES": "-1"},
        )
        self.assertEqual(web_app.execution_environment_overrides("cuda"), {})

    def test_gpu_selection_requires_available_cuda_runtime(self) -> None:
        runtime = {
            "gpu": {
                "available": False,
                "reason": "测试环境没有 CUDA。",
            }
        }
        with self.assertRaisesRegex(ValueError, "environment-gpu.yml"):
            web_app.validate_execution_device("cuda", runtime)
        self.assertEqual(web_app.validate_execution_device("cpu", runtime), "cpu")

    def test_runtime_probe_reports_cpu_and_gpu_options(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.11.0+cu130",
            get_num_threads=lambda: 12,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _index: "Test GPU",
            ),
        )
        with (
            mock.patch.dict("sys.modules", {"torch": fake_torch}),
            mock.patch.object(web_app.os, "cpu_count", return_value=20),
        ):
            runtime = web_app._probe_execution_runtime("python")

        self.assertTrue(runtime["cpu"]["available"])
        self.assertTrue(runtime["gpu"]["available"])
        self.assertEqual(runtime["gpu"]["device_name"], "Test GPU")
        self.assertEqual(runtime["logical_processors"], 20)

    def test_loading_weights_progress_replaces_the_same_log_line(self) -> None:
        job = web_app.Job(id="test", kind="recognition", title="测试")

        web_app._append_job_log(job, "stage=retrieve slot=BODY")
        web_app._append_job_log(job, "Loading weights: 0%| | 0/201")
        web_app._append_job_log(job, "Loading weights: 42%| | 85/201")
        web_app._append_job_log(job, "Loading weights: 100%|#| 201/201")

        self.assertEqual(
            job.logs,
            [
                "stage=retrieve slot=BODY",
                "Loading weights: 100%|#| 201/201",
            ],
        )

    def test_loading_weights_progress_keeps_separate_model_loads(self) -> None:
        job = web_app.Job(id="test", kind="recognition", title="测试")

        for line in (
            "Loading weights: 0%| | 0/201",
            "Loading weights: 100%|#| 201/201",
            "Loading weights: 0%| | 0/197",
            "Loading weights: 100%|#| 197/197",
        ):
            web_app._append_job_log(job, line)

        self.assertEqual(
            job.logs,
            [
                "Loading weights: 100%|#| 201/201",
                "Loading weights: 100%|#| 197/197",
            ],
        )

    def test_frontend_offers_top_20_and_replacement_drop_target(self) -> None:
        html = (web_app.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (web_app.WEB_ROOT / "app.js").read_text(encoding="utf-8")
        stylesheet = (web_app.WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn('<option value="20">20 件</option>', html)
        self.assertIn("[elements.dropZone, elements.selectedFile]", javascript)
        self.assertIn("candidate.is_no_equipment", javascript)
        self.assertIn("不显示外观", javascript)
        self.assertIn(".candidate.no-equipment", stylesheet)
        self.assertIn('name="device" value="cuda"', html)
        self.assertIn('name="device" value="cpu"', html)
        self.assertIn('form.append("device", selectedDevice())', javascript)
        self.assertIn("renderExecutionRuntime(payload.execution)", javascript)
        self.assertIn(".device-option-card", stylesheet)

    def test_frontend_accepts_clipboard_image_paste(self) -> None:
        html = (web_app.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (web_app.WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("按 Ctrl+V 粘贴", html)
        self.assertIn('document.addEventListener("paste"', javascript)
        self.assertIn("event.clipboardData", javascript)
        self.assertIn('item.type.startsWith("image/")', javascript)
        self.assertIn("selectImage(file)", javascript)

    def test_detect_image_extension(self) -> None:
        cases = [
            ("image.png", b"\x89PNG\r\n\x1a\nrest", ".png"),
            ("image.jpeg", b"\xff\xd8\xffrest", ".jpg"),
            ("image.webp", b"RIFF\x04\x00\x00\x00WEBPrest", ".webp"),
        ]
        for filename, content, expected in cases:
            with self.subTest(filename=filename):
                self.assertEqual(
                    web_app.detect_image_extension(filename, content),
                    expected,
                )

    def test_detect_image_extension_rejects_mismatched_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "实际格式"):
            web_app.detect_image_extension("image.png", b"\xff\xd8\xffrest")

    def test_parse_multipart_form_extracts_fields_and_file(self) -> None:
        boundary = "PrismBoundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="slots"\r\n\r\n'
            "HEAD,BODY\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; filename="look.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8")
        body += b"\x89PNG\r\n\x1a\nrest\r\n"
        body += f"--{boundary}--\r\n".encode("utf-8")

        fields, files = web_app.parse_multipart_form(
            f"multipart/form-data; boundary={boundary}",
            body,
        )

        self.assertEqual(fields, {"slots": "HEAD,BODY"})
        self.assertEqual(files["image"][0:2], ("look.png", "image/png"))
        self.assertTrue(files["image"][2].startswith(b"\x89PNG"))

    def test_resolve_served_file_only_allows_web_outputs(self) -> None:
        job_dir = web_app.OUTPUT_ROOT / "test-safe-path"
        job_dir.mkdir(parents=True, exist_ok=True)
        image = job_dir / "preview.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        try:
            self.assertEqual(
                web_app.resolve_served_file(str(image)),
                image.resolve(),
            )
            with self.assertRaises(PermissionError):
                web_app.resolve_served_file(
                    str(web_app.PROJECT_ROOT / "README.md")
                )
        finally:
            image.unlink()
            job_dir.rmdir()

    def test_build_dataset_command_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "模式无效"):
            web_app.build_dataset_command(
                python_executable="python",
                mode="unknown",
            )


if __name__ == "__main__":
    unittest.main()
