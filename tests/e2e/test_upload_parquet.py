"""
E2E browser test: uploads a tiny Quest 3 Parquet log and checks JSON inference
output.  Skips automatically when `OMNI_UI_URL` is undefined (e.g. headless CI).
"""
import os, json, numpy as np, pandas as pd, pytest
from playwright.sync_api import sync_playwright

@pytest.mark.skipif(
    "OMNI_UI_URL" not in os.environ, reason="UI server URL not provided"
)
def test_upload_parquet(tmp_path):
    # --- craft a 10‑frame Parquet file on‑the‑fly -------------------------
    df = pd.DataFrame(
        [
            {
                "timestamp_ns": i * 8_333_333,
                "left_gaze_yaw": np.random.rand(),
                "left_gaze_pitch": np.random.rand(),
                "fix_conf": 0.8,
                "hand_pose_0_x": np.random.rand(),
            }
            for i in range(10)
        ]
    )
    parquet = tmp_path / "tiny.parquet"
    df.to_parquet(parquet, index=False)

    # --- drive the UI -----------------------------------------------------
    ui_url = os.environ["OMNI_UI_URL"]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(ui_url)

        # upload file
        page.set_input_files('input[type="file"]', str(parquet))

        # wait for JSON blob with tensor shapes
        page.wait_for_selector("#inference-json")
        payload = json.loads(page.inner_text("#inference-json"))

        # basic sanity checks
        assert payload["input_shapes"]["gaze"][-1] == 3      # feature-dim
        assert payload["input_shapes"]["hand_pose"][-1] == 1 # feature-dim
        browser.close()
