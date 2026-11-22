from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
SHOTS = ARTIFACTS / "screenshots"
DB = ARTIFACTS / "browser-auth.sqlite"
CREDS = ARTIFACTS / "browser-auth-credentials.txt"
ROLES = (
    (
        "university",
        "University",
        "/admin",
        "University overview",
        [
            "Overview",
            "Accounts",
            "Catalog",
            "Templates",
            "Prompt checker",
            "Sample library",
            "Accuracy",
            "Audit log",
            "Trash",
        ],
    ),
    (
        "coordinator.a",
        "Course Coordinator",
        "/coordinator",
        "Coordinator overview",
        ["Overview", "Courses", "Faculty", "Templates", "Prompt checker", "Sample library", "Usage export"],
    ),
    (
        "faculty.a",
        "Faculty",
        "/faculty",
        "Prompt builder",
        ["Home and builder", "Prompt checker", "Sample library", "History"],
    ),
)
PAGE_KEYS = {
    "Overview": "overview",
    "Accounts": "accounts",
    "Catalog": "catalog",
    "Templates": "templates",
    "Prompt checker": "checker",
    "Sample library": "samples",
    "Accuracy": "accuracy",
    "Audit log": "audit",
    "Trash": "trash",
    "Courses": "courses",
    "Faculty": "faculty",
    "Usage export": "export",
    "Home and builder": "overview",
    "History": "history",
}


def prepare_auth():
    os.environ["PROMPT_LAB_AUTH_DB"] = str(DB)
    from scripts.seed_demo_users import seed

    if DB.exists():
        DB.unlink()
    CREDS.unlink(missing_ok=True)
    seed(credentials_path=CREDS)
    passwords = {}
    for line in CREDS.read_text(encoding="utf-8").splitlines()[2:]:
        username, _, password = line.split("\t")
        passwords[username] = password
    return passwords


async def main():
    ARTIFACTS.mkdir(exist_ok=True)
    SHOTS.mkdir(exist_ok=True)
    passwords = prepare_auth()
    env = os.environ.copy()
    env["PROMPT_LAB_AUTH_DB"] = str(DB)
    env["PROMPT_LAB_DB"] = str(ARTIFACTS / "browser-real.sqlite")
    env["SIMULATED_DB_PATH"] = str(ARTIFACTS / "browser-simulated.sqlite")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8765"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    errors, overflow, screens = [], [], []
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Local UI test server did not start")

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            for width, size_label in ((1440, "desktop"), (400, "mobile")):
                for theme in ("dark", "light"):
                    for username, role, dashboard, landing_title, pages in ROLES:
                        context = await browser.new_context(
                            viewport={"width": width, "height": 980}, device_scale_factor=1
                        )
                        page = await context.new_page()
                        auth_responses = []
                        page.on("pageerror", lambda exc: errors.append(str(exc)))
                        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
                        page.on(
                            "response",
                            lambda response: auth_responses.append((response.url.rsplit("/", 1)[-1], response.status))
                            if "/api/auth/" in response.url or response.url.endswith(dashboard)
                            else None,
                        )
                        await page.goto("http://127.0.0.1:8765", wait_until="networkidle")
                        await page.evaluate(
                            "t => {document.documentElement.dataset.theme=t; localStorage.setItem('prl-theme',t)}",
                            theme,
                        )
                        await page.locator("#username").fill(username)
                        await page.locator("#password").fill(passwords[username])
                        await page.get_by_role("button", name="Sign in").click()
                        try:
                            await page.wait_for_function(
                                "target => location.pathname === target", arg=dashboard, timeout=12000
                            )
                            await page.get_by_role("heading", name=landing_title, exact=True).wait_for(timeout=8000)
                        except Exception as exc:
                            login_error = (
                                await page.locator("#loginError").inner_text()
                                if await page.locator("#loginError").count()
                                else "none"
                            )
                            title = (
                                await page.locator("#pageTitle").inner_text()
                                if await page.locator("#pageTitle").count()
                                else "missing"
                            )
                            cookie_names = [cookie["name"] for cookie in await context.cookies()]
                            raise RuntimeError(
                                f"Dashboard did not render for {role}: url={page.url}, title={title}, login_error={login_error}, cookies={cookie_names}, responses={auth_responses}, browser_errors={errors}"
                            ) from exc
                        for name in pages:
                            await page.locator(f'#roleNav button[data-page="{PAGE_KEYS[name]}"]').click()
                            section_title = {
                                "Overview": "Repository snapshot"
                                if role == "University"
                                else "Scope"
                                if role == "Course Coordinator"
                                else "Build a prompt",
                                "Home and builder": "Build a prompt",
                                "Accounts": "Local accounts",
                                "Catalog": "Record type",
                                "Templates": "Templates",
                                "Courses": "Course cohort mappings",
                                "Faculty": "Assigned faculty",
                                "Usage export": "Coordinator export",
                                "History": "Recent checks",
                                "Audit log": "Recent events",
                                "Trash": "Records",
                                "Accuracy": "Import one annotator record",
                            }.get(name)
                            if name == "Catalog":
                                await page.locator("#catalogKind").wait_for(timeout=8000)
                            elif section_title:
                                await (
                                    page.locator("#pageBody")
                                    .get_by_role("heading", name=section_title, exact=True)
                                    .wait_for(timeout=8000)
                                )
                            if name == "Prompt checker":
                                await page.locator("#promptText").fill(
                                    "Create five multiple choice questions about databases. Give four options A to D, mark the correct answer, explain the rationale, and order by difficulty."
                                )
                                await page.get_by_role("button", name="Check prompt").click()
                                await page.get_by_text("Highlighted evidence", exact=True).wait_for(timeout=5000)
                                await page.get_by_text("Slot completeness", exact=True).wait_for(timeout=5000)
                            if name == "Sample library":
                                await (
                                    page.locator("#pageTitle")
                                    .get_by_text("Sample library", exact=True)
                                    .wait_for(timeout=5000)
                                )
                            if name == "Accuracy":
                                await page.get_by_text("awaiting annotations", exact=False).wait_for(timeout=5000)
                            dimensions = await page.evaluate(
                                "({body:document.documentElement.scrollWidth, viewport:window.innerWidth})"
                            )
                            if dimensions["body"] > dimensions["viewport"]:
                                overflow.append(
                                    {"role": role, "page": name, "theme": theme, "width": width, **dimensions}
                                )
                            slug = name.lower().replace(" ", "-")
                            path = SHOTS / f"{role.lower().replace(' ', '-')}-{slug}-{size_label}-{theme}.png"
                            await page.screenshot(path=str(path), full_page=True)
                            screens.append(str(path.relative_to(ROOT)))
                            if name == "Sample library" and width == 1440 and theme == "dark":
                                await page.get_by_role("button", name="Open in checker").first.click()
                                await page.locator("#checkerResult .pill.sample").wait_for(timeout=5000)
                                sample_path = SHOTS / f"{role.lower().replace(' ', '-')}-sample-open-in-checker.png"
                                await page.screenshot(path=str(sample_path), full_page=True)
                                screens.append(str(sample_path.relative_to(ROOT)))
                        if username == "faculty.a":
                            # Verify the course-to-topic cascade and build a prompt through the faculty flow.
                            await page.locator('#roleNav button[data-page="overview"]').click()
                            course = page.locator("#builderForm select[name=course_id]")
                            await course.wait_for(timeout=5000)
                            await page.locator("#builderForm select[name=topic_ids]").wait_for(timeout=5000)
                            await page.locator("#builderForm select[name=topic_ids] option").first.wait_for(
                                timeout=5000
                            )
                            await page.locator("#builderForm select[name=topic_ids] option").first.evaluate(
                                "option => option.selected = true"
                            )
                            await page.locator("#builderForm button").click()
                            await page.get_by_text("Role: faculty", exact=False).wait_for(timeout=8000)
                        await context.close()
            await browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    report = {
        "roles": [r[1] for r in ROLES],
        "screenshots": len(screens),
        "screen_directory": str(SHOTS.relative_to(ROOT)),
        "console_or_page_errors": errors,
        "horizontal_overflow": overflow,
        "checker_and_sample_flows": "passed" if screens else "not run",
    }
    (ARTIFACTS / "browser-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors or overflow:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
