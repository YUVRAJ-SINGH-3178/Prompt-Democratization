"""Check the local demo credentials through the browser UI without printing them."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "university": ("/admin", "University overview"),
    "coordinator.a": ("/coordinator", "Coordinator overview"),
    "faculty.a": ("/faculty", "Prompt builder"),
}


def main() -> None:
    path = ROOT / "credentials.local.txt"
    if not path.exists():
        raise SystemExit("credentials.local.txt is missing")
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()[2:] if line]
    credentials = {row[0]: row[2] for row in rows if len(row) == 3}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        failures = []
        for username, (route, title) in EXPECTED.items():
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            responses = []

            def capture_response(response):
                name = response.url.split("?")[0].rsplit("/", 1)[-1]
                detail = response.status
                if name == "me":
                    try:
                        detail = response.json().get("authenticated", False)
                    except Exception:
                        pass
                responses.append((name, detail))

            page.on("response", capture_response)
            page.on("pageerror", lambda error: responses.append(("pageerror", str(error))))
            page.goto("http://127.0.0.1:8000", wait_until="networkidle")
            page.locator("#username").fill(username)
            page.locator("#password").fill(credentials[username])
            page.get_by_role("button", name="Sign in").click()
            try:
                page.wait_for_function("route => location.pathname === route", arg=route, timeout=8000)
                page.get_by_role("heading", name=title, exact=True).wait_for(timeout=8000)
                print(f"{username}: sign-in and {route} dashboard passed")
            except Exception:
                message = (
                    page.locator("#loginError").inner_text()
                    if page.locator("#loginError").count()
                    else "dashboard did not load"
                )
                cookie_names = [cookie["name"] for cookie in page.context.cookies()]
                print(
                    f"{username}: sign-in or dashboard failed ({message or 'dashboard did not load'}); path={page.url}; cookies={cookie_names}; responses={responses}"
                )
                failures.append(username)
            page.close()
        browser.close()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
