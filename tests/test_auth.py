async def register(client, email="a@example.com", password="password123"):
    return await client.post("/register", data={"email": email, "password": password})


async def test_register_sets_cookie_and_redirects(client):
    response = await register(client)
    assert response.status_code == 204
    assert response.headers["HX-Redirect"] == "/app"
    assert "access_token" in response.cookies


async def test_register_rejects_short_password(client):
    response = await client.post("/register", data={"email": "b@example.com", "password": "short"})
    assert response.status_code == 200
    assert "at least 8 characters" in response.text


async def test_register_rejects_duplicate_email(client):
    await register(client)
    response = await register(client)
    assert "already exists" in response.text


async def test_login_with_wrong_password_fails(client):
    await register(client)
    response = await client.post(
        "/login", data={"email": "a@example.com", "password": "wrongpassword"}
    )
    assert "Invalid email or password" in response.text


async def test_login_success(client):
    await register(client)
    # Clear the cookie set by registration to test login independently.
    client.cookies.clear()
    response = await client.post(
        "/login", data={"email": "a@example.com", "password": "password123"}
    )
    assert response.status_code == 204
    assert "access_token" in response.cookies


async def test_dashboard_requires_auth(client):
    response = await client.get("/app")
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


async def test_dashboard_accessible_when_authenticated(client):
    await register(client)
    response = await client.get("/app")
    assert response.status_code == 200
    assert "Your documents" in response.text
