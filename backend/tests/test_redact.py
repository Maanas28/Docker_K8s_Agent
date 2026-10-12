from app.redact import MASK, redact


def test_quoted_secrets_in_code():
    text, n = redact('API_KEY = "abc123secret"\nclient_secret: \'topsecret\'\n"password": "hunter22"')
    assert "abc123secret" not in text and "topsecret" not in text and "hunter22" not in text
    assert n == 3 and 'API_KEY = "[REDACTED]"' in text


def test_code_that_only_mentions_secret_names_is_kept():
    code = "api_key: str | None = None\npassword = get_password()\nif settings.api_key:\n"
    assert redact(code, "app/config.py") == (code, 0)


def test_config_files_mask_unquoted_values():
    text, n = redact("DB_PASSWORD=abc\nAPP_PORT=8080\npassword: letmein\n", "config.yaml")
    assert "abc" not in text and "letmein" not in text and "APP_PORT=8080" in text
    assert n == 2


def test_known_token_formats_and_urls():
    text, _ = redact(
        "key AKIAABCDEFGHIJKLMNOP gh ghp_" + "a" * 36 + "\n"
        "db postgres://admin:s3cr3t@db:5432/app\n"
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop"
    )
    assert "AKIA" not in text and "ghp_" not in text and "s3cr3t" not in text and "eyJ" not in text
    assert "postgres://admin:" + MASK + "@db:5432/app" in text


def test_private_key_block():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nabc\n-----END RSA PRIVATE KEY-----"
    text, n = redact(f"before\n{pem}\nafter")
    assert "MIIEow" not in text and n == 1 and "before" in text and "after" in text
