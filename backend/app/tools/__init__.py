"""Tool modules. Call `load_all()` once at startup to register every tool with `registry`.

Registration isn't done on import of this package, so modules like workspace.py can import
`app.tools.registry` without pulling in every tool (which would be a circular import).
"""


def load_all() -> None:
    from app.tools import build_tools, code_tools, deploy_tools, docker_tools, run_tools  # noqa: F401  (importing registers the tools)
